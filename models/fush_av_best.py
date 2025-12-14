import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_zoo.S3D import S3D, load_S3D_weight
from models.model_zoo.MobileNetV2 import load_MobileNetV2_weight
from transformer.SubLayers import MultiHeadAttention
from Transformer_tools.blocks.encoder_layer import EncoderLayer
from Transformer_tools.embedding.positional_hyl import PostionalEncoding
import pdb

# =================================================================
# 1. 基础模块 (保持高效配置)
# =================================================================

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class CCGLU(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(CCGLU, self).__init__()
        self.proj = nn.Linear(in_features, out_features * 2) 
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return self.dropout(x * self.act(gate))

class SMFA(nn.Module):
    def __init__(self, dim, reduction=4):
        super(SMFA, self).__init__()
        self.proj_down = nn.Linear(dim, dim // reduction)
        self.act = nn.GELU()
        self.dw_conv = nn.Conv1d(dim // reduction, dim // reduction, kernel_size=3, padding=1, groups=dim // reduction)
        self.proj_up = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        residual = x
        x = self.proj_down(x)
        x = self.act(x)
        x = x.permute(0, 2, 1)
        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)
        x = self.proj_up(x)
        gate = self.sigmoid(x)
        return residual * gate

# =================================================================
# 2. ### [修改] 新增模块：交叉生成器 & 重建解码器
# =================================================================

class CrossGenerator(nn.Module):
    """
    A2V / V2A 网络: 单层 MLP + 单 Transformer 块
    """
    def __init__(self, dim, n_head=4, drop_prob=0.1):
        super(CrossGenerator, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            Mish(),
            nn.Linear(dim, dim)
        )
        self.transformer_block = EncoderLayer(d_model=dim, ffn_hidden=dim*2, n_head=n_head, drop_prob=drop_prob)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [B, N, C]
        x = self.mlp(x)
        x = self.transformer_block(x, s_mask=None)
        return self.out_proj(x)

class ReconstructionDecoder(nn.Module):
    """
    解码器 G_a, G_v: 用于从补全后的特征重建原始特征
    """
    def __init__(self, dim, n_head=4, num_slices=5):
        super(ReconstructionDecoder, self).__init__()
        self.layer = EncoderLayer(d_model=dim, ffn_hidden=dim*2, n_head=n_head, drop_prob=0.1)
        self.head = nn.Linear(dim, dim)
        # 可学习的位置嵌入 pos_p^g
        self.pos_embed = nn.Parameter(torch.randn(1, num_slices, dim) * 0.02)

    def forward(self, x):
        x = x + self.pos_embed
        x = self.layer(x, s_mask=None)
        return self.head(x)

class ChannelGate(nn.Module):
    def __init__(self, input_dim, num_sources=2):
        super(ChannelGate, self).__init__()
        self.num_sources = num_sources
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            Mish(),
            nn.Linear(input_dim // 2, 512 * num_sources)
        )
    def forward(self, x):
        B = x.size(0)
        weights = self.mlp(x).reshape(B, self.num_sources, 512)
        return F.softmax(weights, dim=1)


class Audio_video_Model(nn.Module):
    def __init__(self, audio_frontend, audio_backbone, video_backbone, classes_num, fusion_type, **kwargs):
        super().__init__(**kwargs)
        self.fusion_type = fusion_type
        self.num_class = classes_num
        self.num_slices = 5 ### [修改] 定义切片数量 = 5
        self.video_encoder = video_backbone
        old_pretrained_video_encoder = torch.load('/root/shared-nvme/weights/audio-visual_pretrainedmodel/video_best.pt')['model_state_dict']
        dict_new = self.video_encoder.state_dict().copy()
        pretrained_video_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_video_encoder.items()}
        trained_list = [i for i in pretrained_video_encoder.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_video_encoder[trained_list[i]]
        self.video_encoder.load_state_dict(dict_new)

        self.audio_encoder = audio_backbone
        self.audio_frontend = audio_frontend
        old_pretrained_audio_encoder = torch.load('/root/shared-nvme/weights/audio-visual_pretrainedmodel/audio_best.pt')['model_state_dict']
        dict_new = self.audio_encoder.state_dict().copy()
        dict_new_frontend = self.audio_frontend.state_dict().copy()
        pretrained_audio_encoder = {k.replace('backbone.', ''): v for k,v in old_pretrained_audio_encoder.items()}
        pretrained_audio1_encoder = {k.replace('frontend.', ''): v for k, v in pretrained_audio_encoder.items()}
        pretrained_audio_encoder = {k:v for k, v in pretrained_audio1_encoder.items() if k in dict_new}
        pretrained_frontend = {k: v for k, v in pretrained_audio1_encoder.items() if k in dict_new_frontend}
        trained_list = [i for i in pretrained_audio_encoder.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list)):
            dict_new[trained_list[i]] = pretrained_audio_encoder[trained_list[i]]
        self.audio_encoder.load_state_dict(dict_new)

        trained_list1 = [i for i in pretrained_frontend.keys() if not ('head' in i or 'pos' in i)]
        for i in range(len(trained_list1)):
            dict_new_frontend[trained_list1[i]] = pretrained_frontend[trained_list1[i]]
        self.audio_frontend.load_state_dict(dict_new_frontend)

   # 投影层 (CCGLU)
        self.att_projection_v = CCGLU(1024, 512)
        self.att_projection_a = CCGLU(1024, 512) # 假设 Audio 也是 1024

        # SMFA 增强 (用于特征预处理)
        self.smfa_video = SMFA(512)
        self.smfa_audio = SMFA(512)

        # 位置编码
        self.pos_encoding = PostionalEncoding(d_model=512, max_len=20, device='cuda')

        if self.fusion_type == 'fc':
            # fc fusion of two clipwise_embed
            self.fusion_linear = nn.Linear(1024, self.num_class)

        elif self.fusion_type == 'atten':
            # attn fusion of two framewise_embed
            # n_head = 1
            # d_model = 512
            # d_k = 512
            # d_v = 512
            dropout = 0.1
            self.slf_attn = MultiHeadAttention(n_head=1, d_model=512, d_k=512, d_v=512, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head=8, d_model=512, d_k=64, d_v=64, dropout=dropout)
            self.dec_attn = MultiHeadAttention(n_head=4, d_model=512, d_k=128, d_v=128, dropout=dropout)
            self.fusion_linear1 = nn.Linear(512, 4)

        elif self.fusion_type == 'MBT':
                  # --- [修改] 定义交叉生成网络 ---
            self.gen_a2v = CrossGenerator(dim=512)
            self.gen_v2a = CrossGenerator(dim=512)

            # --- [修改] 定义重建解码器 ---
            self.decoder_a = ReconstructionDecoder(dim=512, num_slices=self.num_slices)
            self.decoder_v = ReconstructionDecoder(dim=512, num_slices=self.num_slices)

            # --- [修改] 融合决策 Gate ---
            # 融合对象：补全后的音频 + 补全后的视频 (全局池化后)
            # 输入维度 = 512 + 512 = 1024
            self.channel_gate = ChannelGate(input_dim=1024, num_sources=2)

            # BN-Neck & Head
            self.bn_neck = nn.BatchNorm1d(512)
            self.bn_neck.bias.requires_grad_(False)
            self.head_ccglu = CCGLU(512, 512) 
            self.head_final = nn.Linear(512, 4)

    def _generate_masks(self, B, device):
        """
        ### [修改] 生成互补掩码逻辑
        """
        # 随机掩盖 30% -> 5个切片掩盖约 1-2 个 (这里取 2 个)
        num_masked = 2 
        rand_indices = torch.rand(B, self.num_slices, device=device).argsort(dim=1)
        mask_indices = rand_indices[:, :num_masked]
        
        # M_a: 音频掩码 (1=掩盖)
        M_a = torch.zeros(B, self.num_slices, device=device)
        M_a.scatter_(1, mask_indices, 1)
        
        # M_v: 视频掩码 (与 M_a 互补) -> 音频掩盖处视频可见
        # M_a=1 -> M_v=0; M_a=0 -> M_v=1
        M_v = 1 - M_a 
        
        return M_a.bool(), M_v.bool()
    def forward(self, audio, video):
        """
        Input: (batch_size, data_length)ave_precision
        """

        if self.fusion_type == 'fc':
            clipwise_output_video, video_embed = self.video_encoder(video)

            clipwise_output_audio, audio_embed = self.audio_encoder(self.audio_frontend(audio))

            # av = clipwise_output_video + clipwise_output_audio

            av = torch.cat((audio_embed, video_embed), dim =1)

            av = torch.mean(av, 1)
            clipwise_output = self.fusion_linear(av)
            output_dict = {'clipwise_output': clipwise_output}

        elif self.fusion_type == 'atten':
            # TODO feature fusion and Cls
            _, video_embed = self.video_encoder(video)
            _, audio_embed = self.audio_encoder(self.audio_frontend(audio))

            video_features = F.relu_(self.att_linear(video_embed))
            audio_features = F.relu_(self.att_linear(audio_embed))
            dec_output, dec_slf_attn = self.slf_attn(
                video_features, video_features, video_features, mask=None)
            enc_output, enc_slf_attn = self.slf_attn(
                audio_features, audio_features, audio_features, mask=None)
            dec_enc_output, dec_enc_attn = self.enc_attn(
                dec_output, enc_output, enc_output, mask=None)
            # dec_enc_output, dec_enc_attn = self.enc_attn(
            #     video_features, audio_features, audio_features, mask=None)
            av = torch.mean(dec_enc_output, 1)
            clipwise_output = self.fusion_linear1(av)
            output_dict = {'clipwise_output': clipwise_output}

        elif self.fusion_type == 'fused-crossatt':
            # TODO mutual fusion
            _, video_embed = self.video_encoder(video)
            _, audio_embed = self.audio_encoder(self.audio_frontend(audio))
            av_embed = torch.cat((audio_embed, video_embed),dim=1)
            video_features = F.relu_(self.att_linear(video_embed))
            audio_features = F.relu_(self.att_linear(audio_embed))
            av_features = F.relu_(self.att_linear(av_embed))
            video_output, dec_video_attn = self.slf_attn(
                video_features, video_features, video_features, mask=None)
            audio_output, dec_audio_attn = self.slf_attn(
                audio_features, audio_features, audio_features, mask=None)
            av_output, dec_av_attn = self.slf_attn(
                av_features, av_features, av_features, mask=None)
            dec1_enc_output, dec1_enc_attn = self.enc_attn(
                video_output, av_output, av_output, mask=None)
            dec2_enc_output, dec2_enc_attn = self.enc_attn(
                audio_output, av_output, av_output, mask=None)
            dec_enc_output, dec_enc_attn = self.dec_attn(
                dec2_enc_output, dec1_enc_output, dec1_enc_output, mask=None)

            dec_enc_output = torch.mean(dec_enc_output, 1)
            clipwise_output = self.fusion_linear1(dec_enc_output)
            output_dict = {'clipwise_output': clipwise_output}
        elif self.fusion_type == 'MBT':
          # 1. 基础特征提取
            _, video_raw = self.video_encoder(video) 
            _, audio_raw = self.audio_encoder(self.audio_frontend(audio)) 
            
            # --- [修改] 强制切片对齐 (5 Slices) ---
            # 不论原始维度如何，强行池化为 5 个时间步
            if video_raw.dim() == 3:
                # [B, T, C] -> [B, C, T] -> Pool -> [B, 5, C]
                video_slice = F.adaptive_avg_pool1d(video_raw.transpose(1, 2), self.num_slices).transpose(1, 2)
            else: # 处理 [B, C, T, H, W]
                video_slice = video_raw.mean(dim=[-1, -2])
                video_slice = F.adaptive_avg_pool1d(video_slice.transpose(1, 2), self.num_slices).transpose(1, 2)

            if audio_raw.dim() == 3:
                audio_slice = F.adaptive_avg_pool1d(audio_raw.transpose(1, 2), self.num_slices).transpose(1, 2)
            else:
                audio_slice = audio_raw.mean(dim=[-1, -2])
                audio_slice = F.adaptive_avg_pool1d(audio_slice.transpose(1, 2), self.num_slices).transpose(1, 2)

            # 2. 投影与预处理
            video_embed = self.att_projection_v(video_slice) # [B, 5, 512]
            audio_embed = self.att_projection_a(audio_slice) # [B, 5, 512]

            video_embed = self.smfa_video(video_embed)
            audio_embed = self.smfa_audio(audio_embed)

            # 加入位置编码
            video_embed = video_embed + self.pos_encoding(video_embed)
            audio_embed = audio_embed + self.pos_encoding(audio_embed)

            # --- [修改] 3. 掩码生成与拆分 ---
            M_a_bool, M_v_bool = self._generate_masks(video_embed.size(0), video_embed.device)
            
            # Clone 用于最后的补全
            a_prime = audio_embed.clone()
            v_prime = video_embed.clone()

            # --- [修改] 4. 交叉生成 (A2V / V2A) ---
            
            # A2V: 音频可见部分 -> 预测视频
            # 简单策略：将掩盖部分置0，全序列输入，让网络自己学
            a_input_masked = audio_embed * (~M_a_bool.unsqueeze(-1)) 
            v_pred_features = self.gen_a2v(a_input_masked) # [B, 5, 512]
            
            # V2A: 视频可见部分 -> 预测音频
            v_input_masked = video_embed * (~M_v_bool.unsqueeze(-1))
            a_pred_features = self.gen_v2a(v_input_masked) # [B, 5, 512]

            # --- [修改] 5. 特征补全 (Feature Completion) ---
            # 用预测值填补掩码位置
            # a' (audio_fused) = original_audio (vis) + predicted_audio (mask)
            a_prime[M_a_bool] = a_pred_features[M_a_bool]
            
            # v' (video_fused) = original_video (vis) + predicted_video (mask)
            v_prime[M_v_bool] = v_pred_features[M_v_bool]

            # --- [修改] 6. 重建解码 (用于 Loss) ---
            recon_a = self.decoder_a(a_prime)
            recon_v = self.decoder_v(v_prime)

            # --- [修改] 7. 融合分类 ---
            # 对补全后的特征进行全局平均池化
            a_final = torch.mean(a_prime, dim=1) # [B, 512]
            v_final = torch.mean(v_prime, dim=1) # [B, 512]
            
            # Gate 融合
            combined = torch.cat((a_final, v_final), dim=1) # [B, 1024]
            gates = self.channel_gate(combined) # [B, 2, 512]
            w_a, w_v = gates[:, 0, :], gates[:, 1, :]
            
            final_vector = w_a * a_final + w_v * v_final

            # 输出
            final_vector = self.bn_neck(final_vector)
            feat_enhanced = self.head_ccglu(final_vector)
            clipwise_output = self.head_final(feat_enhanced)
            
            # 返回字典：包含分类输出和重建所需的 Target/Prediction
            output_dict = {
                'clipwise_output': clipwise_output,
                'recon_a': recon_a,         # 重建值
                'target_a': audio_embed,    # 真实值 (Ground Truth)
                'recon_v': recon_v,         # 重建值
                'target_v': video_embed     # 真实值 (Ground Truth)
            }
        
        return output_dict