import torch
from torch import nn, Tensor
from src import utils
import torch.nn.functional as F
from src.Fast_Gestures.mamba_ssm import Mamba, Mamba2
from src.Fast_Gestures.components.mamba_block import MambaBlock


def _build_embedding(dim, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(dim).unsqueeze(0)  # [1,dim]
    table = steps * 10.0 ** (dims * 4.0 / dim)  # [T,dim]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels, motion_decoder=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.output = motion_decoder
        # self.conv1d = nn.Conv1d(in_channels=hidden_size, out_channels=out_channels, kernel_size=1)
        # self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        # x = self.linear(x)
        # x = self.norm_final(x)
        x = self.output(x)
        return x


class DiffusionEmbedding(nn.Module):
    def __init__(self, dim, proj_dim, max_steps=5000):  # 5000
        super().__init__()
        self.register_buffer(
            "embedding", _build_embedding(dim, max_steps), persistent=False
        )
        self.projection1 = nn.Linear(dim * 2, proj_dim)
        self.projection2 = nn.Linear(proj_dim, proj_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x


class Transpose(nn.Module):
    """ Wrapper class of torch.transpose() for Sequential module. """

    def __init__(self, shape: tuple):
        super(Transpose, self).__init__()
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        return x.transpose(*self.shape)


class EpsilonTheta(nn.Module):
    def __init__(self,
                 encoder_dim: int = 1024,
                 time_emb_dim: int = 1024,
                 motion_encoder: nn = None,
                 motion_decoder: nn = None,
                 wavLMEncoder: nn = None,
                 wavlm_layer: int = 12,
                 enable_style_encoder: bool = True,
                 cond_dropout_rate: float = 0,
                 gesture_features: int = 69,
                 is_mamba_style_extractor: bool = False,
                 style_mamba_version: str = 'mamba_1',
                 mamba_version: str = "mamba_2",    # "mamba_1", "mamba_2"
                 mamba_d_state=256,  # SSM state expansion factor
                 mamba_d_conv=3,  # Local convolution width
                 mamba_expand=2,  # Block expansion factor
                 style_mamba_d_state=256,  # SSM state expansion factor
                 style_mamba_d_conv=3,  # Local convolution width
                 style_mamba_expand=2,  # Block expansion factor
                 block_depth=2,
                 ):
        super(EpsilonTheta, self).__init__()

        # self.target_length = None
        # self.seed_gestures = None
        self.target_length = None
        self.wav_encoded = None
        self.diffusion_embedding = DiffusionEmbedding(
            time_emb_dim, proj_dim=encoder_dim
        )
        self.motion_encoder = motion_encoder
        self.motion_decoder = motion_decoder
        self.wavLMEncoder = wavLMEncoder
        self.wavlm_layer = wavlm_layer

        self.enable_style_encoder = enable_style_encoder
        self.mamba_block = nn.ModuleList([])
        for i in range(block_depth):
            self.mamba_block.append(MambaBlock(encoder_dim=encoder_dim,  # Model dimension d_model
                                               mamba_d_state=mamba_d_state,  # SSM state expansion factor
                                               mamba_d_conv=mamba_d_conv,  # Local convolution width
                                               mamba_expand=mamba_expand,  # Block expansion factor
                                               mamba_version=mamba_version,
                                               ))
        if is_mamba_style_extractor:
            if style_mamba_version == 'mamba_1':

                self.style_mamba_block = Mamba(d_model=768,
                                               d_state=style_mamba_d_state,
                                               d_conv=style_mamba_d_conv,
                                               expand=style_mamba_expand)
            elif style_mamba_version == 'mamba_2':
                self.style_mamba_block = Mamba2(d_model=768,
                                               d_state=style_mamba_d_state,
                                               d_conv=style_mamba_d_conv,
                                               expand=style_mamba_expand)
        else:
            self.style_mamba_block =  nn.Sequential(
                        Transpose(shape=(1, 2)),
                        nn.Conv1d(in_channels=768, out_channels=768, kernel_size=500, stride=1, dilation=2),
                        Transpose(shape=(1, 2)),
                        nn.LayerNorm(768),
                    )  # the numbers of parameters in Conv1D

        self.final_layer = FinalLayer(encoder_dim, out_channels=gesture_features, motion_decoder=motion_decoder)
        # TODO down_sampler
        self.down_sampler = nn.Sequential(
            Transpose(shape=(1, 2)),
            # for base wavlm model
            # nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=61, stride=2, dilation=1),               # 6s
            # nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=101, stride=2, dilation=1),              # 10s
            nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=201, stride=2, dilation=1),               # 20s
            # nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=301, stride=2, dilation=1),              # 30s
            # nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=41, stride=1, dilation=1),               # 24fps
            #nn.Conv1d(in_channels=768, out_channels=encoder_dim, kernel_size=1, stride=1, dilation=1),                 # 50s

            # for large wavlm model
            # nn.Conv1d(in_channels=1024, out_channels=encoder_dim, kernel_size=201, stride=2, dilation=1),
            # 60 = d * (k - 1)
            Transpose(shape=(1, 2)),
            nn.LayerNorm(encoder_dim),
            nn.LeakyReLU(0.2, True),
        )
        self.dropout = nn.Dropout(p=cond_dropout_rate)

    def WavLM_Encoder(self, audio, enable_style_encoder):
        # print(audio.shape)
        language_tag, all_wavlm_layer = self.wavLMEncoder(audio)  # [64,299,768]
        if language_tag != 'Chinese':
            wavlm_cond = all_wavlm_layer[self.wavlm_layer]
        else:
            wavlm_cond = all_wavlm_layer
        if enable_style_encoder is True:
            style_encoded = self.style_encoder(wavlm_cond)
            wavlm_cond_style_embedded = wavlm_cond + style_encoded
        else:
            # log.info("No Style Encoder!!!")
            wavlm_cond_style_embedded = wavlm_cond

        wavlm_cond_style_embedded = self.down_sampler(wavlm_cond_style_embedded)  # [64,58,512]

        # # 将张量调整为适合 interpolate 的形状
        # # 即从 [B, T, C] 调整为 [B, C, 1, T]
        # wavlm_cond_style_embedded = wavlm_cond_style_embedded.permute(0, 2, 1)
        # # # 使用 PyTorch 的 interpolate 函数进行线性插值
        # wavlm_cond_style_embedded = F.interpolate(wavlm_cond_style_embedded, size=self.target_length, mode='linear',
        #                                            align_corners=False)
        # # # 将张量重新调整回 [B, T, C] 形状
        # wavlm_cond_style_embedded = wavlm_cond_style_embedded.permute(0, 2, 1)
        #
        # # count = (wavlm_cond_style_embedded == 0).sum().item()
        return wavlm_cond_style_embedded

    def style_encoder(self, x):
        y = self.style_mamba_block(x)
        return y

    def forward(self, inputs, cond, dif_time_step, processing_state='training', last_time_stamp=999):
        """
        inputs: [B, T, D] tensor of gestures seq
        condition: [B, T, D] tensor of control seq, such as audio features
        dif_time_step: [N,] tensor of diffusion timesteps
        """
        # global c
        # self.target_length = inputs.shape[1]
        t = self.diffusion_embedding(dif_time_step.type(torch.long)).unsqueeze(1)
        x = self.motion_encoder(inputs)  # [64,58,512]

        if processing_state == 'training' or processing_state == 'validation' or (
                processing_state == 'test' and (last_time_stamp in dif_time_step)):
            self.wav_encoded = self.WavLM_Encoder(audio=cond, enable_style_encoder=self.enable_style_encoder)
            # self.wav_encoded = self.wav_encoded.permute(0, 2, 1)
            # # # # 使用 PyTorch 的 interpolate 函数进行线性插值
            # self.wav_encoded = F.interpolate(self.wav_encoded, size=inputs.shape[1], mode='linear',
            #                                             align_corners=False)
            # # # # 将张量重新调整回 [B, T, C] 形状
            # self.wav_encoded = self.wav_encoded.permute(0, 2, 1)
        # c = t + self.wav_encoded + self.seed_gestures
        c = t + self.wav_encoded

        for block in self.mamba_block:
            x = block(x, c)
        x = self.final_layer(x, c)
        return x
