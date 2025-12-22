import torch.nn as nn
from src.Fast_Gestures.components.conformer.modules import ResidualConnectionModule

from src.Fast_Gestures.components.conformer.convolution import ConformerConvModule
from timm.models.vision_transformer import Mlp


def adaptive_modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift


class MambaBlock(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 mamba_d_state: int,
                 mamba_d_conv: int,
                 mamba_expand: int,
                 conv_kernel_size: int = 3,
                 conv_expansion_factor: int = 2,
                 conv_dropout_p: float = 0.1,
                 mlp_ratio: float = 4.0,
                 mamba_version: str = "mamba_2"  # "mamba_1"
                 ):
        super().__init__()
        if mamba_version == "mamba_2":
            print('Initializing mamba_2 ...')
            from src.Fast_Gestures.mamba_ssm import Mamba2
            self.mamba = nn.Sequential(
                ResidualConnectionModule(
                    module=Mamba2(d_model=encoder_dim,  # Model dimension d_model
                                  d_state=mamba_d_state,  # SSM state expansion factor
                                  d_conv=mamba_d_conv,  # Local convolution width
                                  expand=mamba_expand,  # Block expansion factor
                                  ),
                ),
                ResidualConnectionModule(
                    module=ConformerConvModule(
                        in_channels=encoder_dim,
                        kernel_size=conv_kernel_size,
                        expansion_factor=conv_expansion_factor,
                        dropout_p=conv_dropout_p, ),
                ),
            )
        if mamba_version == "mamba_1":
            print('Initializing mamba_1...')
            from src.Fast_Gestures.mamba_ssm import Mamba
            self.mamba = nn.Sequential(
                ResidualConnectionModule(
                    module=Mamba(d_model=encoder_dim,  # Model dimension d_model
                                      d_state=mamba_d_state,  # SSM state expansion factor
                                      d_conv=mamba_d_conv,  # Local convolution width
                                      expand=mamba_expand,  # Block expansion factor
                                      ),
                    ),
                ResidualConnectionModule(
                    module=ConformerConvModule(
                            in_channels=encoder_dim,
                            kernel_size=conv_kernel_size,
                            expansion_factor=conv_expansion_factor,
                            dropout_p=conv_dropout_p, ),
                    ),
                )
        mlp_hidden_dim = int(encoder_dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=encoder_dim, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.norm1 = nn.LayerNorm(encoder_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(encoder_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, 6 * encoder_dim, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
        x = x + gate_msa * self.mamba(adaptive_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(adaptive_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
