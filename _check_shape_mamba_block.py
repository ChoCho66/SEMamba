from mamba_ssm.modules.mamba_simple import Mamba

import torch
# from mamba_ssm import Mamba
from torchinfo import summary


batch, length, dim = 7, 137, 37
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=16,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

# summary_str = summary(model, input_size=[(7, 201, 286), (7, 201, 286)], depth=5, col_names=("input_size", "output_size", "num_params"), verbose=0)
summary_str = summary(model, input_size=[(batch, length, dim)], depth=15, col_names=("input_size", "output_size", "num_params"), verbose=0)
print(summary_str)

y = model(x)
print(x.shape)
print(y.shape)
assert y.shape == x.shape