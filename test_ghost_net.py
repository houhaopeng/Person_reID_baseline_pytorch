import torch
from ghost_net import ghost_net

model = ghost_net(width_mult=1.0)
input = torch.randn(32,3,224,224)
y = model(input)
print(y)