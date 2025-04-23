import torch

t = torch.tensor([[-5., 5.], [-8., 6.], [-2., 6.]])

print(t)

r = torch.min(torch.abs(t[:, 0] - t[:, 1]) / 2)
z_bounds = torch.tensor([[-r], [r]]).expand(-1, t.shape[0])

print(z_bounds)
