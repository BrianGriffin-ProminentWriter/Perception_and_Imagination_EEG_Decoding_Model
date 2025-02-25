import torch
import torch.nn as nn
import torch.nn.functional as F
from KAN import KAN


class MMPINet(nn.Module):
    def __init__(self, n_classes=3, num_heads=8):
        super(MMPINet, self).__init__()

        self.eegnet_per = EEGNet_per_DBB(n_classes=3, channels=124, samples=4097)
        self.eegnet_ima = EEGNet_img_DBB(n_classes=3, channels=124, samples=4097)

        self.attn_per_to_img = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)
        self.attn_img_to_per = nn.MultiheadAttention(embed_dim=2048, num_heads=num_heads, batch_first=True)

        self.final_linear = nn.Linear(8192, n_classes)
        self.final_kan1 = KAN([8192, 3], spline_order=4, grid_size=6)
        self.final_kan2 = KAN([3, 3], spline_order=4, grid_size=6)
        self.final_linear_out = nn.Linear(6, 3)

    def forward(self, x):
        x_per, x_img = self.eegnet_per(x), self.eegnet_ima(x)
        x_per_fla, x_img_fla = x_per.view(x.size(0), 1, -1), x_img.view(x.size(0), 1, -1)

        A, _ = self.attn_per_to_img(x_per_fla, x_per_fla, x_img_fla)
        B, _ = self.attn_img_to_per(x_img_fla, x_img_fla, x_per_fla)

        AB_concat = torch.cat((A, B), dim=-1)
        output_kan = self.final_kan2(self.final_kan1(AB_concat.squeeze(1)))
        output = torch.cat((output_kan, self.final_linear(AB_concat.squeeze(1))), dim=-1)
        output = self.final_linear_out(output)

        euclidean_dist = torch.norm(x_per_fla.squeeze(1) - x_img_fla.squeeze(1), p=2, dim=-1).mean().abs()
        kl_div = 0.5 * (F.kl_div(F.log_softmax(x_per.view(x.size(0), -1), dim=-1),
                                 F.softmax(x_img.view(x.size(0), -1), dim=-1), reduction='batchmean') +
                        F.kl_div(F.log_softmax(x_img.view(x.size(0), -1), dim=-1),
                                 F.softmax(x_per.view(x.size(0), -1), dim=-1), reduction='batchmean'))

        return output, 1 / (euclidean_dist + 0.1) + kl_div
