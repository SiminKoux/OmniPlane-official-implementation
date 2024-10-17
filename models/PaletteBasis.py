import torch
import torch.nn.functional as F
from models.Render_mlp import positional_encoding

class Palette_Basis_Net(torch.nn.Module):
    def __init__(self, num_basis):
        super(Palette_Basis_Net, self).__init__()
        
        self.pos_encode_dim = 2
        self.input_data_dim = 3
        self.out_color_dim = 3

        self.num_layers_basis = 2
        self.hidden_dim = 64
        self.fea_channel = self.input_data_dim * (2 * self.pos_encode_dim + 1) + 1
        self.out_channel = 2 * self.out_color_dim * self.pos_encode_dim + 1
        self.num_basis = num_basis    # 4 or 5

        basis_net = []
        for l in range(self.num_layers_basis):
            if l == 0:
                in_dim = self.fea_channel + 3  # pos: (2*3*freq+3) + time: 1 + diffuse_color: 3
            else:
                in_dim = self.hidden_dim   # 64
            if l == self.num_layers_basis - 1:
                out_dim = self.out_channel
            else:
                out_dim = self.hidden_dim
            basis_net.append(torch.nn.Linear(in_dim, out_dim, bias=False))
        self.basis_net = torch.nn.ModuleList(basis_net)

        # color offset, radiance and color weights
        self.offsets_radiance_net = torch.nn.Linear(self.out_channel, self.num_basis*3 + 1)  # num_basis*3: offset, 1: radiance
        self.omega_net = torch.nn.Sequential(torch.nn.Linear(self.out_channel, self.num_basis, bias=False), 
                                             torch.nn.Softplus())

        print("Palette Basis Networks:")
        print("(basis_net:)", self.basis_net)
        print("(offsets_radiance_net:)", self.offsets_radiance_net)
        print("(omega_net):", self.omega_net)
    
    def forward(self, coords, times, diff_colors):
        '''
            coords: 3D positions, a tensor of shape [B, N, 3]        (4096, 256, 3)
            times: times of the video, a tensor of shape [B, N, 1]   (4096, 256, 1)
            diff_colors: RGB values, a tensor of shape [B, N, 3]     (4096, 256, 3)
            return:
                (1) omega
                (2) color offsets
                (3) radiance
        '''
        input_data = [times]
        input_data += [coords]
        input_data += [positional_encoding(coords, 2)]
        input_data += [diff_colors]
        indata = torch.cat(input_data, dim=-1)  # [B*N, in_dim]
        for l in range(self.num_layers_basis):
            indata = self.basis_net[l](indata)
            if l != self.num_layers_basis - 1:
                indata = F.elu(indata, inplace=True)
        palette_geo_feat = indata
        
        offsets_radiance = self.offsets_radiance_net(palette_geo_feat) # B, N_B*3
        
        omega = self.omega_net(palette_geo_feat) + 0.05   # B, N_B
        omega = omega / (omega.sum(dim=-1, keepdim=True)) # B, N_B
        
        return offsets_radiance, omega