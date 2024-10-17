import torch

def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts

class Render_MLP(torch.nn.Module):
    """
        A general MLP module for rendering in the training process
        Input:
            (1) Time PE: t_pe
            (2) Appearance Feature PE: fea_pe
            (3) 3D Position PE: pos_pe
            (4) View Direction PE: view_pe
        Rule:
            pe > 0: use PE with frequency = pe
            pe < 0: not use this feature
            pe = 0: only use original value
    """
    def __init__(self, 
                 inChannel: int, 
                 outChannel: int,
                 t_pe: int = 6,
                 fea_pe: int = 6,
                 pos_pe: int = 6,
                 view_pe: int = 6, 
                 featureC: int = 128):
        super(Render_MLP, self).__init__()

        self.in_mlpC = inChannel
        self.use_t = t_pe >= 0
        self.use_fea = fea_pe >= 0
        self.use_pos = pos_pe >= 0
        self.use_view = view_pe >= 0

        self.t_pe = t_pe
        self.fea_pe = fea_pe
        self.pos_pe = pos_pe
        self.view_pe = view_pe

        # Input channel
        if self.use_t:
            self.in_mlpC += 1 + 2 * t_pe * 1
        if self.use_fea:
            self.in_mlpC += 2 * fea_pe * inChannel
        if self.use_pos:
            self.in_mlpC += 3 + 2 * pos_pe * 3
        if self.use_view:
            self.in_mlpC += 3 + 2 * view_pe * 3
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)  # input channel, 128
        layer2 = torch.nn.Linear(featureC, featureC)      # 128, 128
        layer3 = torch.nn.Linear(featureC, outChannel)    # 128, outchannel

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, 
                pts: torch.Tensor, 
                viewdirs: torch.Tensor, 
                features: torch.Tensor,
                frame_time: torch.Tensor,
                ) -> torch.Tensor:
        # Collect input data
        indata = [features]
        if self.use_t:
            indata += [frame_time]
            if self.t_pe > 0:
                indata += [positional_encoding(frame_time, self.t_pe)]
        if self.use_fea:
            # rgb feature encoding
            if self.fea_pe > 0:
                indata += [positional_encoding(features, self.fea_pe)]
        if self.use_pos:
            indata += [pts]
            # 3D position encoding
            if self.pos_pe > 0:
                indata += [positional_encoding(pts, self.pos_pe)]
        if self.use_view:
            indata += [viewdirs]
            # view direction encoding
            if self.view_pe > 0:
                indata += [positional_encoding(viewdirs, self.view_pe)]
        mlp_in = torch.cat(indata, dim=-1)  # computed input channel counts
        out = self.mlp(mlp_in)              # decode for rgb or density values
        output = torch.sigmoid(out)

        return output