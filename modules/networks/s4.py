import torch.nn as nn
# from models.s4.s4 import S4Block as S4  # Can use full version instead of minimal S4D standalone below
from models.s4.s4d import S4D
from src.models.sequence.modules.s4nd import S4ND
import torch

class S4Model(nn.Module):
    def __init__(
        self,
        d_input: int,
        # d_output,
        d_model: int,
        n_layers=1,
        dropout=0.2,
        prenorm=False,
        lr=0.01,
    ):
        super().__init__()

        self.prenorm = prenorm
        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=lr)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(nn.Dropout1d(dropout))

        # Linear decoder
        # self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        # x = x.mean(dim=1)

        # Decode the outputs
        # x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x


class S4Model2D(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_state: int,
        d_model: int,
        n_layers=1,
        dropout=0.1,
        prenorm=False,
    ):
        super().__init__()
        self.prenorm = prenorm
        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        # self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList([])
        self.s4_first_layer = S4ND(d_model=d_input, 
                    d_state=d_state,
                    dim=2,
                    out_channels=d_model,
                    channels=1,
                    bidirectional=True,
                    activation='gelu', # activation in between SS and FF
                    ln=False, # Extra normalization
                    final_act=None, # activation after FF
                    initializer=None, # initializer on FF
                    dropout=dropout,
                    weight_norm=False, # weight normalization on FF
                    hyper_act=None, # Use a "hypernetwork" multiplication
                    tie_dropout=False,
                    transposed=True, # axis ordering (B, L, D) or (B, D, L)
                    verbose=False,
                    trank=1, # tensor rank of C projection tensor
                    linear=True,
                    return_state=True,
                    contract_version=0,
                    # SSM Kernel arguments
                    kernel=None,  # New option
                    mode='dplr',  # Old option)
                    )
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4ND(d_model=d_model, 
                    d_state=d_state,
                    dim=2,
                    out_channels=d_model,
                    channels=1,
                    bidirectional=True,
                    activation='gelu', # activation in between SS and FF
                    ln=False, # Extra normalization
                    final_act=None, # activation after FF
                    initializer=None, # initializer on FF
                    dropout=dropout,
                    weight_norm=False, # weight normalization on FF
                    hyper_act=None, # Use a "hypernetwork" multiplication
                    tie_dropout=False,
                    transposed=True, # axis ordering (B, L, D) or (B, D, L)
                    verbose=False,
                    trank=1, # tensor rank of C projection tensor
                    linear=True,
                    return_state=True,
                    contract_version=0,
                    # SSM Kernel arguments
                    kernel=None,  # New option
                    mode='dplr',  # Old option)
                    ))
            self.norms.append(nn.BatchNorm2d(d_model))
        
    def forward(self, x):
        # x = self.encoder(x)
        # print(x.shape)
        x, _ = self.s4_first_layer(x)
        for layer, norm in zip(self.s4_layers, self.norms):
            z = x
            if self.prenorm:
                z = norm(z)
            z, _ = layer(z)
            x = z + x
            if not self.prenorm:
                x = norm(x)
        x = nn.AdaptiveAvgPool2d(output_size=1)(x)
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        return x


if __name__ == "__main__":
    # rand_img = torch.randn([32, 64, 128, 128])
    # out, _ = s4nd_img(rand_img)
    # print(out.shape)
    # rand_seq = torch.randn([32, 4, 64, 64])
    # out, _ = s4nd_vid(rand_img)
    # print(out.shape)
    s4nd_img = S4ND(
        d_model=64,
        d_state=64,
        l_max=None, # Maximum length of sequence (list or tuple). None for unbounded
        dim=2, # Dimension of data, e.g. 2 for images and 3 for video
        out_channels=512, # Do depthwise-separable or not
        channels=1, # maps 1-dim to C-dim
        bidirectional=True,
        # Arguments for FF
        activation='gelu', # activation in between SS and FF
        ln=False, # Extra normalization
        final_act=None, # activation after FF
        initializer=None, # initializer on FF
        weight_norm=False, # weight normalization on FF
        hyper_act=None, # Use a "hypernetwork" multiplication
        dropout=0.0, tie_dropout=False,
        transposed=True, # axis ordering (B, L, D) or (B, D, L)
        verbose=False,
        trank=1, # tensor rank of C projection tensor
        linear=True,
        return_state=True,
        contract_version=0,
        # SSM Kernel arguments
        kernel=None,  # New option
        mode='dplr',  # Old option
        # **kernel_args,
    )

    s4_2d = S4Model2D(
            d_input=4,
            d_state=64,
            d_model=256,
            n_layers=2,
            dropout=0.1,
            prenorm=False,
            )
    rand_img = torch.randn([32, 4, 64, 64])
    out = s4_2d(rand_img)
    print(out.shape)
    