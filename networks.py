import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_dim=2048,
        hidden_layers=1,
        out_dim=2,
        activation='relu'
    ):
        super().__init__()
        if activation == 'relu':
            act = nn.ReLU()
        elif activation is None:
            act = nn.Identity()
        self.encoder = nn.ModuleList([
            nn.Linear(input_size, hidden_dim),
            act,
        ])
        self.encoder += [nn.Linear(hidden_dim, hidden_dim), act] * (hidden_layers - 1)
        self.decoder = nn.ModuleList([
            nn.Linear(hidden_dim, out_dim)
        ])

    def forward(self, x):
        """
        """
        for layer in self.encoder + self.decoder:
            x = layer(x)
        return x


class VAEMLP(nn.Module):
    def __init__(
        self, input_size=64, latent_dim=1024, hidden_dim=2048, relu_out=True
    ):
        """
        """
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.ModuleList([
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
            nn.ReLU(),
        ])
        self.decoder = nn.ModuleList([
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_size),
        ])
        if relu_out:
            self.decoder.append(nn.ReLU())

    def _sample(self, mu, log_var):
        eps = torch.randn(self.batch_size, self.latent_dim)
        eps = eps.to(device)
        std = torch.exp(log_var / 2)
        std = std.to(device)
        return mu + std * eps

    def forward(self, x):
        self.batch_size = x.shape[0]
        # Encoder
        for layer in self.encoder:
            x = layer(x)
        mu, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]
        mu1 = self._sample(mu, log_var)
        # Decoder
        x = mu1.clone()
        for layer in self.decoder:
            x = layer(x)
        return mu, mu1, log_var, x


class VAEConv(nn.Module):
    def __init__(self, input_size=64, latent_dim=2048):
        """
        """
        super().__init__()
        self.input_size = input_size
        self.in_ch = 3
        self.out_ch = 3
        self.latent_dim = latent_dim
        self.bottleneck_dim = 5
        self.bridge_channels = 512

        if input_size == 64:
            self.encoder = nn.ModuleList([
                nn.Conv2d(self.in_ch, 64, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 256, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 512, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
                # nn.Linear(256 * 5 * 5, latent_dim * 2),
            ])
            self.enc_focal = nn.ModuleList([
                nn.Linear(512 * 5 * 5 + input_size ** 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim * 2),
            ])
            self.bridge = nn.Linear(
                latent_dim,
                self.bridge_channels * self.bottleneck_dim * self.bottleneck_dim
            )
            self.decoder = nn.ModuleList([
                nn.ConvTranspose2d(self.bridge_channels, 256, 5, stride=2, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, stride=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, self.out_ch, 5, stride=2, output_padding=1),
                nn.ReLU(),
            ])
        elif input_size == 256:
            self.encoder = nn.ModuleList([
                nn.Conv2d(self.in_ch, 64, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 256, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(256, 512, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(512, 512, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(512, 512, 5, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            ])
            self.enc_focal = nn.ModuleList([
                nn.Linear(512 * 5 * 5 + input_size ** 2, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim * 2),
            ])
            self.bridge = nn.Linear(
                latent_dim,
                self.bridge_channels * self.bottleneck_dim * self.bottleneck_dim
            )
            self.decoder = nn.ModuleList([
                nn.ConvTranspose2d(self.bridge_channels, 512, 5, stride=2, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 5, stride=2, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, stride=2, output_padding=0),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 256, 5, stride=2, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, self.out_ch, 5, stride=2, output_padding=1),
                nn.ReLU(),
            ])

    def _sample(self, mu, log_var):
        eps = torch.randn(self.batch_size, self.latent_dim)
        eps = eps.to(device)
        std = torch.exp(log_var / 2)
        std = std.to(device)
        return mu + std * eps

    def forward(self, x0, att_map):
        self.batch_size = x0.shape[0]

        # Encoder
        x = self.encoder[0](x0)
        for layer in self.encoder[1:]:
            x = layer(x)

        # Inject focal point coordinate info right before latent layer
        focal_xy = nn.Flatten()(att_map)
        x = torch.cat([x, focal_xy], dim=1)
        for layer in self.enc_focal:
            x = layer(x)

        mu, log_var = x[:, :self.latent_dim], x[:, self.latent_dim:]
        mu1 = self._sample(mu, log_var)
        x = self.bridge(mu1).reshape(
            self.batch_size,
            self.bridge_channels,
            self.bottleneck_dim,
            self.bottleneck_dim
        )

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        return mu, mu1, log_var, x
