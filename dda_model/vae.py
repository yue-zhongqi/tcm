import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, x_dim=2048, z_dim=100, enc_layers='1200 600', dec_layers='600'):
        super(VAE, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        enc_layers = enc_layers.split()
        encoder = []
        for i in range(len(enc_layers)):
            num_hidden = int(enc_layers[i])
            pre_hidden = int(enc_layers[i-1])
            if i == 0:
                encoder.append(nn.Linear(x_dim, num_hidden))
                encoder.append(nn.ReLU())
            else:
                encoder.append(nn.Dropout(p=0.3))
                encoder.append(nn.Linear(pre_hidden, num_hidden))
                encoder.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder)

        last_hidden = int(enc_layers[-1])
        self.mu_net = nn.Sequential(
            nn.Linear(last_hidden, z_dim)
        )

        self.sig_net = nn.Sequential(
            nn.Linear(last_hidden, z_dim)
        )

        dec_layers = dec_layers.split()
        decoder = []
        for i in range(len(dec_layers)):
            num_hidden = int(dec_layers[i])
            pre_hidden = int(dec_layers[i-1])
            if i == 0:
                decoder.append(nn.Linear(z_dim, num_hidden))
                decoder.append(nn.ReLU())
            else:
                decoder.append(nn.Linear(pre_hidden, num_hidden))
                decoder.append(nn.ReLU())
            if i == len(dec_layers) - 1:
                decoder.append(nn.Linear(num_hidden, x_dim))
                decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

    def encode(self, X):
        hidden = self.encoder(X)
        mu = self.mu_net(hidden)
        log_sigma = self.sig_net(hidden)
        return mu, log_sigma

    def decode(self, mu, log_sigma):
        eps = torch.rand(mu.size()).cuda()
        Z = mu + torch.exp(log_sigma / 2) * eps
        ZS = Z
        Xp = self.decoder(ZS)
        return Xp

    def forward(self, X):
        hidden = self.encoder(X)
        mu = self.mu_net(hidden)
        log_sigma = self.sig_net(hidden)
        eps = torch.rand(mu.size())
        eps = eps.cuda()
        Z = mu + torch.exp(log_sigma / 2) * eps
        ZS = Z
        Xp = self.decoder(ZS)
        return Xp, mu, log_sigma, Z

    def sample(self):
        Z = torch.rand([self.z_dim]).cuda()
        ZS = Z
        return self.decoder(ZS)

    def vae_loss(self, X, Xp, mu, log_sigma, beta=1.0):
        reconstruct_loss = 0.5 * torch.sum(torch.pow(X - Xp, 2), 1)
        KL_divergence = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1 - log_sigma, 1)
        return torch.mean(reconstruct_loss + beta * KL_divergence)