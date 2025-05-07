from typing import Optional, Sequence, Callable, List

from deeplay.components import ConvolutionalEncoder2d, ConvolutionalDecoder2d
from deeplay.applications import Application
from deeplay.external import External, Optimizer, Adam

from pytorch_msssim import ssim
import torch
import torch.nn as nn


class VariationalAutoEncoder(Application):
    input_size: tuple
    channels: list
    latent_dim: int
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    beta: float
    reconstruction_loss: torch.nn.Module
    metrics: list
    optimizer: Optimizer

    def __init__(
        self,
        input_size: Optional[Sequence[int]] = (28, 28),
        channels: Optional[List[int]] = [32, 64],
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        reconstruction_loss: Optional[Callable] = nn.BCELoss(reduction="sum"),
        latent_dim=int,
        beta=1,
        optimizer=None,
        **kwargs,
    ):
        red_size = [int(dim / (2 ** len(channels))) for dim in input_size]
        self.encoder = encoder or self._get_default_encoder(channels)
        self.fc_mu = nn.Linear(
            channels[-1] * red_size[0] * red_size[1],
            latent_dim,
        )
        self.fc_var = nn.Linear(
            channels[-1] * red_size[0] * red_size[1],
            latent_dim,
        )
        self.fc_dec = nn.Linear(
            latent_dim,
            channels[-1] * red_size[0] * red_size[1],
        )
        self.decoder = decoder or self._get_default_decoder(channels[::-1], red_size)
        self.reconstruction_loss = reconstruction_loss or nn.BCELoss(reduction="sum")
        self.latent_dim = latent_dim
        self.beta = beta

        super().__init__(**kwargs)

        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    def _get_default_encoder(self, channels):
        encoder = ConvolutionalEncoder2d(
            1,
            channels,
            channels[-1],
        )
        encoder.postprocess.configure(nn.Flatten)
        return encoder

    def _get_default_decoder(self, channels, red_size):
        decoder = ConvolutionalDecoder2d(
            channels[0],
            channels,
            1,
            out_activation=nn.Sigmoid,
        )
        # for block in decoder.blocks[:-1]:
        #     block.upsample.configure(nn.ConvTranspose2d, kernel_size=3, stride=2, in_channels=block.in_channels, out_channels=block.out_channels)

        decoder.preprocess.configure(
            nn.Unflatten,
            dim=1,
            unflattened_size=(channels[0], red_size[0], red_size[1]),
        )
        return decoder

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        x = self.fc_dec(z)
        x = self.decoder(x)
        return x
    
    def make_bandpass_weights(self, H, boost_range=(50, 450), boost_factor=5.0, base_weight=1.0, device="cuda"):
        weights = torch.full((H,), base_weight, device=device)
        start, end = boost_range
        weights[start:end] = boost_factor
        return weights.view(1, 1, H, 1)
    
    #LogCosh loss which with possibility to increase loss for different parts of the image in height (to prevent the network learning that the low frequencies often exists and using only that)
    def logCoshLoss(self, y_hat, y, use_freq_weights=False):
        B, C, H, W = y.shape
        device = y.device

        diff = y_hat - y
        log_cosh = torch.log(torch.cosh(diff + 1e-12))

        if use_freq_weights:
            freq_weights = self.make_bandpass_weights(H)
            log_cosh = log_cosh * freq_weights

        return log_cosh.mean()

    
    def training_step(self, batch, batch_idx):
        x, y = self.train_preprocess(batch)
        y_hat, mu, log_var = self(x)
        rec_loss, KLD, kl_penalty = self.compute_loss(y_hat, y, mu, log_var)
        tot_loss = rec_loss + kl_penalty
        loss = {"rec_loss": rec_loss, "KL_penalty": kl_penalty, "total_loss": tot_loss}
        for name, v in loss.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
        return tot_loss
    
    def combined_ssim_logcosh(self, y_hat, y):
        alpha = 5
        beta = 1
        #rec_loss = (alpha*self.logCoshLoss(y_hat, y, use_freq_weights=False) + beta*(1.0 - ssim(y_hat, y, data_range=1.0, size_average=True)))*0.5 //Averaged
        rec_loss = alpha*self.logCoshLoss(y_hat, y, use_freq_weights=False) + beta*(1.0 - ssim(y_hat, y, data_range=1.0, size_average=True)) #Summed
        return rec_loss
    def compute_loss(self, y_hat, y, mu, log_var):
        #rec_loss = self.logCoshLoss(y_hat, y, use_freq_weights=False) //Very good at capturing intensities for an entire melspectrogram band but bad at details
        #rec_loss = self.reconstruction_loss(y_hat, y) //Not able to get anything decent using MSE
        #rec_loss = 1.0 - ssim(y_hat, y, data_range=1.0, size_average=True) //Used for the vaeUsedForFirstSongs and the vaeLowDKL weights
        rec_loss = self.combined_ssim_logcosh(y_hat, y) #Combined ssim and logcosh losses which hopefully gives a better result
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)

        #Compute KL divergence
        KLD = -0.5 * torch.mean(torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1))

        #δ-VAE parameters
        delta = 1   # target KL divergence
        gamma = 0.1 # strength of KL penalty

        #δ-VAE KL penalty: encourages KL to stay near delta
        kl_penalty = gamma * (KLD - delta) ** 2

        return rec_loss, KLD, kl_penalty

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        y_hat = self.decode(z)
        return y_hat, mu, log_var
