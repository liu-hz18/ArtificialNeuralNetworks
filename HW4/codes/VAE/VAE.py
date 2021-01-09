import torch.nn as nn
import torch
import os

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view((x.size(0),) + self.shape)

class VAE(nn.Module):
    def __init__(self, num_channals, latent_dim):
        super(VAE, self).__init__()
        self.num_channals = num_channals
        self.latent_dim = latent_dim
        # Define the architecture of VAE here
        # TODO START
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=num_channals, out_channels=16, kernel_size=3, stride=1, padding=1), # out: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # out: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # out: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1), # out: 16 x 8 x 8
            nn.ReLU(),
            nn.Flatten(start_dim=1, end_dim=-1), # 16*8*8 = 1024
            nn.Linear(16*8*8, 2*latent_dim),
            nn.ReLU(),
            nn.Linear(2*latent_dim, 2*latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 16*8*8),
            Reshape(16, 8, 8),
            nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.Sigmoid(),
        )
        # TODO END

    def reparameterize(self, mu, log_var):
        '''
        *   Arguments:
            *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
        *   Returns:
            *   reparameterized samples (torch.FloatTensor): [batch_size, latent_dim]
        '''
        # TODO START
        std = (log_var * 0.5).exp()
        eps = torch.randn(mu.size(0), mu.size(1)).to(std.device)
        sampled_z = eps * std + mu
        return sampled_z
        # TODO END

    def forward(self, x=None, z=None):
        '''
        *   Arguments:
            *   x (torch.FloatTensor): [batch_size, 1, 32, 32]
            *   z (torch.FloatTensor): [batch_size, latent_dim]
        *   Returns:
            *   if `x` is not `None`, return a list:
                *   Reconstruction of `x` (torch.FloatTensor)
                *   mu (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
                *   log_var (torch.FloatTensor): [batch_size, latent_dim], parameters of the diagnoal Gaussian posterior q(z|x)
            *  if `x` is `None`, return samples generated from the given `z` (torch.FloatTensor): [num_samples, 1, 32, 32]
        '''
        if x is not None:
            # TODO START
            mu, log_var = self.encoder(x).chunk(2, dim=-1)
            sampled_z = self.reparameterize(mu, log_var)
            recon_x = self.decoder(sampled_z)
            return recon_x, mu, log_var
            # TODO END
        else:
            assert z is not None
            # TODO START
            gen_x = self.decoder(z)
            return gen_x
            # TODO END

    def restore(self, ckpt_dir):
        try:
            if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
                path = os.path.join(ckpt_dir, 'pytorch_model.bin')
            else:
                path = os.path.join(ckpt_dir, str(max(int(name) for name in os.listdir(ckpt_dir))), 'pytorch_model.bin')
        except:
            return None
        self.load_state_dict(torch.load(path))
        return os.path.split(path)[0]

    def save(self, ckpt_dir, global_step):
        os.makedirs(os.path.join(ckpt_dir, str(global_step)), exist_ok=True)
        path = os.path.join(ckpt_dir, str(global_step), 'pytorch_model.bin')
        torch.save(self.state_dict(), path)
        return os.path.split(path)[0]
