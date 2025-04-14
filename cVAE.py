import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn import Parameter

# set torch random seed
torch.manual_seed(42)


def compute_ll(x, x_recon):
    return x_recon.log_prob(x).sum(1, keepdims=True).mean(0)

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

import torch

class mmJSD:
    """
    A class to compute the Mixture of Experts (MoE) for the mmJSD approach.

    This class will return the multimodal mean and variance based on the experts' means and variances.
    """

    def __call__(self, mus, variances):
        """
        Args:
            mus (torch.Tensor): Tensor of shape (M, D) where M is the number of experts, and D is the dimensionality.
            variances (torch.Tensor): Tensor of shape (M, D) where M is the number of experts, and D is the dimensionality.

        Returns:
            mu_multimodal (torch.Tensor): The combined multimodal mean.
            variance_multimodal (torch.Tensor): The combined multimodal variance.
        """

        # Compute the multimodal mean (arithmetic mean of the experts' means)
        mu_multimodal = torch.mean(mus, dim=0)
        
        # Compute the multimodal variance (mean of variances plus the variance of means)
        variance_multimodal = torch.mean(variances, dim=0) + torch.var(mus, dim=0)

        return mu_multimodal, variance_multimodal

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Layer Normalization
            nn.LeakyReLU(0.1),         # Leaky ReLU
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim),  # Project back to input dimension
            nn.LayerNorm(input_dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # Residual connection




mlp_input_dim = 270


mlp = nn.Sequential(
    nn.Linear(mlp_input_dim, 256),    # Increased neurons to 256
    nn.BatchNorm1d(256),              # Batch Normalization
    nn.ReLU(),
    nn.Dropout(0.3),                  # Dropout for regularization
    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),                # Added extra layer with 32 neurons
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()                      # Final layer for binary classification
).to(DEVICE)




mlp_residual = nn.Sequential(
    nn.Linear(mlp_input_dim, 512),   # Increased neurons to 512
    nn.LayerNorm(512),               # Layer Normalization
    nn.LeakyReLU(0.1),               # Leaky ReLU activation
    nn.Dropout(0.3),                 # Dropout for regularization

    # Residual Blocks (512 -> 256 -> 512)
    ResidualBlock(512, 256),
    ResidualBlock(512, 256),
    ResidualBlock(512, 256),

    # Final Layers
    nn.Linear(512, 128),             # Decreasing neuron count
    nn.LayerNorm(128),
    nn.LeakyReLU(0.1),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid()                     # Output layer for binary classification
).to(DEVICE)




class FocalLoss(nn.Module):
    def __init__(self, alpha_focal, gamma_focal, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha_focal = alpha_focal
        self.gamma_focal = gamma_focal
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        if targets[0].item() == 0:
            F_loss = self.alpha_focal * (1-pt)**self.gamma_focal * BCE_loss
        else:
            F_loss = (1-self.alpha_focal) * pt**self.gamma_focal * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        
class Encoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim, 
                c_dim,
                non_linear=False):
        super().__init__()

        self.input_size = input_dim
        self.hidden_dims = hidden_dim
        self.z_dim = hidden_dim[-1]
        self.c_dim = c_dim
        self.non_linear = non_linear
        self.layer_sizes_encoder = [input_dim + c_dim] + self.hidden_dims
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_encoder[:-1], self.layer_sizes_encoder[1:])]
               
        self.encoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.enc_mean_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)
        self.enc_logvar_layer = nn.Linear(self.layer_sizes_encoder[-2], self.layer_sizes_encoder[-1], bias=True)

    def forward(self, x, c):

        h1 = torch.cat((x, c), dim=1)
        for it_layer, layer in enumerate(self.encoder_layers):
            h1 = layer(h1)
            if self.non_linear:
                h1 = F.leaky_relu(h1)

        mu = self.enc_mean_layer(h1)
        logvar = self.enc_logvar_layer(h1)

        return mu, logvar

class Decoder(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                non_linear=False, 
                init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        self.init_logvar = init_logvar
        self.c_dim = c_dim
        self.layer_sizes_decoder = self.hidden_dims + [input_dim]
        self.layer_sizes_decoder[0] = self.hidden_dims[0] + c_dim
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_decoder[:-1], self.layer_sizes_decoder[1:])]
        self.decoder_layers = nn.Sequential(*lin_layers[0:-1])
        self.decoder_mean_layer = nn.Linear(self.layer_sizes_decoder[-2],self.layer_sizes_decoder[-1], bias=True)
        tmp_noise_par = torch.FloatTensor(1, self.input_size).fill_(self.init_logvar)
        self.logvar_out = Parameter(data=tmp_noise_par, requires_grad=True)


    def forward(self, z, c):
        c = c.reshape(-1, self.c_dim)
        x_rec = torch.cat((z, c),dim=1)
        for it_layer, layer in enumerate(self.decoder_layers):
            x_rec = layer(x_rec)
            if self.non_linear:
                x_rec = F.leaky_relu(x_rec)

        mu_out = self.decoder_mean_layer(x_rec)
        return Normal(loc=mu_out, scale=self.logvar_out.exp().pow(0.5))
 

    
class Discriminator(nn.Module):
    def __init__(
                self, 
                input_dim, 
                hidden_dim,
                c_dim,
                non_linear=False, 
                init_logvar=-3):
        super().__init__()
        self.input_size = input_dim
        self.hidden_dims = hidden_dim[::-1]
        self.non_linear = non_linear
        #self.c_dim = c_dim
        self.layer_sizes_discriminator = self.hidden_dims + [1]
        self.layer_sizes_discriminator[0] = self.hidden_dims[0] 
        lin_layers = [nn.Linear(dim0, dim1, bias=True) for dim0, dim1 in zip(self.layer_sizes_discriminator[:-1], self.layer_sizes_discriminator[1:])]
        self.discriminator_layers = nn.Sequential(*lin_layers[0:-1])
        self.discriminator_mean_layer = nn.Linear(self.layer_sizes_discriminator[-2],self.layer_sizes_discriminator[-1], bias=True)

    def forward(self, z):
        x_pred = z
        for it_layer, layer in enumerate(self.discriminator_layers):
            x_pred = layer(x_pred)
            if self.non_linear:
                x_pred = F.leaky_relu(x_pred)

        mu_out = self.discriminator_mean_layer(x_pred)
        return mu_out   
    
    



class mmcVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                non_linear=False):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
        self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.discriminator = Discriminator(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.optimizer1 = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate) 
        self.optimizer2 = optim.Adam(list(self.discriminator.parameters()), lr=self.learning_rate) 
        self.optimizer3 = optim.Adam(list(self.encoder.parameters()), lr=self.learning_rate) 
    
    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c):
        return self.decoder(z, c)
    
    def discriminat(self, z):
        return self.discriminator(z)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn
    
    def forward2(self, x, c, z_dim):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        real_distribution = torch.normal(mean=0.0,std=1.0,size=(x.shape[0],z_dim)).to(DEVICE)
        dc_real = self.discriminat(real_distribution)
        fwd_rtn = {'dc_fake': dc_fake,
                   'dc_real': dc_real}
        return fwd_rtn
    
    def forward3(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        fwd_rtn = {'dc_fake': dc_fake}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
    
        return losses

    # focal loss only for the discriminator
    def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, lambda_reg=0, logits=True, reduction='mean'):
        if alpha_focal == 0:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
        real_output = fwd_rtn['dc_real']
        fake_output = fwd_rtn['dc_fake']

        # print('alpha_focal: ', alpha_focal)
        # print('gamma_focal: ', gamma_focal)

        loss_real = loss(real_output, torch.ones_like(real_output))
        # print('loss_real: ', loss_real)
        loss_fake = loss(fake_output, torch.zeros_like(fake_output))
        # print('loss_fake: ', loss_fake)
        dc_loss = loss_real + loss_fake  # 0*loss_real because we don't want to train the discriminator on real data
        if alpha_focal == 0:
            dc_loss = 0*loss_real + loss_fake
        # dc_loss = 0
        losses = {'dc_loss':dc_loss}     
        return losses

    def loss_function3(self, x, fwd_rtn):
        loss = nn.BCEWithLogitsLoss()
        fake_output = fwd_rtn['dc_fake']    
        gen_loss = loss(fake_output, torch.ones_like(fake_output))
        losses = {'gen_loss':gen_loss}     
        return losses
    
    
    def pred_latent(self, x, c, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x, c)   
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, c,  DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x, c)
            x_pred = self.decode(mu, c).loc.cpu().detach().numpy()
        return x_pred
    
    def pred_recon_tensor(self, x,c, test_latent, test_var, DEVICE):
        x = torch.FloatTensor(x.numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            x_pred = self.decode(torch.from_numpy(test_latent),c).loc.cpu().detach().numpy()
        return x_pred





class cVAE(nn.Module):
    def __init__(self, 
                input_dim, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                modalities=4,
                non_linear=False):
        
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
        self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)         
        self.discriminator = Discriminator(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 
        self.optimizer1 = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate) 
        self.optimizer2 = optim.Adam(list(self.discriminator.parameters()), lr=self.learning_rate) 
        self.optimizer3 = optim.Adam(list(self.encoder.parameters()), lr=self.learning_rate) 
    
    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c):
        return self.decoder(z, c)
    
    def discriminat(self, z):
        return self.discriminator(z)

    def calc_kl(self, mu, logvar):
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn
    

    def forward_multimodal(self, xes, c):
        self.zero_grad()
        mus, logvars = self.encode(xes, c)
        variance = torch.exp(logvars)
    
        # Mixture of experts of how to get that

        # \mu = \frac{\sum_{m=1}^{M} \frac{\mu_m}{\sigma_m^2}}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}, m is self.modalities
        mu_multimodal = torch.sum(mus / variance, dim=0) / torch.sum(1 / variance, dim=0)
        # variance = \frac {1}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}
        variance_multimodal = 1 / torch.sum(1 / variance, dim=0)

        logvar_multimodal = torch.log(variance_multimodal)

        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
        
        # x_recons = self.decode(z_multimodal, c)
        fwd_rtn = { 'z_multimodal': z_multimodal,
                    'mu_multimodal': mu_multimodal,
                    'logvar_multimodal': logvar_multimodal
                    }
        return fwd_rtn
    
    def forward2(self, x, c, z_dim):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        real_distribution = torch.normal(mean=0.0,std=1.0,size=(x.shape[0],z_dim)).to(DEVICE)
        dc_real = self.discriminat(real_distribution)
        fwd_rtn = {'dc_fake': dc_fake,
                   'dc_real': dc_real}
        return fwd_rtn
    
    def forward3(self, x, c):
        self.zero_grad()
        mu, logvar = self.encode(x, c)
        z = self.reparameterise(mu, logvar)
        dc_fake = self.discriminat(z)
        fwd_rtn = {'dc_fake': dc_fake}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function(self, x, fwd_rtn):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
    
        return losses
    
    # focal loss only for the discriminator
    def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, lambda_reg=0, logits=True, reduction='mean'):
        if alpha_focal == 0:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
        real_output = fwd_rtn['dc_real']
        fake_output = fwd_rtn['dc_fake']

        # print('alpha_focal: ', alpha_focal)
        # print('gamma_focal: ', gamma_focal)

        loss_real = loss(real_output, torch.ones_like(real_output))
        # print('loss_real: ', loss_real)
        loss_fake = loss(fake_output, torch.zeros_like(fake_output))
        # print('loss_fake: ', loss_fake)
        dc_loss = loss_real + loss_fake  # 0*loss_real because we don't want to train the discriminator on real data
        if alpha_focal == 0:
            dc_loss = 0*loss_real + loss_fake
        # dc_loss = 0
        losses = {'dc_loss':dc_loss}     
        return losses




    def loss_function3(self, x, fwd_rtn):
        loss = nn.BCEWithLogitsLoss()
        fake_output = fwd_rtn['dc_fake']    
        gen_loss = loss(fake_output, torch.ones_like(fake_output))
        losses = {'gen_loss':gen_loss}     
        return losses
    
    
    def pred_latent(self, x, c, DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x, c)   
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, x, c,  DEVICE):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, _ = self.encode(x, c)
            x_pred = self.decode(mu, c).loc.cpu().detach().numpy()
        return x_pred
    
    def pred_recon_tensor(self, x,c, test_latent, test_var, DEVICE):
        x = torch.FloatTensor(x.numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            x_pred = self.decode(torch.from_numpy(test_latent),c).loc.cpu().detach().numpy()
        return x_pred
    


class cVAE_multimodal_before_refactor(nn.Module):
    def __init__(self, 
                input_dim_list, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                modalities=3,
                non_linear=False):
        
        super().__init__()
        # self.input_dim = input_dim
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate
        
        # self.encoder = Encoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
        # self.decoder = Decoder(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)         
        # self.discriminator = Discriminator(input_dim=input_dim, hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) 

        # parameter of PoE
        self.alpha_m_list = nn.ParameterList([nn.Parameter(torch.randn(1, requires_grad=True)) for i in range(modalities)])

        # create a list of encoder_list and decoder_list
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.decoder_list = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.discriminator_list = nn.ModuleList([Discriminator(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.discriminator_full = Discriminator(input_dim=sum(input_dim_list), hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)

        # self.optimizer1 = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.learning_rate) 
        # self.optimizer2 = optim.Adam(list(self.discriminator.parameters()), lr=self.learning_rate) 
        # self.optimizer3 = optim.Adam(list(self.encoder.parameters()), lr=self.learning_rate) 
        
        self.optimizer1 = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)
        self.optimizer2 = optim.Adam(list(self.discriminator_list.parameters()), lr=self.learning_rate)
        

        # create a list of optimizer1, optimizer2, optimizer3
        self.optimizer1_list = [optim.Adam(list(self.encoder_list[i].parameters()) + list(self.decoder_list[i].parameters()), lr=self.learning_rate) for i in range(modalities)]
        self.optimizer2_list = [optim.Adam(list(self.discriminator_list[i].parameters()), lr=self.learning_rate) for i in range(modalities)]
        self.optimizer3_list = [optim.Adam(list(self.encoder_list[i].parameters()), lr=self.learning_rate) for i in range(modalities)]

        # self.optimizer_multimodal = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)
        # optimizer_multimodal is used to optimize all the encoder_list and decoder_list together
        # self.optimizer_multimodal = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)
        self.optimizer_multimodal = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list for p in model.parameters()] +
            list(self.alpha_m_list.parameters()),  
            lr=self.learning_rate
        )

    
    def encode(self, x, c, m):
        # return self.encoder(x, c)
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c, m):
        # return self.decoder(z, c)
        return self.decoder_list[m](z, c)
    
    def discriminat(self, z, m):
        # return self.discriminator(z)
        return self.discriminator_list[m](z)

    def calc_kl(self, mu, logvar):
        # return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
        return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def forward(self, x, c, m):
        # self.zero_grad()
        # mu, logvar = self.encode(x, c)
        # z = self.reparameterise(mu, logvar)
        # x_recon = self.decode(z, c)
        # fwd_rtn = {'x_recon': x_recon,
        #             'mu': mu,
        #             'logvar': logvar}
        # return fwd_rtn
        self.zero_grad()
        mu, logvar = self.encode(x, c, m)
        z = self.reparameterise(mu, logvar)
        x_recon = self.decode(z, c, m)
        fwd_rtn = {'x_recon': x_recon,
                    'mu': mu,
                    'logvar': logvar}
        return fwd_rtn
    

    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()
 
        # now i have xes, cs shaped as modalities*numberofpeople*dimension, i want to send them to the encoder, and get the mu and logvar for each modality, the output result would be modalities*numberofpeople. now i know that my model.encoder can take numberofpeople*dimension and give output of numberofpeople*latent_dim. what i want the modalities do is to combine  modalities together using the MoE below. how should i do?
        mus_all = []
        logvars_all = []
        for modality in range(self.modalities):
            mus, logvars = self.encode(xes[modality], cs[modality], modality)
            mus_all.append(mus) 
            logvars_all.append(logvars)
        
        # turn the list of mus and logvars to tensor
        mus = torch.stack(mus_all)
        
        logvars = torch.stack(logvars_all)
            
        variances = torch.exp(logvars)

        mu_multimodal = None
        variance_multimodal = None



        if combine == 'poe':
            # (PoE) Product of experts of how to get that
            # \mu = \frac{\sum_{m=1}^{M} \frac{\mu_m}{\sigma_m^2}}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}, m is self.modalities
            mu_multimodal = torch.sum(mus / variances, dim=0) / torch.sum(1 / variances, dim=0)
            # variance = \frac {1}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}
            variance_multimodal = 1 / torch.sum(1 / variances, dim=0)

            logvar_multimodal = torch.log(variance_multimodal)

            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)


        elif combine == 'gpoe':
            # # (gPoE) generalized Product of experts of how to get that
            # \mu = \frac{\sum_{m=1}^{M} \mu_m \frac{\alpha_m}{\sigma_m^2}}{\sum_{m=1}^{M} \frac{\alpha_m}{\sigma_m^2}}
            # \sigma^2 = \sum_{m=1}^{M} \frac{1}{\frac{\alpha_m}{\sigma_m^2}} .
            # \sum_{m=1}^{M} \alpha_m = 1， and \alpha_m is the weight of each modality, learnable, during the training the model should learn the weight of each modality
            # (PoE) Product of experts
                # Convert ParameterList to list of tensors and apply softmax

            alpha_m_tensors = [param for param in self.alpha_m_list]  # Extract tensors
            alpha_m = torch.softmax(torch.stack(alpha_m_tensors), dim=0)  # Apply softmax and stack

            # reshape alpha_m from 3, 1 to 3, 1, 1
            alpha_m = alpha_m.reshape(alpha_m.shape[0], 1, 1)
            # Weighted average of means
            mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            # Compute combined variance
            variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)

            logvar_multimodal = torch.log(variance_multimodal)

            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)

        elif combine == 'moe':
            # Mixture of Experts (MoE)
            # Weighted average of means
            weights = torch.softmax(torch.stack([torch.ones_like(mu) for mu in mus]), dim=0)
            mu_multimodal = torch.sum(mus * weights, dim=0)
            # Weighted average of variances (assuming independence)
            variance_multimodal = torch.sum(variances * weights, dim=0)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)

        else:
            print('No such combination method')
        
        x_recons = []
        for i in range(self.modalities):
            x_recon = self.decode(z_multimodal, cs[i], i)
            x_recons.append(x_recon)
        
        # x_recons = self.decode(z_multimodal, c)
        fwd_rtn = { 'x_recons': x_recons,
                   'mu_multimodal': mu_multimodal,
                    'logvar_multimodal': logvar_multimodal
                }
        return fwd_rtn
    
    def forward2(self, x, c, z_dim, m):
        self.zero_grad()
        # mu, logvar = self.encode(x, c)
        mu, logvar = self.encode(x, c, m)
        z = self.reparameterise(mu, logvar)
        # dc_fake = self.discriminat(z)
        dc_fake = self.discriminat(z, m)
        real_distribution = torch.normal(mean=0.0,std=1.0,size=(x.shape[0],z_dim)).to(DEVICE)
        # dc_real = self.discriminat(real_distribution)
        dc_real = self.discriminat(real_distribution, m)
        fwd_rtn = {'dc_fake': dc_fake,
                   'dc_real': dc_real}
        return fwd_rtn
    
    def forward3(self, x, c, m):
        self.zero_grad()
        # mu, logvar = self.encode(x, c)
        mu, logvar = self.encode(x, c, m)
        z = self.reparameterise(mu, logvar)
        # dc_fake = self.discriminat(z)
        dc_fake = self.discriminat(z, m)
        fwd_rtn = {'dc_fake': dc_fake}
        return fwd_rtn

    def sample_from_normal(self, normal):
        return normal.loc



    def loss_function_multimodal(self, xes, fwd_rtn):
        losses_list = []
        for i in range(self.modalities):
            # x_recon = fwd_rtn['x_recons']
            x_recon = fwd_rtn['x_recons'][i]
            mu = fwd_rtn['mu_multimodal']
            logvar = fwd_rtn['logvar_multimodal']

            kl = self.calc_kl(mu, logvar)

            # recon = self.calc_ll(x, x_recon)
            recon = self.calc_ll(xes[i], x_recon)

            total = kl - recon
            losses = {'total': total,
                    'kl': kl,
                    'll': recon}
            losses_list.append(losses)
        return losses_list
        
    def loss_function_multimodal(self, xes, fwd_rtn):


        losses = {'total': 0,
                    'kl': 0,
                    'll': 0
                    }
        
        for i in range(self.modalities):
            # x_recon = fwd_rtn['x_recons']
            x_recon = fwd_rtn['x_recons'][i]
            mu = fwd_rtn['mu_multimodal']
            logvar = fwd_rtn['logvar_multimodal']

            kl = self.calc_kl(mu, logvar)

            # recon = self.calc_ll(x, x_recon)
            recon = self.calc_ll(xes[i], x_recon)

            
            total = kl - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon

        return losses


    def loss_function(self, x, fwd_rtn, m):
        x_recon = fwd_rtn['x_recon']
        mu = fwd_rtn['mu']
        logvar = fwd_rtn['logvar']

        kl = self.calc_kl(mu, logvar)
        recon = self.calc_ll(x, x_recon)

        total = kl - recon
        losses = {'total': total,
                'kl': kl,
                'll': recon}
    
        return losses
    


    # focal loss only for the discriminator
    def loss_function2(self, x, fwd_rtn, alpha_focal, gamma_focal, lambda_reg=0, logits=True, reduction='mean'):
        if alpha_focal == 0:
            loss = nn.BCEWithLogitsLoss()
        else:
            loss = FocalLoss(alpha_focal=alpha_focal, gamma_focal=gamma_focal, logits=True, reduction='mean')  # logits=True because we are applying it before the sigmoid activation
        real_output = fwd_rtn['dc_real']
        fake_output = fwd_rtn['dc_fake']

        # print('alpha_focal: ', alpha_focal)
        # print('gamma_focal: ', gamma_focal)

        loss_real = loss(real_output, torch.ones_like(real_output))
        # print('loss_real: ', loss_real)
        loss_fake = loss(fake_output, torch.zeros_like(fake_output))
        # print('loss_fake: ', loss_fake)
        dc_loss = loss_real + loss_fake  # 0*loss_real because we don't want to train the discriminator on real data
        if alpha_focal == 0:
            dc_loss = 0*loss_real + loss_fake
        # dc_loss = 0
        losses = {'dc_loss':dc_loss}     
        return losses


    def loss_function3(self, x, fwd_rtn):
        loss = nn.BCEWithLogitsLoss()
        fake_output = fwd_rtn['dc_fake']    
        gen_loss = loss(fake_output, torch.ones_like(fake_output))
        losses = {'gen_loss':gen_loss}     
        return losses
    
    
    def pred_latent(self, x, c, DEVICE, m):
        x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        c = torch.LongTensor(c).to(DEVICE)
        with torch.no_grad():
            mu, logvar = self.encode(x, c, m)
        latent = mu.cpu().detach().numpy()
        latent_var = logvar.exp().cpu().detach().numpy()
        return latent, latent_var

    def pred_recon(self, xes, c,  DEVICE, combine):
        # x = torch.FloatTensor(x.to_numpy()).to(DEVICE)
        # c = torch.LongTensor(c).to(DEVICE)
        # with torch.no_grad():
        #     mu, _ = self.encode(x, c, m)
        #     x_pred = self.decode(mu, c, m).loc.cpu().detach().numpy()
        # return x_pred
        with torch.no_grad():
            mus_all = []
            logvars_all = []
            tensor_c = torch.tensor(c, dtype=torch.long)
            for modality in range(self.modalities):
                tensor_x = torch.tensor(xes[modality].values, dtype=torch.float32)  # 如果是DataFrame
                # ndarray to tensor
                

                mus, logvars = self.encode(tensor_x, tensor_c, modality)
                mus_all.append(mus) 
                logvars_all.append(logvars)
            
            # turn the list of mus and logvars to tensor
            mus = torch.stack(mus_all)
            logvars = torch.stack(logvars_all)
                
            variances = torch.exp(logvars)
        
            # Mixture of experts of how to get that

            if combine == 'poe':
                # (PoE) Product of experts of how to get that
                # \mu = \frac{\sum_{m=1}^{M} \frac{\mu_m}{\sigma_m^2}}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}, m is self.modalities
                mu_multimodal = torch.sum(mus / variances, dim=0) / torch.sum(1 / variances, dim=0)
                # variance = \frac {1}{\sum_{m=1}^{M} \frac{1}{\sigma_m^2}}
                variance_multimodal = 1 / torch.sum(1 / variances, dim=0)

                logvar_multimodal = torch.log(variance_multimodal)

                z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)


            elif combine == 'gpoe':
                # # (gPoE) generalized Product of experts of how to get that
                # \mu = \frac{\sum_{m=1}^{M} \mu_m \frac{\alpha_m}{\sigma_m^2}}{\sum_{m=1}^{M} \frac{\alpha_m}{\sigma_m^2}}
                # \sigma^2 = \sum_{m=1}^{M} \frac{1}{\frac{\alpha_m}{\sigma_m^2}} .
                # \sum_{m=1}^{M} \alpha_m = 1， and \alpha_m is the weight of each modality, learnable, during the training the model should learn the weight of each modality
                # (PoE) Product of experts
                    # Convert ParameterList to list of tensors and apply softmax
                alpha_m_tensors = [param for param in self.alpha_m_list]  # Extract tensors
                alpha_m = torch.softmax(torch.stack(alpha_m_tensors), dim=0)  # Apply softmax and stack

                # reshape alpha_m from 3, 1 to 3, 1, 1
                alpha_m = alpha_m.reshape(alpha_m.shape[0], 1, 1)
                # Weighted average of means
                mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
                # Compute combined variance
                variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)

                logvar_multimodal = torch.log(variance_multimodal)

                z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)

            elif combine == 'moe':
                # Mixture of Experts (MoE)
                # Weighted average of means
                weights = torch.softmax(torch.stack([torch.ones_like(mu) for mu in mus]), dim=0)
                mu_multimodal = torch.sum(mus * weights, dim=0)
                # Weighted average of variances (assuming independence)
                variance_multimodal = torch.sum(variances * weights, dim=0)
                logvar_multimodal = torch.log(variance_multimodal)
                z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
            


            x_recons = []
            # x_recon = self.decode(z_multimodal, c, m).loc.cpu().detach().numpy()
            for i in range(self.modalities):
                x_recon = self.decode(z_multimodal, tensor_c, i).loc.cpu().detach().numpy()
                x_recons.append(x_recon)
        # print('x_recon:', x_recon)
        # print('x_recon shape:', x_recon.shape)
        # print('x_recons:', x_recons)
        # print('x_recons shape:', len(x_recons))
        return x_recons

    
    def reconstruction_deviation_multimodal(self, xes, x_preds):
        # np.sum((x - x_pred)**2, axis=1)/x.shape[1]

        reconstruction_deviation_list = []
        
        for m in range(self.modalities):
            x = xes[m]
            x_pred = x_preds[m]
            # if m == 0:
            #     reconstruction_deviation = np.sum((x - x_pred)**2, axis=1)/x.shape[1]
            # else:
            #     reconstruction_deviation += np.sum((x - x_pred)**2, axis=1)/x.shape[1]
            reconstruction_deviation = np.sum((x - x_pred)**2, axis=1)/x.shape[1]
            reconstruction_deviation_list.append(reconstruction_deviation)
        return reconstruction_deviation_list

EPS = 1e-8


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """
    def forward(self, mu, var):
        T = 1.0 / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1.0 / torch.sum(T, dim=0)
        pd_logvar = pd_var
        return pd_mu, pd_logvar

class MixtureOfExperts(nn.Module):
    """Return parameters for mixture of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """
    def forward(self, mus, variances):
        weights = torch.softmax(torch.stack([torch.ones_like(mu) for mu in mus]), dim=0)
        mu_multimodal = torch.sum(mus * weights, dim=0)
        variance_multimodal = torch.sum(variances * weights, dim=0)
        return mu_multimodal, variance_multimodal

class mmJSD(nn.Module):
    """Return parameters for mixture of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """
    def forward(self, mus, variances):
        # Calculate the precision (inverse of variance)
        precisions = 1.0 / variances
        
        # Compute the combined multimodal variance (inverse of summed precisions)
        variance_multimodal = 1.0 / torch.sum(precisions, dim=0)
        
        # Compute the combined multimodal mean (weighted sum of means by precision)
        mu_multimodal = torch.sum(mus * precisions, dim=0) * variance_multimodal
        
        return mu_multimodal, variance_multimodal
    

class mmVAEPlus(nn.Module):
    """Return parameters for mixture of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """
    def forward(self, mus, variances):
        weights = torch.softmax(torch.stack([torch.ones_like(mu) for mu in mus]), dim=0)
        mu_multimodal = torch.sum(mus * weights, dim=0)
        variance_multimodal = torch.sum(variances * weights, dim=0)
        return mu_multimodal, variance_multimodal
    
class MVTCAE(nn.Module):
    """Return parameters for mixture of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    logvar (torch.Tensor): Log of variance of experts distribution. M x D for M experts
    """
    def forward(self, mus, variances):
        weights = torch.softmax(torch.stack([torch.ones_like(mu) for mu in mus]), dim=0)
        mu_multimodal = torch.sum(mus * weights, dim=0)
        variance_multimodal = torch.sum(variances * weights, dim=0)
        return mu_multimodal, variance_multimodal


class MoPoE(nn.Module):
    """Return parameters for mixture of product of independent experts.

    Args:
    mu (torch.Tensor): Mean of experts distribution. M x D for M experts
    var (torch.Tensor): Variance of experts distribution. M x D for M experts
    """
    def __init__(self):
        super().__init__()
        self.product_of_experts = ProductOfExperts()
        self.mixture_of_experts = MixtureOfExperts()

    def forward(self, mus, variances):
        # Apply Product of Experts
        poe_mu, poe_var = self.product_of_experts(mus, variances)

        # Stack the results to create a tensor for Mixture of Experts
        mus = torch.cat((mus, poe_mu.unsqueeze(0)), dim=0)
        variances = torch.cat((variances, poe_var.unsqueeze(0)), dim=0)

        # Apply Mixture of Experts
        moe_mu, moe_var = self.mixture_of_experts(mus, variances)
        
        return moe_mu, moe_var



class cVAE_multimodal(nn.Module):
    def __init__(self, 
                input_dim_list, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                modalities=3,
                non_linear=False):
        
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate

        # self.optimizer1 = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)

        self.alpha_m_list = nn.ParameterList([nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(modalities)])
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.decoder_list = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        self.optimizer1 = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list for p in model.parameters()] +
            list(self.alpha_m_list.parameters()),  
            lr=self.learning_rate
        )

    def product_of_experts(self, mus, variances):
        return ProductOfExperts()(mus, variances)
    
    def mixture_of_experts(self, mus, variances):
        return MixtureOfExperts()(mus, variances)
    
    def mixture_of_product_of_experts(self, mus, variances):
        return MoPoE()(mus, variances)

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def combine_latent(self, mus, variances, combine):
        # if mus and variances are only single modality, then return them
        if mus.shape[0] == 1:
            return mus[0], variances[0]
        # print('mus:', mus)
        # print('mus shape:', mus.shape)
        # combine to lower case string
        combine = combine.lower()
        if combine == 'poe':
            mu_multimodal, variance_multimodal = self.product_of_experts(mus, variances)
        elif combine == 'gpoe':
            alpha_m = torch.softmax(torch.stack([param for param in self.alpha_m_list]), dim=0).reshape(self.modalities, 1, 1)
            mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)
        elif combine == 'moe':
            mu_multimodal, variance_multimodal = self.mixture_of_experts(mus, variances)
        elif combine == 'mopoe':
            mu_multimodal, variance_multimodal = self.mixture_of_product_of_experts(mus, variances)
        else:
            raise ValueError('No such combination method')
        return mu_multimodal, variance_multimodal
    
    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()
        # check if xes and cs contain nan or null
        if any([x.isnan().any().item() for x in xes]):
            print('x contains nan')
        if any([c.isnan().any().item() for c in cs]):
            print('c contains nan')
        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
        variances = torch.exp(logvars)

        mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
        logvar_multimodal = torch.log(variance_multimodal)
        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]

        return {'x_recons': x_recons, 'mu_multimodal': mu_multimodal, 'logvar_multimodal': logvar_multimodal}

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            total = kl - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon
        return losses

    def pred_recon(self, xes, c, DEVICE, combine):
        with torch.no_grad():
            tensor_c = torch.tensor(c, dtype=torch.long)
            mus_all, logvars_all = zip(*[self.encode(torch.tensor(xes[i].values, dtype=torch.float32), tensor_c, i) for i in range(self.modalities)])
            mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
            variances = torch.exp(logvars)

            mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
            return [self.decode(z_multimodal, tensor_c, i).loc.cpu().detach().numpy() for i in range(self.modalities)]

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m])**2, axis=1)/xes[m].shape[1] for m in range(self.modalities)]

    # def reconstruction_deviation_multimodal(self, xes, x_preds):
        # return [(xes[m] - x_preds[m])**2 / xes[m].shape[1] for m in range(self.modalities)]

class cVAE_multimodal_endtoend(nn.Module):
    def __init__(self, 
                input_dim_list, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                modalities=3,
                non_linear=False):
        
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate

        # self.optimizer1 = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)

        self.alpha_m_list = nn.ParameterList([nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(modalities)])
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.decoder_list = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        self.optimizer1 = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list for p in model.parameters()] +
            list(self.alpha_m_list.parameters()),  
            lr=self.learning_rate
        )

        input_dim = sum(input_dim_list)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def product_of_experts(self, mus, variances):
        return ProductOfExperts()(mus, variances)
    
    def mixture_of_experts(self, mus, variances):
        return MixtureOfExperts()(mus, variances)
    
    def mixture_of_product_of_experts(self, mus, variances):
        return MoPoE()(mus, variances)

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def combine_latent(self, mus, variances, combine):
        # combine to lower case string
        combine = combine.lower()
        if combine == 'poe':
            mu_multimodal, variance_multimodal = self.product_of_experts(mus, variances)
        elif combine == 'gpoe':
            alpha_m = torch.softmax(torch.stack([param for param in self.alpha_m_list]), dim=0).reshape(self.modalities, 1, 1)
            mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)
        elif combine == 'moe':
            mu_multimodal, variance_multimodal = self.mixture_of_experts(mus, variances)
        elif combine == 'mopoe':
            mu_multimodal, variance_multimodal = self.mixture_of_product_of_experts(mus, variances)
        else:
            raise ValueError('No such combination method')
        return mu_multimodal, variance_multimodal
    
    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()
        # check if xes and cs contain nan or null
        if any([x.isnan().any().item() for x in xes]):
            print('x contains nan')
        if any([c.isnan().any().item() for c in cs]):
            print('c contains nan')
        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
        variances = torch.exp(logvars)

        mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
        logvar_multimodal = torch.log(variance_multimodal)
        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]

        diagnosis = self.mlp(torch.cat(xes, dim=1))

        return {'x_recons': x_recons, 'mu_multimodal': mu_multimodal, 'logvar_multimodal': logvar_multimodal, 'diagnosis': diagnosis}

    def sample_from_normal(self, normal):
        return normal.loc

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0, 'classification': 0}
        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            total = kl - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon
        return losses

    def pred_recon(self, xes, c, DEVICE, combine):
        with torch.no_grad():
            tensor_c = torch.tensor(c, dtype=torch.long)
            mus_all, logvars_all = zip(*[self.encode(torch.tensor(xes[i].values, dtype=torch.float32), tensor_c, i) for i in range(self.modalities)])
            mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
            variances = torch.exp(logvars)

            mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
            return [self.decode(z_multimodal, tensor_c, i).loc.cpu().detach().numpy() for i in range(self.modalities)]

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m])**2, axis=1)/xes[m].shape[1] for m in range(self.modalities)]




import torch.optim as optim
from torch.distributions import Normal, kl_divergence

class mmJSD(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.0001, 
                 modalities=3,
                 non_linear=False):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate

        self.alpha_m_list = nn.ParameterList([nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(modalities)])
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.decoder_list = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        self.optimizer1 = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list for p in model.parameters()] +
            list(self.alpha_m_list.parameters()),  
            lr=self.learning_rate
        )

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def combine_latent(self, mus, logvars):
        variance_multimodal = 1 / torch.sum(1 / torch.exp(logvars), dim=0)
        mu_multimodal = variance_multimodal * torch.sum(mus / torch.exp(logvars), dim=0)
        return mu_multimodal, variance_multimodal

    def multimodal_jsd(self, mus, logvars):
        jsd = 0
        n = len(mus)
        for i in range(n):
            for j in range(i + 1, n):
                kl_ij = kl_divergence(Normal(mus[i], torch.exp(0.5 * logvars[i])), Normal(mus[j], torch.exp(0.5 * logvars[j])))
                jsd += kl_ij.mean()
        return jsd / (n * (n - 1) / 2)

    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()
        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)

        mu_multimodal, variance_multimodal = self.combine_latent(mus, logvars)
        logvar_multimodal = torch.log(variance_multimodal)
        z_multimodal = self.reparameterize(mu_multimodal, logvar_multimodal)

        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]
        return {'x_recons': x_recons, 'mu_multimodal': mu_multimodal, 'logvar_multimodal': logvar_multimodal}

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        jsd_loss = self.multimodal_jsd([fwd_rtn['mu_multimodal']] * self.modalities, [fwd_rtn['logvar_multimodal']] * self.modalities)
        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            total = kl + jsd_loss - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon
        return losses

    def pred_recon(self, xes, c, DEVICE, combine):
        with torch.no_grad():
            tensor_c = torch.tensor(c, dtype=torch.long).to(DEVICE)
            mus_all, logvars_all = zip(*[self.encode(torch.tensor(xes[i].values, dtype=torch.float32).to(DEVICE), tensor_c, i) for i in range(self.modalities)])
            mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
            mu_multimodal, variance_multimodal = self.combine_latent(mus, logvars)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterize(mu_multimodal, logvar_multimodal)
            return [self.decode(z_multimodal, tensor_c, i).loc.cpu().detach().numpy() for i in range(self.modalities)]

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m])**2, axis=1) / xes[m].shape[1] for m in range(self.modalities)]
    
import torch
import torch.nn as nn
import torch.optim as optim

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, s_dim):
        super(VariationalEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim )  # Combined output for private (s_dim) and shared (latent_dim) latents
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim )

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VariationalDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dims, combined_dim):
        super(VariationalDecoder, self).__init__()
        self.fc1 = nn.Linear(combined_dim, hidden_dims[1])
        self.fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc_out = nn.Linear(hidden_dims[0], output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        h = torch.relu(self.fc2(h))
        x_recon = torch.sigmoid(self.fc_out(h))
        return x_recon

class ProductOfExperts2(nn.Module):
    def forward(self, mu, logvar):
        var = torch.exp(logvar)
        var_inv = 1.0 / var
        mu = torch.sum(mu * var_inv, dim=0) / torch.sum(var_inv, dim=0)
        var = 1.0 / torch.sum(var_inv, dim=0)
        logvar = torch.log(var)
        return mu, logvar

class DMVAE(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.0001, 
                 modalities=3,
                 non_linear=False):
        super(DMVAE, self).__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.s_dim = c_dim
        self.beta = 1.0
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.non_linear = non_linear

        self.encoder_list = nn.ModuleList([
            VariationalEncoder(input_dim=input_dim_list[i], hidden_dims=hidden_dim, latent_dim=latent_dim, s_dim=c_dim)
            for i in range(modalities)
        ])
        combined_dim = latent_dim  # Combined dimension for latent space
        self.decoder_list = nn.ModuleList([
            VariationalDecoder(output_dim=input_dim_list[i], hidden_dims=hidden_dim, combined_dim=combined_dim)
            for i in range(modalities)
        ])
        self.join_z = ProductOfExperts2()
        self.optimizer1 = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, x, c, m):
        mu, logvar = self.encoder_list[m](x)
        mu_s, mu_c = mu[:, :self.s_dim], mu[:, self.s_dim:]
        logvar_s, logvar_c = logvar[:, :self.s_dim], logvar[:, self.s_dim:]
        return mu_s, logvar_s, mu_c, logvar_c

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward_multimodal(self, xes, cs, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(xes[i], cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)
        
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)

        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon)
        
        return {'x_recons': x_recons, 'mu_c': mu_c, 'logvar_c': logvar_c}

    def decode(self, z_combined, c, m):
        return self.decoder_list[m](z_combined)

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        kl = 0
        ll = 0
        for i in range(self.modalities):
            kl += -0.5 * torch.sum(1 + fwd_rtn['logvar_c'] - fwd_rtn['mu_c'].pow(2) - torch.exp(fwd_rtn['logvar_c']), dim=1).mean(0)
            ll += -0.5 * torch.sum((xes[i] - fwd_rtn['x_recons'][i]) ** 2, dim=1).mean(0)
        total_loss = kl * self.beta - ll
        losses['total'] = total_loss
        losses['kl'] = kl
        losses['ll'] = ll
        return losses
    
    def pred_recon(self, xes, cs, device, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            # tensor_x = torch.tensor(xes[i].values, dtype=torch.float32).to(device)
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(torch.tensor(xes[i].values, dtype=torch.float32).to(device), cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)  
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)
        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon.cpu().detach().numpy())
        
        return x_recons
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m]) ** 2, axis=1) / xes[m].shape[1] for m in range(self.modalities)]
    













import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class WeightedDMVAE(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.0001, 
                 modalities=3,
                 non_linear=False):
        super(WeightedDMVAE, self).__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.s_dim = c_dim
        self.beta = 1.0
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.non_linear = non_linear

        self.encoder_list = nn.ModuleList([
            VariationalEncoder(input_dim=input_dim_list[i], hidden_dims=hidden_dim, latent_dim=latent_dim, s_dim=c_dim)
            for i in range(modalities)
        ])
        combined_dim = latent_dim  # Combined dimension for latent space
        self.decoder_list = nn.ModuleList([
            VariationalDecoder(output_dim=input_dim_list[i], hidden_dims=hidden_dim, combined_dim=combined_dim)
            for i in range(modalities)
        ])
        self.join_z = ProductOfExperts2()
        # self.weights = nn.Parameter(torch.ones(modalities))  # Initialize weights for each modality
        self.weights = nn.Parameter(torch.abs(torch.randn(modalities)))  # Absolute value to ensure positivity

        self.optimizer1 = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, x, c, m):
        mu, logvar = self.encoder_list[m](x)
        mu_s, mu_c = mu[:, :self.s_dim], mu[:, self.s_dim:]
        logvar_s, logvar_c = logvar[:, :self.s_dim], logvar[:, self.s_dim:]
        return mu_s, logvar_s, mu_c, logvar_c

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward_multimodal(self, xes, cs, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(xes[i], cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)
        
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)

        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon)
        
        return {'x_recons': x_recons, 'mu_c': mu_c, 'logvar_c': logvar_c}

    def decode(self, z_combined, c, m):
        return self.decoder_list[m](z_combined)

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        kl = 0
        ll = 0
        for i in range(self.modalities):
            kl_i = -0.5 * torch.sum(1 + fwd_rtn['logvar_c'] - fwd_rtn['mu_c'].pow(2) - torch.exp(fwd_rtn['logvar_c']), dim=1).mean(0) * self.weights[i]
            ll_i = -0.5 * torch.sum((xes[i] - fwd_rtn['x_recons'][i]) ** 2, dim=1).mean(0) * self.weights[i]
            kl += kl_i
            ll += ll_i
            # Debug prints to check weights application
            print(f"Weight[{i}]: {self.weights[i].item()}, KL[{i}]: {kl_i.item()}, LL[{i}]: {ll_i.item()}")
        
        total_loss = kl - ll
        losses['total'] = total_loss
        losses['kl'] = kl
        losses['ll'] = ll
        return losses
    
    def pred_recon(self, xes, cs, device, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(torch.tensor(xes[i].values, dtype=torch.float32).to(device), cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)  
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)
        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon.cpu().detach().numpy())
        
        return x_recons
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        deviations = []
        for m in range(self.modalities):
            x_m = xes[m].values if isinstance(xes[m], pd.DataFrame) else xes[m]
            x_pred_m = x_preds[m].values if isinstance(x_preds[m], pd.DataFrame) else x_preds[m]
            
            if isinstance(x_m, torch.Tensor):
                x_m = x_m.detach().cpu().numpy()
            if isinstance(x_pred_m, torch.Tensor):
                x_pred_m = x_pred_m.detach().cpu().numpy()

            deviation = np.sum((x_m - x_pred_m) ** 2, axis=1) / x_m.shape[1]
            deviations.append(deviation)
        return deviations






class mvtCAE(nn.Module):
    def __init__(self, 
                input_dim_list, 
                hidden_dim, 
                latent_dim,
                c_dim, 
                learning_rate=0.0001, 
                modalities=3,
                non_linear=False):
        
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.beta = 0.0001

        # self.optimizer1 = optim.Adam(list(self.encoder_list.parameters()) + list(self.decoder_list.parameters()), lr=self.learning_rate)

        self.alpha_m_list = nn.ParameterList([nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(modalities)])
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        self.decoder_list = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        # self.optimizer1 = optim.Adam(nn.ParameterList([p for model in self.encoder_list for p in model.parameters()] + [p for model in self.decoder_list for p in model.parameters()] + list(self.alpha_m_list.parameters()), lr=self.learning_rate))
        self.optimizer1 = optim.Adam(self.parameters(), lr=self.learning_rate)

    def product_of_experts(self, mus, variances):
        return ProductOfExperts2()(mus, variances)
    
    def mixture_of_experts(self, mus, variances):
        return MixtureOfExperts()(mus, variances)
    
    def mixture_of_product_of_experts(self, mus, variances):
        return MoPoE()(mus, variances)

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)
    
    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def combine_latent(self, mus, variances, combine):
        # combine to lower case string
        combine = combine.lower()
        if combine == 'poe':
            mu_multimodal, variance_multimodal = self.product_of_experts(mus, variances)
        elif combine == 'gpoe':
            alpha_m = torch.softmax(torch.stack([param for param in self.alpha_m_list]), dim=0).reshape(self.modalities, 1, 1)
            mu_multimodal = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            variance_multimodal = 1 / torch.sum(alpha_m / variances, dim=0)
        elif combine == 'moe':
            mu_multimodal, variance_multimodal = self.mixture_of_experts(mus, variances)
        elif combine == 'mopoe':
            mu_multimodal, variance_multimodal = self.mixture_of_product_of_experts(mus, variances)
        else:
            raise ValueError('No such combination method')
        
        variance_multimodal = torch.clamp(variance_multimodal, min=1e-6)  # Ensure no negative values

        return mu_multimodal, variance_multimodal
    
    def forward_multimodal(self, xes, cs, combine):
        self.zero_grad()

        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
        variances = torch.exp(logvars)

        # Debugging for variances and mus
        assert not torch.isnan(mus).any(), "mus contains NaN values!"
        assert not torch.isnan(variances).any(), "variances contain NaN values!"

        mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
        assert not torch.isnan(mu_multimodal).any(), "mu_multimodal contains NaN values!"
        assert not torch.isnan(variance_multimodal).any(), "variance_multimodal contains NaN values!"
        
        logvar_multimodal = torch.log(variance_multimodal)

        assert not torch.isnan(mu_multimodal).any(), "mu_multimodal contains NaN values!"
        assert not torch.isnan(logvar_multimodal).any(), "logvar_multimodal contains NaN values!"

        # Debugging the reparameterization
        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
        assert not torch.isnan(z_multimodal).any(), "z_multimodal contains NaN values!"

        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]

        return {'x_recons': x_recons, 'mu_multimodal': mu_multimodal, 'logvar_multimodal': logvar_multimodal, 'qz_xs': mus, 'qz_x': mu_multimodal}

    def sample_from_normal(self, normal):
        return normal.loc
    
    def total_correlation(self, qz_xs, qz_x):
        tc = 0
        for i in range(self.latent_dim):
            log_qz_xi = qz_x[:, i].logsumexp(dim=0) - qz_x[:, i].logsumexp(dim=0).mean()
            log_qz_xi_marginal = torch.stack([qz_xs[j][:, i].logsumexp(dim=0) for j in range(len(qz_xs))]).mean(dim=0)
            tc += log_qz_xi - log_qz_xi_marginal
        return tc

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0, 'tc': 0}
        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            tc = self.total_correlation(fwd_rtn['qz_xs'], fwd_rtn['qz_x'])
            total = kl + 0.00001 * recon + self.beta * tc
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon
            losses['tc'] += tc
        return losses

    def pred_recon(self, xes, c, DEVICE, combine):
        with torch.no_grad():
            tensor_c = torch.tensor(c, dtype=torch.long)
            mus_all, logvars_all = zip(*[self.encode(torch.tensor(xes[i].values, dtype=torch.float32), tensor_c, i) for i in range(self.modalities)])
            mus, logvars = torch.stack(mus_all), torch.stack(logvars_all)
            variances = torch.exp(logvars)

            mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
            logvar_multimodal = torch.log(variance_multimodal)
            z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)
            return [self.decode(z_multimodal, tensor_c, i).loc.cpu().detach().numpy() for i in range(self.modalities)]

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m])**2, axis=1)/xes[m].shape[1] for m in range(self.modalities)]

class mmVAEPlus(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.0001, 
                 modalities=3,
                 non_linear=False):
        super(mmVAEPlus, self).__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.s_dim = c_dim
        self.beta = 0.05
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.non_linear = non_linear

        self.encoder_list = nn.ModuleList([
            VariationalEncoder(input_dim=input_dim_list[i], hidden_dims=hidden_dim, latent_dim=latent_dim, s_dim=c_dim)
            for i in range(modalities)
        ])
        combined_dim = latent_dim  # Combined dimension for latent space
        self.decoder_list = nn.ModuleList([
            VariationalDecoder(output_dim=input_dim_list[i], hidden_dims=hidden_dim, combined_dim=combined_dim)
            for i in range(modalities)
        ])
        self.join_z = ProductOfExperts2()
        self.optimizer1 = optim.Adam(self.parameters(), lr=learning_rate)

    def encode(self, x, c, m):
        mu, logvar = self.encoder_list[m](x)
        mu_s, mu_c = mu[:, :self.s_dim], mu[:, self.s_dim:]
        logvar_s, logvar_c = logvar[:, :self.s_dim], logvar[:, self.s_dim:]
        return mu_s, logvar_s, mu_c, logvar_c

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def forward_multimodal(self, xes, cs, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(xes[i], cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)
        
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)

        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon)
        
        return {'x_recons': x_recons, 'mu_c': mu_c, 'logvar_c': logvar_c}

    def decode(self, z_combined, c, m):
        return self.decoder_list[m](z_combined)

    def loss_function_multimodal(self, xes, fwd_rtn):
        losses = {'total': 0, 'kl': 0, 'll': 0}
        kl = 0
        ll = 0
        for i in range(self.modalities):
            kl += -0.5 * torch.sum(1 + fwd_rtn['logvar_c'] - fwd_rtn['mu_c'].pow(2) - torch.exp(fwd_rtn['logvar_c']), dim=1).mean(0)
            ll += -0.5 * torch.sum((xes[i] - fwd_rtn['x_recons'][i]) ** 2, dim=1).mean(0)
        total_loss = kl * self.beta - ll
        losses['total'] = total_loss
        losses['kl'] = kl
        losses['ll'] = ll
        return losses
    
    def pred_recon(self, xes, cs, device, combine):
        mu_s, logvar_s, mu_c, logvar_c = [], [], [], []
        for i in range(self.modalities):
            # tensor_x = torch.tensor(xes[i].values, dtype=torch.float32).to(device)
            mu_s_i, logvar_s_i, mu_c_i, logvar_c_i = self.encode(torch.tensor(xes[i].values, dtype=torch.float32).to(device), cs, i)
            mu_s.append(mu_s_i)
            logvar_s.append(logvar_s_i)
            mu_c.append(mu_c_i)
            logvar_c.append(logvar_c_i)  
        mu_c = torch.stack(mu_c)
        logvar_c = torch.stack(logvar_c)
        mu_c, logvar_c = self.join_z(mu_c, logvar_c)
        z = self.reparameterize(mu_c, logvar_c)

        x_recons = []
        for i in range(self.modalities):
            z_combined = torch.cat((z, mu_s[i]), dim=1)  # Combine shared latent z with private latent mu_s[i]
            x_recon = self.decode(z_combined, cs, i)
            x_recons.append(x_recon.cpu().detach().numpy())
        
        return x_recons
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def reconstruction_deviation_multimodal(self, xes, x_preds):
        return [np.sum((xes[m] - x_preds[m]) ** 2, axis=1) / xes[m].shape[1] for m in range(self.modalities)]
    
class Classifier(nn.Module):
    def __init__(self, latent_dim, classifier_layers, dropout_rate, num_classes=2):
        super().__init__()
        layers = []
        layer_sizes = [latent_dim] + classifier_layers
        for i in range(len(layer_sizes)-1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, z):
        return self.classifier(z)
    

class cVAE_multimodal_endtoend(nn.Module):
    def __init__(self,
                 input_dim_list,
                 hidden_dim,
                 latent_dim,
                 c_dim,
                 learning_rate=0.0001,
                 modalities=3,
                 non_linear=False,
                 classifier_layers=[128, 64],
                 dropout_rate=0.5,
                 num_classes=2):
        super().__init__()
        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.non_linear = non_linear
        self.num_classes = num_classes
        

        # 编码器列表（共享）
        self.encoder_list = nn.ModuleList([Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        # 健康解码器列表
        self.decoder_list_health = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])
        # 疾病解码器列表
        self.decoder_list_disease = nn.ModuleList([Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear) for i in range(modalities)])

        # 分类器
        self.classifier = Classifier(latent_dim, classifier_layers, dropout_rate, num_classes)

        # 优化器
        self.optimizer = optim.Adam(
            [p for model in self.encoder_list for p in model.parameters()] +
            [p for model in self.decoder_list_health for p in model.parameters()] +
            [p for model in self.decoder_list_disease for p in model.parameters()] +
            list(self.classifier.parameters()),
            lr=self.learning_rate
        )

    def encode(self, xes, cs):
        # xes: 每个模态的输入数据列表
        # cs: 每个模态的协变量列表
        mus_all = []
        logvars_all = []
        for i in range(self.modalities):
            mu, logvar = self.encoder_list[i](xes[i], cs[i])
            mus_all.append(mu)
            logvars_all.append(logvar)
        # 堆叠模态
        mus = torch.stack(mus_all)
        logvars = torch.stack(logvars_all)
        return mus, logvars

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std

    def combine_latent(self, mus, logvars):
        # 使用专家的乘积（Product of Experts）组合潜在变量
        variances = torch.exp(logvars)
        T = 1 / variances
        mu_combined = torch.sum(mus * T, dim=0) / torch.sum(T, dim=0)
        var_combined = 1 / torch.sum(T, dim=0)
        logvar_combined = torch.log(var_combined)
        return mu_combined, logvar_combined

    def decode(self, z, cs, group):
        # group: 'health' 或 'disease'
        x_recons = []
        if group == 'health':
            decoder_list = self.decoder_list_health
        elif group == 'disease':
            decoder_list = self.decoder_list_disease
        else:
            raise ValueError("group must be 'health' or 'disease'")
        for i in range(self.modalities):
            x_recon = decoder_list[i](z, cs[i])
            x_recons.append(x_recon)
        return x_recons

    def forward(self, xes, cs):
        # xes: 每个模态的输入数据列表
        # cs: 每个模态的协变量列表
        mus, logvars = self.encode(xes, cs)
        mu_combined, logvar_combined = self.combine_latent(mus, logvars)
        z = self.reparameterise(mu_combined, logvar_combined)
        # 使用两个解码器进行重建
        x_recons_health = self.decode(z, cs, group='health')
        x_recons_disease = self.decode(z, cs, group='disease')
        # 分类器
        logits = self.classifier(z)
        return {
            'x_recons_health': x_recons_health,
            'x_recons_disease': x_recons_disease,
            'mu': mu_combined,
            'logvar': logvar_combined,
            'logits': logits
        }

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    def calc_recon_loss(self, x, x_recon):
        # x: 原始数据张量
        # x_recon: 解码器的输出（Normal 分布）
        recon_loss = -x_recon.log_prob(x).sum(dim=1).mean()
        return recon_loss

    def compute_deviation(self, x, x_recon):
        # x: 原始数据张量
        # x_recon: 解码器的输出（Normal 分布）
        deviation = ((x - x_recon.mean)**2).mean(dim=1)
        return deviation

    def loss_function(self, xes, fwd_rtn, labels, margin=1.0, weightcontrastive=0.1, weight_kl=0.1, weight_rec=0.1):
        # xes: 每个模态的原始数据张量列表
        # fwd_rtn: forward 方法的输出字典
        # labels: 类别标签张量
        # margin: 对比损失的边距
        # alpha: 对比损失的权重

        recon_loss_health = 0
        recon_loss_disease = 0
        deviations_health = []
        deviations_disease = []

        for i in range(self.modalities):
            x = xes[i]
            x_recon_health = fwd_rtn['x_recons_health'][i]
            x_recon_disease = fwd_rtn['x_recons_disease'][i]

            # 重构损失
            recon_loss_h = self.calc_recon_loss(x, x_recon_health)
            recon_loss_d = self.calc_recon_loss(x, x_recon_disease)
            recon_loss_health += recon_loss_h
            recon_loss_disease += recon_loss_d

            # 计算偏差
            deviation_h = self.compute_deviation(x, x_recon_health)
            deviation_d = self.compute_deviation(x, x_recon_disease)
            deviations_health.append(deviation_h)
            deviations_disease.append(deviation_d)

        # 堆叠模态的偏差，并计算每个样本的平均偏差
        deviation_health = torch.stack(deviations_health).mean(dim=0)
        deviation_disease = torch.stack(deviations_disease).mean(dim=0)

        # 对比损失
        # 对于健康样本（标签为 0），希望 deviation_health < deviation_disease + margin
        # 对于疾病样本（标签为 1），希望 deviation_disease < deviation_health + margin
        contrastive_loss = torch.mean(
            (1 - labels) * F.relu(margin + deviation_health - deviation_disease) +
            labels * F.relu(margin + deviation_disease - deviation_health)
        )

        # KL 散度
        kl_loss = self.calc_kl(fwd_rtn['mu'], fwd_rtn['logvar'])

        # 分类损失
        classification_loss = F.cross_entropy(fwd_rtn['logits'], labels)

        # 总损失
        # total_loss = recon_loss_health + recon_loss_disease + kl_loss + classification_loss + alpha * contrastive_loss
        total_loss = weight_rec * (recon_loss_health + recon_loss_disease) + weight_kl * kl_loss + classification_loss + weightcontrastive * contrastive_loss

        losses = {
            'total_loss': total_loss,
            'recon_loss_health': recon_loss_health,
            'recon_loss_disease': recon_loss_disease,
            'kl_loss': kl_loss,
            'classification_loss': classification_loss,
            'contrastive_loss': contrastive_loss
        }

        return losses

    def predict(self, xes, cs):
        with torch.no_grad():
            mus, logvars = self.encode(xes, cs)
            mu_combined, logvar_combined = self.combine_latent(mus, logvars)
            logits = self.classifier(mu_combined)
            return logits
        


class cVAE_multimodal_regression(nn.Module):
    def __init__(self, 
                 input_dim_list, 
                 hidden_dim, 
                 latent_dim,
                 c_dim, 
                 learning_rate=0.0001, 
                 modalities=3,
                 non_linear=False):
        super().__init__()

        self.input_dim_list = input_dim_list
        self.hidden_dim = hidden_dim + [latent_dim]
        self.latent_dim = latent_dim
        self.c_dim = c_dim
        self.modalities = modalities
        self.learning_rate = learning_rate
        self.non_linear = non_linear

        self.encoder_list = nn.ModuleList([
            Encoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
            for i in range(modalities)
        ])

        self.decoder_list = nn.ModuleList([
            Decoder(input_dim=input_dim_list[i], hidden_dim=self.hidden_dim, c_dim=c_dim, non_linear=non_linear)
            for i in range(modalities)
        ])

        self.alpha_m_list = nn.ParameterList([
            nn.Parameter(torch.randn(1, requires_grad=True)) for _ in range(modalities)
        ])

        # Regression head: simple MLP
        self.regressor = nn.Sequential(
            # nn.Linear(input_dim, 128),
            nn.Linear(sum(input_dim_list), 128),  # Concatenate all modalities
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )


        self.mse_loss = nn.MSELoss()

        self.optimizer1 = torch.optim.Adam(
            list(self.encoder_list.parameters()) +
            list(self.decoder_list.parameters()) +
            list(self.regressor.parameters()) +
            list(self.alpha_m_list.parameters()),
            lr=self.learning_rate
        )

    def product_of_experts(self, mus, variances):
        return ProductOfExperts()(mus, variances)

    def mixture_of_experts(self, mus, variances):
        return MixtureOfExperts()(mus, variances)

    def mixture_of_product_of_experts(self, mus, variances):
        return MoPoE()(mus, variances)

    def encode(self, x, c, m):
        return self.encoder_list[m](x, c)

    def reparameterise(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def decode(self, z, c, m):
        return self.decoder_list[m](z, c)

    def calc_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

    def calc_ll(self, x, x_recon):
        return compute_ll(x, x_recon)

    def combine_latent(self, mus, variances, combine):
        if mus.shape[0] == 1:
            return mus[0], variances[0]
        combine = combine.lower()
        if combine == 'poe':
            return self.product_of_experts(mus, variances)
        elif combine == 'gpoe':
            alpha_m = torch.softmax(torch.stack([param for param in self.alpha_m_list]), dim=0).reshape(self.modalities, 1, 1)
            mu = torch.sum(mus * alpha_m / variances, dim=0) / torch.sum(alpha_m / variances, dim=0)
            var = 1 / torch.sum(alpha_m / variances, dim=0)
            return mu, var
        elif combine == 'moe':
            return self.mixture_of_experts(mus, variances)
        elif combine == 'mopoe':
            return self.mixture_of_product_of_experts(mus, variances)
        else:
            raise ValueError(f"Invalid combine strategy: {combine}")

    def forward_multimodal(self, xes, cs, combine):
        mus_all, logvars_all = zip(*[self.encode(xes[i], cs[i], i) for i in range(self.modalities)])
        mus = torch.stack(mus_all)
        logvars = torch.stack(logvars_all)
        variances = torch.exp(logvars)

        mu_multimodal, variance_multimodal = self.combine_latent(mus, variances, combine)
        logvar_multimodal = torch.log(variance_multimodal)
        z_multimodal = self.reparameterise(mu_multimodal, logvar_multimodal)

        x_recons = [self.decode(z_multimodal, cs[i], i) for i in range(self.modalities)]
        recon_diffs = [xes[i] - x_recons[i].loc for i in range(self.modalities)]
        recon_concat = torch.cat(recon_diffs, dim=1)

        fi_pred = self.regressor(recon_concat)

        return {
            'x_recons': x_recons,
            'mu_multimodal': mu_multimodal,
            'logvar_multimodal': logvar_multimodal,
            'fi_pred': fi_pred
        }

    def loss_function_multimodal(self, xes, fwd_rtn, true_fi, lambda_reg=1.0):
        losses = {'total': 0, 'kl': 0, 'll': 0, 'regression': 0}

        for i in range(self.modalities):
            kl = self.calc_kl(fwd_rtn['mu_multimodal'], fwd_rtn['logvar_multimodal'])
            recon = self.calc_ll(xes[i], fwd_rtn['x_recons'][i])
            total = kl - recon
            losses['total'] += total
            losses['kl'] += kl
            losses['ll'] += recon

        regression_loss = self.mse_loss(fwd_rtn['fi_pred'].squeeze(), true_fi.squeeze())
        losses['regression'] = regression_loss
        losses['total'] += lambda_reg * regression_loss

        return losses