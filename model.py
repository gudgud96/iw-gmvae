import torch
from torch import nn
import numpy as np
import math as m
from torch.distributions import kl_divergence, Normal
from torch.nn import functional as F
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score


class VAE(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=784, z_dims=64):
        super().__init__()

        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels=1, 
                                    out_channels=8, 
                                    kernel_size=3, 
                                    stride=2, padding=1),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(),
                            nn.Conv2d(in_channels=8, 
                                    out_channels=16, 
                                    kernel_size=3, 
                                    stride=2, padding=1),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU()
                        )

        self.mu, self.var = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.dec_init = nn.Linear(z_dims, hidden_dims)
        self.clf = nn.Linear(z_dims, 10)

        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(16,
                                       8,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(8,
                                       1,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                            nn.BatchNorm2d(1),
                            nn.Sigmoid()
                        )

        self.z_dims = z_dims
    
    def encode(self, x):
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        z = repar(mu, var)
        cls_z_prob = self.clf(z)
    
        return z, mu, var, cls_z_prob
    
    def decode(self, z):
        
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)

        return x_hat

    def forward(self, x):

        z, mu, var, cls_z_prob = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, z, mu, var, None, cls_z_prob
    
    def loss_function(self, step, x_hat, x, mu, var, cls_z_prob, labels, beta=1):
        # kl annealing
        beta_1 = min(step / 1000 * beta, beta)

        recon_loss = torch.nn.BCELoss()(x_hat, x)

        def std_normal(shape):
            N = Normal(torch.zeros(shape), torch.ones(shape))
            if torch.cuda.is_available():
                N.loc = N.loc.cuda()
                N.scale = N.scale.cuda()
            return N

        normal = std_normal(mu.size())
        dis = Normal(mu, var)
        kl_loss = kl_divergence(dis, normal).mean()

        labels = labels.cuda().long()
        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, labels)        
        clf_acc = accuracy_score(torch.argmax(cls_z_prob, dim=-1).cpu().detach().numpy().reshape(-1),
                                labels.cpu().detach().numpy().reshape(-1))

        loss = recon_loss + beta_1 * kl_loss + cls_loss
        
        return loss, recon_loss, kl_loss, clf_acc
        

class IW_VAE(VAE):
    def __init__(self, input_dims=80, hidden_dims=784, z_dims=64,
                num_samples=10):
        super(IW_VAE, self).__init__()
        self.num_samples = num_samples
    
    def encode(self, x):

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        var = var.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        z = repar(mu, var)
        cls_z_prob = self.clf(z.view(z.shape[0] * z.shape[1], z.shape[2]))
    
        return z, mu, var, cls_z_prob
    
    def decode(self, z):
        
        z = z.view(z.shape[0] * z.shape[1], z.shape[2])
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(x_hat.shape[0] // self.num_samples, 
                           self.num_samples, 
                           x_hat.shape[1],
                           x_hat.shape[2],
                           x_hat.shape[3])

        return x_hat

    def forward(self, x):

        z, mu, var, cls_z_prob = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, var, None, cls_z_prob

    def evaluate(self, x):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        z = repar(mu, var)
        cls_z_prob = self.clf(z)
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)

        return x_hat, z, mu, var, None, cls_z_prob

    def loss_function(self, step, x_hat, x, mu, var, cls_z_prob, labels, beta=1):
        # kl annealing
        beta_1 = min(step / 1000 * beta, beta)

        x = x.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) #[B x S x C x H x W]
        recon_loss = torch.nn.BCELoss(reduce=False)(x_hat, x).flatten(2).mean(-1)

        def std_normal(shape):
            N = Normal(torch.zeros(shape), torch.ones(shape))
            if torch.cuda.is_available():
                N.loc = N.loc.cuda()
                N.scale = N.scale.cuda()
            return N

        normal = std_normal(mu.size())        
        dis = Normal(mu, var)
        kl_loss = kl_divergence(dis, normal).mean(-1)

        labels = labels.cuda().long()
        labels = labels.repeat(self.num_samples, 1).permute(1, 0).contiguous().view(-1)
        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, labels)        
        clf_acc = accuracy_score(torch.argmax(cls_z_prob, dim=-1).cpu().detach().numpy().reshape(-1),
                                labels.cpu().detach().numpy().reshape(-1))

        # Get importance weights
        log_weight = (recon_loss + beta_1 * kl_loss)

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0) + cls_loss
        
        return loss, recon_loss.mean(), kl_loss.mean(), clf_acc


class GMVAE(torch.nn.Module):
    def __init__(self, input_dims=80, hidden_dims=784, z_dims=64, n_component=10):
        super().__init__()

        self.encoder = nn.Sequential(
                            nn.Conv2d(in_channels=1, 
                                    out_channels=8, 
                                    kernel_size=3, 
                                    stride=2, padding=1),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(),
                            nn.Conv2d(in_channels=8, 
                                    out_channels=16, 
                                    kernel_size=3, 
                                    stride=2, padding=1),
                            nn.BatchNorm2d(16),
                            nn.LeakyReLU()
                        )

        self.mu, self.var = nn.Linear(hidden_dims, z_dims), nn.Linear(hidden_dims, z_dims)
        self.dec_init = nn.Linear(z_dims, hidden_dims)

        self.decoder = nn.Sequential(
                            nn.ConvTranspose2d(16,
                                       8,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                            nn.BatchNorm2d(8),
                            nn.LeakyReLU(),
                            nn.ConvTranspose2d(8,
                                       1,
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                            nn.BatchNorm2d(1),
                            nn.Sigmoid()
                        )

        self.n_component = n_component
        self.z_dims = z_dims

        self._build_mu_lookup()
        self._build_logvar_lookup(pow_exp=-2)
    
    def encode(self, x):
        
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        z = repar(mu, var)

        log_logits_z, cls_z_prob = self.approx_qy_x(z, self.mu_lookup, 
                                            self.logvar_lookup, 
                                            n_component=self.n_component)
    
        return z, mu, var, log_logits_z, cls_z_prob
    
    def decode(self, z):
        
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)

        return x_hat

    def forward(self, x):

        z, mu, var, log_logits_z, cls_z_prob = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, z, mu, var, log_logits_z, cls_z_prob
    
    def _build_mu_lookup(self):
        mu_lookup = nn.Embedding(self.n_component, self.z_dims)
        nn.init.xavier_uniform_(mu_lookup.weight, gain=1.0)
        mu_lookup.weight.requires_grad = True
        self.mu_lookup = mu_lookup

    def _build_logvar_lookup(self, pow_exp=0, logvar_trainable=False):
        logvar_lookup = nn.Embedding(self.n_component, self.z_dims)
        init_sigma = np.exp(pow_exp)
        init_logvar = np.log(init_sigma ** 2)
        nn.init.constant_(logvar_lookup.weight, init_logvar)
        logvar_lookup.weight.requires_grad = logvar_trainable
        self.logvar_lookup = logvar_lookup

    def approx_qy_x(self, z, mu_lookup, logvar_lookup, n_component):
        def log_gauss_lh(z, mu, logvar):
            """
            Calculate p(z|y), the likelihood of z w.r.t. a Gaussian component
            """
            llh = - 0.5 * (torch.pow(z - mu, 2) / torch.exp(logvar) + logvar + np.log(2 * np.pi))
            llh = torch.sum(llh, dim=1)  # sum over dimensions
            return llh

        logLogit_qy_x = torch.zeros(z.shape[0], n_component).cuda()  # log-logit of q(y|x)
        for k_i in torch.arange(0, n_component):
            mu_k, logvar_k = mu_lookup(k_i.cuda()), logvar_lookup(k_i.cuda())
            logLogit_qy_x[:, k_i] = log_gauss_lh(z, mu_k, logvar_k) + np.log(1 / n_component)

        qy_x = torch.nn.functional.softmax(logLogit_qy_x, dim=1)
        return logLogit_qy_x, qy_x
    
    def loss_function(self, step, x_hat, x, mu, var, cls_z_prob, labels, beta=1):
        # kl annealing
        beta_1 = min(step / 1000 * beta, beta)

        recon_loss = torch.nn.BCELoss()(x_hat, x)
        recon_loss_2 = torch.nn.BCELoss(reduce=False)(x_hat, x).mean(-1)

        mu_ref, var_ref = self.mu_lookup(labels.cuda().long()), \
                        self.logvar_lookup(labels.cuda().long()).exp_()
        dis_ref = Normal(mu_ref, var_ref)
        dis = Normal(mu, var)
        kl_loss = kl_divergence(dis, dis_ref).mean()

        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, labels.cuda().long())        
        clf_acc = accuracy_score(torch.argmax(cls_z_prob, dim=-1).cpu().detach().numpy().reshape(-1),
                                labels.cpu().detach().numpy().reshape(-1))

        loss = recon_loss + beta_1 * kl_loss + cls_loss
        
        return loss, recon_loss, kl_loss, clf_acc
        

class IW_GMVAE(GMVAE):
    def __init__(self, input_dims=80, hidden_dims=784, z_dims=64, n_component=10,
                num_samples=10):
        super(IW_GMVAE, self).__init__()
        self.num_samples = num_samples
    
    def encode(self, x):

        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        mu = mu.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        var = var.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        z = repar(mu, var)

        log_logits_z, cls_z_prob = self.approx_qy_x(z.view(z.shape[0] * z.shape[1], z.shape[2]), 
                                            self.mu_lookup, 
                                            self.logvar_lookup, 
                                            n_component=self.n_component)
    
        return z, mu, var, log_logits_z, cls_z_prob
    
    def decode(self, z):
        
        z = z.view(z.shape[0] * z.shape[1], z.shape[2])
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)
        x_hat = x_hat.view(x_hat.shape[0] // self.num_samples, 
                           self.num_samples, 
                           x_hat.shape[1],
                           x_hat.shape[2],
                           x_hat.shape[3])

        return x_hat

    def forward(self, x):

        z, mu, var, log_logits_z, cls_z_prob = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, var, log_logits_z, cls_z_prob
    
    def evaluate(self, x):
        def repar(mu, stddev, sigma=1):
            eps = Normal(0, sigma).sample(sample_shape=stddev.size()).cuda()
            z = mu + stddev * eps  # reparameterization trick
            return z
        
        q_zx = self.encoder(x)
        q_zx_flatten = q_zx.view(q_zx.shape[0], -1)
        mu, var = self.mu(q_zx_flatten), self.var(q_zx_flatten).exp_()
        z = repar(mu, var)
        h = self.dec_init(z)
        h = h.view(h.shape[0], 16, 7, 7)
        x_hat = self.decoder(h)

        _, cls_z_prob = self.approx_qy_x(z, 
                                        self.mu_lookup, 
                                        self.logvar_lookup, 
                                        n_component=self.n_component)

        return x_hat, z, mu, var, None, cls_z_prob
    
    def loss_function(self, step, x_hat, x, mu, var, cls_z_prob, labels, beta=1):
        # kl annealing
        beta_1 = min(step / 1000 * beta, beta)

        x = x.repeat(self.num_samples, 1, 1, 1, 1).permute(1, 0, 2, 3, 4) #[B x S x C x H x W]
        recon_loss = torch.nn.BCELoss(reduce=False)(x_hat, x).flatten(2).mean(-1)

        mu_ref, var_ref = self.mu_lookup(labels.cuda().long()), \
                            self.logvar_lookup(labels.cuda().long()).exp_()
        mu_ref = mu_ref.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]
        var_ref = var_ref.repeat(self.num_samples, 1, 1).permute(1, 0, 2) # [B x S x D]

        dis_ref = Normal(mu_ref, var_ref)
        dis = Normal(mu, var)
        kl_loss = kl_divergence(dis, dis_ref).mean(-1)

        labels = labels.cuda().long()
        labels = labels.repeat(self.num_samples, 1).permute(1, 0).contiguous().view(-1)
        cls_loss = torch.nn.CrossEntropyLoss()(cls_z_prob, labels)        
        clf_acc = accuracy_score(torch.argmax(cls_z_prob, dim=-1).cpu().detach().numpy().reshape(-1),
                                labels.cpu().detach().numpy().reshape(-1))

        # Get importance weights
        log_weight = (recon_loss + beta_1 * kl_loss)

        # Rescale the weights (along the sample dim) to lie in [0, 1] and sum to 1
        weight = F.softmax(log_weight, dim=-1)

        loss = torch.mean(torch.sum(weight * log_weight, dim=-1), dim=0) + cls_loss
        
        return loss, recon_loss.mean(), kl_loss.mean(), clf_acc
        


