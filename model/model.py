from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from turtle import forward

import tqdm
import logging
import math
from os import path
import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score
from torch.utils import data
from torch.utils.data import DataLoader
#from torch_geometric.nn import GCNConv, GATConv

import time
from model.layers import GCNLayer, SAGELayer, SpGraphAttentionLayer





def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1




class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = (self.dim // 2) + 1
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        # print(emb.shape)
        # print(self.dim)
        # print(emb[:self.dim].shape)
        return emb[:,:self.dim]





class Block(nn.Module):
    def __init__(self, in_ft, out_ft) -> None:
        super(Block, self).__init__()

        self.lin = nn.Linear(in_ft, out_ft)
        self.time = nn.Sequential(
            nn.SiLU(),
            nn.Linear(out_ft, out_ft * 2)
        )


    def forward(self, h, t):

        t = self.time(t)
        
        #print(t.shape)
        scale, shift = t.chunk(2, dim=1)
        #print(scale.shape, shift.shape)

        h = (scale+1) * h + shift

        return h


class Encoder(nn.Module):
    def __init__(self, in_ft, out_ft) -> None:
        super(Encoder, self).__init__()

        self.l1 = Block(in_ft, out_ft) 
        self.l2 = Block(out_ft, out_ft)

        sinu_pos_emb = SinusoidalPosEmb(out_ft)

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(out_ft, out_ft),
            nn.GELU(),
            nn.Linear(out_ft, out_ft)
        )



    def forward(self, h ,t):

    
        t = self.time_mlp(t)

        h = self.l1(h, t)
        h = self.l2(h, t)
        
        return h



class Diffusion(nn.Module):
    def __init__(self, in_feat, out_feat, args) -> None:
        super(Diffusion, self).__init__()

        self.encoder = Encoder(in_feat, out_feat)

        self.timesteps = 50

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=self.timesteps)
        #self.betas = cosine_beta_schedule(timesteps=self.timesteps)



        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    

    # forward diffusion (using the nice property)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def p_losses(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    
        predicted_noise = self.encoder(x_noisy, t)
       
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
        return loss


    def p_sample(self, model, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 


    # Algorithm 2 (including returning all images)
    def p_sample_loop(self, model, shape):
        device = next(model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        #for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
        for i in reversed(range(0, self.timesteps)):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            #imgs.append(F.normalize(img)) #.cpu().numpy())
            imgs.append(img)

        imgs = imgs[::-1]
        steps = [0, int(self.timesteps/8), int(self.timesteps/4), int(self.timesteps/2)]
        
        out = [imgs[step] for step in steps]
        return out


    @torch.no_grad()
    def sample(self, shape):
        return self.p_sample_loop(self.encoder, shape=shape)


    def forward(self, input, device): 
        t = torch.randint(0, self.timesteps, (input.shape[0],), device=device).long()
        return self.p_losses(input, t)
        
