from typing import Any, Optional, Union, List
from lightning.pytorch.utilities.types import STEP_OUTPUT
from functools import partial
from contextlib import contextmanager
from torch.optim.lr_scheduler import LambdaLR
from lightning import LightningModule, LightningDataModule
import torch
import torch.nn as nn
import numpy as np
from src.utils.LDM.util import exists, instantiate_from_config
from src.utils.LDM.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from src.utils.LDM.util import default
from src.utils.LDM.modules.ema import LitEma
from src import utils
from tqdm import tqdm
import os
import time


# region Diffusion Block--------------------------------------------------------
######################################Diffusion Block#################################################################
def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                      linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)  # β
    alphas = 1. - betas  # α
    alphas_cumprod = np.cumprod(alphas, axis=0)  #
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    self.num_timesteps = int(timesteps)
    self.linear_start = linear_start
    self.linear_end = linear_end
    assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

    to_torch = partial(torch.tensor, dtype=torch.float32)

    self.register_buffer('betas', to_torch(betas))
    self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
    self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
    self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
    self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
    self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
    self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod) + self.v_posterior * betas
    # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
    self.register_buffer('posterior_variance', to_torch(posterior_variance))
    # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
    self.register_buffer('posterior_mean_coef1', to_torch(
        betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
    self.register_buffer('posterior_mean_coef2', to_torch(
        (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    if self.parameterization == "eps":
        lvlb_weights = self.betas ** 2 / (
                2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
    elif self.parameterization == "x0":
        lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
    else:
        raise NotImplementedError("mu not supported")
    # TODO how to choose this term
    lvlb_weights[0] = lvlb_weights[1]
    self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

    assert not torch.isnan(self.lvlb_weights).all()


def repeat_tensor(self, tensor, dim=0):
    return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)


@contextmanager
def ema_scope(self, context=None):
    if self.use_ema:
        self.model_ema.store(self.parameters())
        self.model_ema.copy_to(self)
        # self.model_ema.store(self.model.parameters())
        # self.model_ema.copy_to(self.model)
        if context is not None:
            # log.info(f"{context}: Switched to EMA weights")
            pass
    try:
        yield None
    finally:
        if self.use_ema:
            self.model_ema.restore(self.parameters())
            if context is not None:
                # log.info(f"{context}: Restored training weights")
                pass


def predict_start_from_noise(self, x_t, t, noise):
    return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )


def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
    posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


def q_sample(self, x_start, t, noise=None):
    """
    This is the forward process step: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise
    Args:
        x_start: x_0
        t: timestep
        noise: standard Gaussian distribution

    Returns: sqrt(alpha_cumprod)* x_0 + sqrt(1 - alpha_cumprod) * noise

    """
    noise = default(noise, lambda: torch.randn_like(x_start))  # [B, 360, 65]
    x_t = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
           extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)  # [64, 360, 65]
    return x_t


def get_loss(self, pred, target, mean=True):
    if self.loss_type == 'l1':
        loss = (target - pred).abs()
        if mean:
            loss = loss.mean()
    elif self.loss_type == 'l2':
        if mean:
            loss = torch.nn.functional.mse_loss(target, pred)
        else:
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
    elif self.loss_type == "huber":
        loss = torch.nn.functional.smooth_l1_loss(pred, target)
    else:
        raise NotImplementedError("unknown loss type '{loss_type}'")

    return loss


def p_losses(self, x_start, cond, timestep, noise=None):
    noise = default(noise, lambda: torch.randn_like(x_start))  # [B, 120, 65]
    x_t = self.q_sample(x_start=x_start, t=timestep, noise=noise)  # [B, 120, 65]
    model_out = self.model(inputs=x_t, cond=cond,
                           time=timestep, processing_state=self.process_state,
                           last_time_stamp=self.num_timesteps)  # x_t: [B, 120, 65]  cond: [B, 120, 27]  time: [32]            # cond_inplanes [6080,1,45] t [6080]  model_out [6080,1,45]

    loss_dict = {}
    if self.parameterization == "eps":
        target = noise
    elif self.parameterization == "x0":
        target = x_start
    else:
        raise NotImplementedError(f"Parameterization {self.parameterization} not yet supported")

    loss = self.get_loss(model_out, target, mean=False)  # [320,120,65]
    # loss = loss.mean(dim=[1, 2, 3])
    loss = loss.mean(dim=[1, 2])  # [B,]

    log_prefix = 'train' if self.training else 'val'

    loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
    loss_simple = loss.mean() * self.l_simple_weight

    loss_vlb = (self.lvlb_weights[timestep] * loss).mean()
    loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

    loss = loss_simple + self.original_elbo_weight * loss_vlb

    loss_dict.update({f'{log_prefix}/loss': loss})

    return loss, loss_dict


def p_mean_variance(self, x, cond, t, clip_denoised: bool):
    # model_out = self.model(x, t)
    model_out = self.model(inputs=x, cond=cond, time=t, processing_state=self.process_state,
                           last_time_stamp=self.timesteps - 1)  # [B, 400, 65]
    # model_out = self.model(x, cond, t)
    if self.parameterization == "eps":
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
    elif self.parameterization == "x0":
        x_recon = model_out
    if clip_denoised:
        x_recon.clamp_(-1., 1.)

    model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
    return model_mean, posterior_variance, posterior_log_variance


@torch.no_grad()
def p_sample(self, x, cond, t, clip_denoised=False, repeat_noise=False):
    '''
    This is the reverse step(Denoise step)
    '''
    b, *_, device = *x.shape, x.device
    model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, clip_denoised=clip_denoised)
    noise = noise_like(x.shape, device, repeat_noise)
    if self.batch_smooth:
        for i in torch.arange(1, noise.shape[0]):
            i = torch.tensor(i).type(torch.long)
            noise[i, :self.concate_length, :] = noise[i - 1, -self.concate_length:, :]
    # no noise when t == 0
    nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
    return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


@torch.no_grad()
def sample(self, cond, shape, return_intermediates=False):  # cond [16,1000, 768]
    return self.p_sample_loop(cond, shape=shape,
                              return_intermediates=return_intermediates)


@torch.no_grad()
def p_sample_loop(self, cond, shape, return_intermediates=False):
    device = self.betas.device
    b = shape[0]
    img = torch.randn(shape, device=device)  # [B, 400, 65]
    if return_intermediates:
        intermediates = [img]
    if self.batch_smooth:
        for i in torch.arange(1, img.shape[0]):
            i = torch.tensor(i).type(torch.long)
            img[i, :self.concate_length, :] = img[i - 1, -self.concate_length:, :]

    for i in reversed(range(0, self.num_timesteps)):
        # for i in tqdm(reversed(range(0, self.num_timesteps)),
        #               total=self.num_timesteps, leave=True, position=0, desc='Sample T',dynamic_ncols=True,
        #               ):
        # self.log("Inference step", i)
        # self.log("Inference_step", float(i),
        #          prog_bar=True, logger=True, on_step=True, on_epoch=False)
        img = self.p_sample(x=img, cond=cond,
                            t=torch.full((b,), i, device=device, dtype=torch.long),
                            clip_denoised=self.clip_denoised)
        if return_intermediates:
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
    if return_intermediates:
        return img, intermediates

    return img

##################################################################################################################
# endregion Diffusion Block
