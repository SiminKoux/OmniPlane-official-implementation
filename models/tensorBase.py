import time
import numpy as np
from math import pi
from einops import rearrange
from typing import List, Optional, Tuple, Union

import torch
import torch.nn
import torch.nn.functional as F

from warnings import warn

from .sh import eval_sh_bases
from .Render_mlp import Render_MLP
from .coordinates import Coordinates
from .envmap import EnvironmentMap


def raw2alpha(sigma: torch.Tensor, dist: torch.Tensor) -> torch.Tensor:
    # sigma, dist are both tensor of shape [N_rays, N_samples], namely [B, N]
    alpha = 1. - torch.exp(-sigma * dist) # opacity of per sampled point
    # T[:, :-1] is the accumulcated transmittance of the entire ray
    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)
    # weights for importance sampling
    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:, -1:]


def SHRender(xyz_sampled: torch.Tensor, 
             viewdirs: torch.Tensor, 
             features: torch.Tensor,
             time: torch.Tensor,
             ) -> torch.Tensor:
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled: torch.Tensor, 
             viewdirs: torch.Tensor, 
             features: torch.Tensor,
             time: torch.Tensor,
             ) -> torch.Tensor:
    rgb = features
    return rgb


def DensityRender(xyz_sampled: torch.Tensor, 
                  viewdirs: torch.Tensor, 
                  features: torch.Tensor,
                  time: torch.Tensor,
                 ) -> torch.Tensor:
    density = features
    return density


class AlphaGridMask(torch.nn.Module):
    def __init__(self, device, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device
        self.alpha_volume = alpha_volume.view(1, 1, *alpha_volume.shape[-3:])

    def sample_alpha(self, norm_samples):
        alpha_vals = F.grid_sample(self.alpha_volume, norm_samples.view(1, -1, 1, 1, 3), align_corners=True).view(-1)
        return alpha_vals


class TensorBase(torch.nn.Module):
    def __init__(
            self, 
            aabb: torch.Tensor,   # bounding box
            gridSize: List[int],  # grid's resolution
            device: torch.device, # cuda or cpu 
            coordinates: Coordinates, # append unique coordinates
            time_grid: int = 300, # the grid_size of time dimension
            num_basis: int = 4, # the number of palette bases
            color_space: str = 'srgb', # 'srgb' or 'linear'
            use_palette: bool = False,  # whether optimize palette or not
            init_palette: torch.Tensor = None,
            init_hist_weights: torch.Tensor = None,
            density_n_comp: Union[int, List[int]] = 8, # R_sigma
            appearance_n_comp: Union[int, List[int]] = 24, # R_c
            density_dim: int = 1, # density feature dimension
            densityMode: str = 'plain',  # which density regression mode to use, 'plain' or 'density_MLP'
            app_dim: int = 27,    # appearance feature dimension
            shadingMode: str = 'render_MLP', # which shading mode to use
            alphaMask: Optional[AlphaGridMask] = None, 
            near_far: torch.Tensor = [2.0, 6.0], 
            density_shift: float = -10,     # density shift for density activation function
            alphaMask_thres: float = 0.001, # density threshold for emptiness mask
            distance_scale: float = 25,     # distance scale for density activation function
            rayMarch_weight_thres: float = 0.0001,   # density threshold for rendering color
            t_pe: int = -1,
            pos_pe: int = 6, 
            density_view_pe: int = -1,
            app_view_pe: int = 6,
            fea_pe: int = 6, 
            featureC: int = 128, 
            step_ratio: float = 2.0, 
            fea2denseAct: str = 'softplus', # feature to density activation function
            init_scale: float = 0.1,   # the scale of gaussian noise for feature planes initialization
            init_shift: float = 0.0,   # the mean of gaussian noise for feature planes initialization
            use_envmap: bool = False, 
            envmap_res_H: int = 1000,
            envmap: Optional[EnvironmentMap] = None, 
            coarse_sigma_grid_update_rule: str = None, # ["conv", "samp"]
            coarse_sigma_grid_reso = None, 
            interval_th: bool = False # force minimum r-grid interval to be r0
    ):
        super().__init__()

        self.aabb = aabb
        self.device = device
        self.time_grid = time_grid
        self.near_far = near_far
        self.step_ratio = step_ratio
        self.update_stepSize(gridSize)
        
        # Density and appearance components numbers
        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.density_dim = density_dim
        self.app_dim = app_dim

        # OmniPlanes weights initialization
        self.init_scale = init_scale
        self.init_shift = init_shift
            
        # Density mask and other acceleration tricks
        self.alphaMask = alphaMask
        self.alphaMask_thres = alphaMask_thres
        self.rayMarch_weight_thres = rayMarch_weight_thres

        # Plane Index
        self.matMode = None  # [[0, 1], [0, 2], [1, 2]]
        self.vecMode = None  # [2, 1, 0]
        self.comp_w = [1, 1, 1]

        # Define tensors for only VM decomposition
        self.density_plane = None
        self.density_line = None
        self.app_plane = None
        self.app_line = None
        self.basis_mat = None

        # Coordinates setting
        self.coordinates = coordinates

        # Paltte bases setting
        self.num_basis = num_basis
        self.color_space = color_space
        self.use_palette = use_palette
        self.init_palette = init_palette
        self.init_hist_weights = init_hist_weights

        # coarse sigma grid updating strategy
        self.coarse_sigma_grid_update_rule = coarse_sigma_grid_update_rule
        
        # Density calculation settings
        self.fea2denseAct = fea2denseAct
        self.density_shift = density_shift
        self.distance_scale = distance_scale

        # Environment map calculation settings
        self.envmap = None
        # self.init_svd_volume(gridSize[0], device) --> moved to children classes
        if use_envmap:
            if envmap is None:
                self.init_envmap(envmap_res_H, init_strategy='random', device=device)
            else:
                self.envmap = EnvironmentMap(h=envmap.emission.shape[2])
                self.envmap.load_envmap(envmap.emission, device=device)

        # Density and appearance regression settings
        self.densityMode = densityMode  # 'density_MLP' or 'plain'
        self.shadingMode = shadingMode  # 'render_MLP' or 'SH' or 'RGB'
        self.t_pe = t_pe       # -1
        self.pos_pe = pos_pe   # -1
        self.density_view_pe = density_view_pe  # -1
        self.app_view_pe = app_view_pe          # 2
        self.fea_pe = fea_pe   # 2
        self.featureC = featureC  # 128
        self.init_density_func(self.densityMode, 
                               self.t_pe, 
                               self.pos_pe, 
                               self.density_view_pe, 
                               self.fea_pe, 
                               self.featureC, 
                               self.device)
        self.init_render_func(self.shadingMode, 
                              self.t_pe, 
                              self.pos_pe, 
                              self.app_view_pe, 
                              self.fea_pe, 
                              self.featureC, 
                              self.device)
        if use_palette:
            self.init_diffuse_func(self.shadingMode, 
                                   self.t_pe, 
                                   self.pos_pe, 
                                   self.density_view_pe,
                                   self.fea_pe, 
                                   self.featureC, 
                                   self.device)

    def init_density_func(self, densityMode, t_pe, pos_pe, view_pe, fea_pe, featureC, device):
        """
            Initialize density regression function
        """
        # Use extracted features directly from 360Plane as density
        if densityMode == 'plain':
            print("self.density_dim:", self.density_dim)
            assert self.density_dim == 1 # Assert the extracted features are scalers
            print("Density Regressor (Indensity):")
            self.density_regressor = DensityRender
        # Use Render_mlp to regress density
        elif densityMode == "density_MLP":
            assert view_pe < 0  # Assert no view position encoding; Density should not depend on view direction
            print("Density Regressor (MLP):")
            self.density_regressor = Render_MLP(self.density_dim, 1, t_pe, fea_pe, pos_pe, view_pe, featureC).to(device)
        else:
            raise NotImplementedError("Unrecognized Density Regression Mode!")
        print("t_pe:", t_pe, "pos_pe:", pos_pe, "view_pe:", view_pe, "fea_pe:", fea_pe)
        print(self.density_regressor)
       
    def init_render_func(self, shadingMode, t_pe, pos_pe, view_pe, fea_pe, featureC, device):
        """ 
            Initialize appearance regression function
        """
        if shadingMode == 'render_MLP':
            print("Appearance Render Module (MLP):")
            self.renderModule = Render_MLP(self.app_dim, 3, t_pe, fea_pe, pos_pe, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            print("Appearance Render Module (SH):")
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            print("Appearance Render Module (Indensity):")
            self.renderModule = RGBRender
        else:
            raise NotImplementedError("Unrecognized Appearance Render Mode!")
        print("t_pe:", t_pe, "pos_pe:", pos_pe, "view_pe:", view_pe, "fea_pe:", fea_pe)
        print(self.renderModule)
    
    def init_diffuse_func(self, shadingMode, t_pe, pos_pe, view_pe, fea_pe, featureC, device):
        """ 
            Initialize diffuse color regression function
        """
        if shadingMode == 'render_MLP':
            print("Diffuse Color Regressor (MLP):")
            self.diffuseModule = Render_MLP(self.app_dim, 3, t_pe, fea_pe, pos_pe, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            print("Diffuse Color Regressor (SH):")
            self.diffuseModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            print("Diffuse Color Regressor (Indensity):")
            self.diffuseModule = RGBRender
        else:
            raise NotImplementedError("Unrecognized Diffuse Color Regressor Mode!")
        print("t_pe:", t_pe, "pos_pe:", pos_pe, "view_pe:", view_pe, "fea_pe:", fea_pe)
        print(self.diffuseModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)

        # like [3, 3, 3]
        self.aabbSize = self.aabb[1] - self.aabb[0]
        # like [2/3, 2/3, 2/3]
        self.invaabbSize = 2.0 / self.aabbSize
        # like [300, 300, 300]
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        # the size of each voxel
        self.units = self.aabbSize / (self.gridSize - 1)

        self.stepSize = torch.mean(self.units) * self.step_ratio
        # use half of diagonal, since our scene is egocentric
        self.aabbHalfDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize))) / 2.0 
        # the number of samples = length of diagonal / stepsize
        self.nSamples = int((self.aabbHalfDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)  # maybe float (not int)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def init_envmap(self, envmap_res_H, init_strategy, device):
        pass

    def compute_features(self, xyz_sampled):
        pass
    
    def compute_densityfeature(self, xyz_sampled, frame_time):
        pass
    
    def compute_appfeature(self, xyz_sampled):
        pass
    
    def normalize_coord(self, xyz_sampled):
        """
            Normalize the sampled coordinates to [-1, 1] range.
        """
        warn('This method is deprecated, use coordinates.normalized_coord instead.', DeprecationWarning, stacklevel=2)
        return (xyz_sampled - self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial=0.02, lr_init_network=0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize':self.gridSize.tolist(),
            'coordinates': self.coordinates,
            'time_grid': self.time_grid,
            'num_basis': self.num_basis,
            'color_space': self.color_space,
            'use_palette': self.use_palette,
            'init_palette': self.init_palette,
            'init_hist_weights': self.init_hist_weights,

            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'near_far': self.near_far,
            'densityMode': self.densityMode,
            'shadingMode': self.shadingMode,
            'alphaMask_thres': self.alphaMask_thres,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'density_shift': self.density_shift,
            'distance_scale': self.distance_scale,

            't_pe': self.t_pe,
            'pos_pe': self.pos_pe,
            'density_view_pe': self.density_view_pe,
            'app_view_pe': self.app_view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'density_dim': self.density_dim,

            'step_ratio': self.step_ratio,
            'fea2denseAct': self.fea2denseAct,
            'init_scale': self.init_scale,
            'init_shift': self.init_shift,
            'use_envmap': self.envmap is not None,
            'envmap': self.envmap,
            'coarse_sigma_grid_update_rule': self.coarse_sigma_grid_update_rule,
        }

    def save(self, path, global_step):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict(), 'global_step': global_step}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape': alpha_volume.shape})
            ckpt.update({'alphaMask.mask': np.packbits(alpha_volume.reshape(-1))})
            # ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        if self.envmap is not None:
            envmap_emission = self.envmap.emission.detach().cpu().numpy()
            ckpt.update({'envmap.emission': envmap_emission})
            ckpt.update({'envmap_res_H': self.envmap.emission.shape[2]})
        torch.save(ckpt, path)

    def load(self, ckpt):
        # if 'alphaMask.aabb' in ckpt.keys():
        if 'alphaMask.shape' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            # self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
            self.alphaMask = AlphaGridMask(self.device, alpha_volume.float().to(self.device))
        if self.envmap is not None:
            self.envmap = EnvironmentMap(h=ckpt['envmap_res_H'])
            self.envmap.load_envmap(emission=ckpt['envmap.emission'], device=self.device)
        self.load_state_dict(ckpt['state_dict'])
        return ckpt['global_step']

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, 
                   rays_o: torch.Tensor, 
                   rays_d: torch.Tensor, 
                   is_train: bool = True, 
                   N_samples: int = -1,
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample points along rays based on the given ray origin and direction

        Args:
            rays_o: [Batch_size, 3], ray origin.
            rays_d: [Batch_size, 3], ray direction.
            is_train: bool, whether in training mode.
            N_samples: int, number of samples along each ray.
        
        Returns:
            rays_pts: [Batch_size, N_samples, 3], sampled points along each ray.
            interpx: [Batch_size, N_samples], sampled points' distance to ray origin.
            ~mask_outbbox: [Batch_size, N_samples], mask for points within bounding box.
        """
        N_samples = N_samples if N_samples > 0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        
        # the start point of sampling
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2], 1) # [bs, samples]
            rng += torch.rand_like(rng)
        step = stepsize * rng.to(rays_o.device)
        
        interpx = (t_min[..., None] + step) # interpolate points
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def sample_ray_exp(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        
        ratio = 1 + pi / N_samples # approximate ratio
        r0 = max((far - near) * (ratio - 1) / (pow(ratio, N_samples) - 1), 0.002)
        
        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng)
        
        interpx = (near + torch.pow(ratio, rng) @ torch.tril(torch.ones(N_samples, N_samples), diagonal=-1).T * r0).to(rays_o.device)
        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)
        # mask_outbbox[...] = False

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self, gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        norm_coords_locs = samples * 2 - 1

        alpha = torch.zeros_like(norm_coords_locs[..., 0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(rearrange(norm_coords_locs[i], 'h w c -> (h w) c'), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        alpha = self.getDenseAlpha(gridSize)
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        # like [1, 1, 300, 300, 300] -> [300, 300, 300]
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, alpha)

        total = torch.sum(alpha)
        print(f"alpha rest %%%f"%(total/total_voxels*100))

    @torch.no_grad()
    def filtering_rays(self, 
                       all_rays: torch.Tensor, 
                       all_rgbs: torch.Tensor, 
                       all_times: torch.Tensor,
                       all_depths: Optional[torch.Tensor] = None, 
                       N_samples: int = 256, 
                       chunk: int = 10240 * 5, 
                       bbox_only: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Filter out rays that are not within the bounding box
        It is used out of "forward function"

        Args:
            all_rays: [N_rays, N_samples, 6], rays_o and rays_d.
            all_rgbs: [N_rays, N_samples, 3], RGB values.
            all_times: [N_rays, N_samples], time values.
            all_depths: [N_rays, N_samples], depth values.
            N_samples: int, number of samples along each ray.
        
        Returns:
            all_rays: [N_rays, N_samples, 6], filtered rays [rays_o, rays_d]
            all_rgbs: [N_rays, N_samples, 3], filtered RGB values.
            all_times: [N_rays, N_samples], filtered time values.
            all_depths: [N_rays, N_samples], filtered depth values.
        """
        
        print('========> filtering rays ...')
        
        tt = time.time()
        # The number of all rays
        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            # certain rays
            rays_chunk = all_rays[idx_chunk].to(self.device)
            # rays_d are unit vectors
            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            
            # Filter based on bounding box
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec   # max point - origin
                rate_b = (self.aabb[0] - rays_o) / vec   # min point - origin
                t_min = torch.minimum(rate_a, rate_b).amax(-1) # .clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1) # .clamp(min=near, max=far)
                mask_inbbox = t_max > t_min
            # Filter based on emptiness mask
            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())
        
        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])
        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')

        if all_depths is None:
            return all_rays[mask_filtered], all_rgbs[mask_filtered]
            # return all_rays[mask_filtered], all_rgbs[mask_filtered], all_times[mask_filtered]
        else:
            return all_rays[mask_filtered], all_rgbs[mask_filtered], all_depths[mask_filtered]
            # return all_rays[mask_filtered], all_rgbs[mask_filtered], all_times[mask_filtered], all_depths[mask_filtered]

    def feature2density(self, density_features):
        """
            feature to density activation function
        """
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, norm_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(norm_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(norm_locs[:,0], dtype=bool)

        sigma = torch.zeros(norm_locs.shape[:-1], device=norm_locs.device)

        if alpha_mask.any():
            sigma_feature = self.compute_densityfeature(norm_locs[alpha_mask])
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma

        alpha = 1 - torch.exp(-sigma*length).view(norm_locs.shape[:-1])
        return alpha

    def forward(self, 
                rays_chunk: torch.Tensor,  # [Batch_size, 6], rays [rays_0, rays_d]
                frame_time: torch.Tensor,  # [Batch_size, 1], time value
                is_train: bool = False,    # whether in training mode
                N_samples: int = -1,       # number of samples along each ray
                exp_sampling: bool = False,    # whether use expotential sampling
                pretrain_envmap: bool = False, # whether pretrain environment map
                ) -> Tuple: 
        # prepare rays (sample points)
        viewdirs = rays_chunk[:, 3:6]
        if pretrain_envmap:
            env_map = self.envmap.get_radiance(viewdirs)
            return env_map

        if exp_sampling:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_exp(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train, N_samples=N_samples)
        # dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat((dists, dists[..., -1:]), dim=-1)  # (N_rays, N_samples)
        
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        frame_time = frame_time.view(-1, 1, 1).expand(xyz_sampled.shape[0], xyz_sampled.shape[1], 1)
        
        # Get the normalized Yinyang coordiantes
        coords_sampled = self.coordinates.from_cartesian(xyz_sampled)
        coords_sampled = self.coordinates.normalize_coord(coords_sampled)

        # If emptiness mask is avaliable, we first filter out rays with low opacities
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(coords_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        # Initialize sigma and rgb values
        sigma = torch.zeros(coords_sampled.shape[:-1], device=coords_sampled.device)
        rgb = torch.zeros((*coords_sampled.shape[:2], 3), device=coords_sampled.device)

        # Compute density feature and density if there are valid rays
        if ray_valid.any():
            sigma_feature = self.compute_densityfeature(coords_sampled[ray_valid], frame_time[ray_valid])
            density_feature = self.density_regressor(coords_sampled[ray_valid],
                                                     viewdirs[ray_valid],
                                                     sigma_feature,
                                                     frame_time[ray_valid])
            validsigma = self.feature2density(density_feature)
            sigma[ray_valid] = validsigma.view(-1)
            # validsigma = self.feature2density(sigma_feature)
            # sigma[ray_valid] = validsigma

        # 'alpha' is the opacity; 'weight' is the accumulated weight; 
        # 'bg_weight' is the accumulated weight for last sampling point
        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        # Compute appearance feature and rgb if there are valid rays
        # Valid rays' weight are above a threshold (rayMarch_weight_thres)
        app_mask = weight > self.rayMarch_weight_thres
        if app_mask.any():
            # app_features = self.compute_appfeature(coords_sampled[app_mask])
            # valid_rgbs = self.renderModule(coords_sampled[app_mask], viewdirs[app_mask], app_features)
            app_features = self.compute_appfeature(coords_sampled[app_mask], frame_time[app_mask]) # [B, N_C+N_F, 27]
            valid_rgbs = self.renderModule(coords_sampled[app_mask], 
                                           viewdirs[app_mask], 
                                           app_features, 
                                           frame_time[app_mask])
            rgb[app_mask] = valid_rgbs
        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        # if white_bg or (is_train and torch.rand((1,))<0.5):
        #     rgb_map = rgb_map + (1. - acc_map[..., None])
        bg_map = None
        env_map = None
        if self.envmap is not None:
            alpha = torch.cat((alpha, torch.ones_like(alpha[..., :1])), dim=-1)
            env_map = self.envmap.get_radiance(viewdirs[:, 0, :])
            bg_map = bg_weight * env_map
            rgb_map = rgb_map + bg_map
            # rgb_map = rgb_map + bg_weight * self.envmap.get_radiance(viewdirs[:, 0, :]).view(-1, 3)

        rgb_map = rgb_map.clamp(0, 1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            # depth_map = depth_map + torch.squeeze(bg_weight) * z_vals[:, -1]
            depth_map = depth_map + (1.0 - acc_map) * rays_chunk[..., -1]  # TODO: check this line

        return rgb_map, depth_map, bg_map, env_map, alpha  # rgb, sigma, alpha, weight, bg_weight
