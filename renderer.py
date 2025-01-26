import os
import gc
import sys
import time
import imageio
import torch
from tqdm.auto import tqdm

from models.OmniPlanes import OmniPlanes
from models.YinYang_OmniPlanes import YinYang_OmniPlanes
from utils import *


def volume_renderer(
        rays, 
        times,
        model, 
        chunk=4096, 
        n_coarse=-1, 
        n_fine=0,
        is_train=False, 
        exp_sampling=False, 
        device='cuda', 
        empty_gpu_cache=False, 
        pretrain_envmap=False,
        pivotal_sample_th=0., 
        resampling=False, 
        use_coarse_sample=True, 
        use_palette=False,
        interval_th=False
):
    if pretrain_envmap:
        env_map = model(rays_chunk=rays.to(device), pretrain_envmap=True)
        return env_map

    # Initialize outputs as lists
    rgbs, depth_maps, alphas, acc_maps = [], [], [], []
    bg_maps, env_maps = [], []
    if use_palette:
        final_rgbs, diffuse_rgbs, direct_rgbs = [], [], []
        basis_rgbs, final_colors, soft_colors = [], [], []
        omegas, radiance_maps = [], []
        omega_sparsitys, offset_norms, view_dep_norms = [], [], []
    N_rays_all = rays.shape[0]  # 4096
    start = time.time()
    
    # chunk is equal to batch_size
    for chunk_idx in range(N_rays_all // chunk + int(N_rays_all % chunk > 0)):
        rays_chunk = rays[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        time_chunk = times[chunk_idx * chunk:(chunk_idx + 1) * chunk].to(device)
        if isinstance(model, OmniPlanes) or isinstance(model, YinYang_OmniPlanes):
            '''
               If 'model' is an instance of the class 'OmniPlanes'
               return 'True', and execute the following block
            '''
            # n_coarse = 128, n_fine = 128
            # the stuff fed into network are rays, not points (nerf)
            output_dict = model(rays_chunk,
                                time_chunk,
                                is_train=is_train,
                                n_coarse=n_coarse,
                                n_fine=n_fine,
                                exp_sampling=exp_sampling,
                                pivotal_sample_th=pivotal_sample_th,
                                resampling=resampling,
                                use_coarse_sample=use_coarse_sample, 
                                interval_th=interval_th)
        else:
            output_dict = model(rays_chunk, 
                                time_chunk, 
                                is_train=is_train,
                                N_samples=n_coarse,
                                exp_sampling=exp_sampling)
        if empty_gpu_cache:
            env_map_np = None

            rgb_map_np = output_dict['rgb_map'].cpu().numpy()
            depth_map_np = output_dict['depth_map'].cpu().numpy()
            alpha_np = output_dict['alpha'].cpu().numpy()
            acc_map_np = output_dict['acc_map'].cpu().numpy()    
            
            rgbs.append(rgb_map_np)
            depth_maps.append(depth_map_np)
            alphas.append(alpha_np)
            acc_maps.append(acc_map_np)
            
            if output_dict['env_map'] is not None:
                bg_map_np = output_dict['bg_map'].cpu().numpy()
                env_map_np = output_dict['env_map'].cpu().numpy()
                
                bg_maps.append(bg_map_np)
                env_maps.append(env_map_np)
            
            if use_palette:
                diffuse_rgb_map_np = output_dict['diffuse_rgb_map'].cpu().numpy()
                direct_rgb_map_np = output_dict['direct_rgb_map'].cpu().numpy()
                final_rgb_map_np = output_dict['final_rgb_map'].cpu().numpy()
                
                basis_rgb_map_np = output_dict['basis_rgb_map'].cpu().numpy()
                final_color_map_np = output_dict['final_color_map'].cpu().numpy()
                soft_color_map_np = output_dict['soft_color_map'].cpu().numpy()
                
                omega_map_np = output_dict['omega_map'].cpu().numpy()
                radiance_map_np = output_dict['radiance_map'].cpu().numpy()

                omega_sparsity_map_np = output_dict['omega_sparsity_map'].cpu().numpy()
                offset_norm_map_np = output_dict['offset_norm_map'].cpu().numpy()
                view_dep_norm_np = output_dict['view_dep_norm_map'].cpu().numpy()

                diffuse_rgbs.append(diffuse_rgb_map_np)
                direct_rgbs.append(direct_rgb_map_np)
                final_rgbs.append(final_rgb_map_np)

                basis_rgbs.append(basis_rgb_map_np)
                final_colors.append(final_color_map_np)
                soft_colors.append(soft_color_map_np)
                
                omegas.append(omega_map_np)
                radiance_maps.append(radiance_map_np)
                
                omega_sparsitys.append(omega_sparsity_map_np)
                offset_norms.append(offset_norm_map_np)
                view_dep_norms.append(view_dep_norm_np)   
            
            del output_dict

        else:
            rgbs.append(output_dict['rgb_map'])
            depth_maps.append(output_dict['depth_map'])
            alphas.append(output_dict['alpha'])
            acc_maps.append(output_dict['acc_map'])
            
            if output_dict['env_map'] is not None:
                bg_maps.append(output_dict['bg_map'])
                env_maps.append(output_dict['env_map'])

            if use_palette:
                diffuse_rgbs.append(output_dict['diffuse_rgb_map'])
                direct_rgbs.append(output_dict['direct_rgb_map'])
                final_rgbs.append(output_dict['final_rgb_map'])

                basis_rgbs.append(output_dict['basis_rgb_map'])
                final_colors.append(output_dict['final_color_map'])
                soft_colors.append(output_dict['soft_color_map'])
                
                omegas.append(output_dict['omega_map'])
                radiance_maps.append(output_dict['radiance_map'])
                
                omega_sparsitys.append(output_dict['omega_sparsity_map'])
                offset_norms.append(output_dict['offset_norm_map'])
                view_dep_norms.append(output_dict['view_dep_norm_map'])
                
        if chunk_idx % 100 == 99:
            gc.collect()
            torch.cuda.empty_cache()
    
    if not empty_gpu_cache:
        if not is_train:
            print(f"elapsed time per image: {time.time() - start}")
        if use_palette:
            if output_dict['env_map'] is not None:
                return {"rgb_maps": torch.cat(rgbs),
                        "depth_maps": torch.cat(depth_maps),
                        "bg_maps": torch.cat(bg_maps),
                        "env_maps": torch.cat(env_maps),
                        "alphas": torch.cat(alphas),
                        "acc_maps": torch.cat(acc_maps),
                        
                        "diffuse_rgbs": torch.cat(diffuse_rgbs),
                        "direct_rgbs": torch.cat(direct_rgbs),
                        "final_rgbs": torch.cat(final_rgbs),

                        "basis_rgbs": torch.cat(basis_rgbs),
                        "final_colors": torch.cat(final_colors),
                        "soft_colors": torch.cat(soft_colors),
                        
                        "basis_acc": torch.cat(omegas),
                        "radiance_maps": torch.cat(radiance_maps),
                        
                        "omega_sparsitys": torch.cat(omega_sparsitys),
                        "offset_norms": torch.cat(offset_norms),
                        "view_dep_norms": torch.cat(view_dep_norms)}
            else:
                return {"rgb_maps": torch.cat(rgbs),
                        "depth_maps": torch.cat(depth_maps),
                        "bg_maps": None,
                        "env_maps": None,
                        "alphas": torch.cat(alphas),
                        "acc_maps": torch.cat(acc_maps),

                        "diffuse_rgbs": torch.cat(diffuse_rgbs),
                        "direct_rgbs": torch.cat(direct_rgbs),
                        "final_rgbs": torch.cat(final_rgbs),
                        
                        "basis_rgbs": torch.cat(basis_rgbs),
                        "final_colors": torch.cat(final_colors),
                        "soft_colors": torch.cat(soft_colors),

                        "basis_acc": torch.cat(omegas),
                        "radiance_maps": torch.cat(radiance_maps),
                        
                        "omega_sparsitys": torch.cat(omega_sparsitys),
                        "offset_norms": torch.cat(offset_norms),
                        "view_dep_norms": torch.cat(view_dep_norms)}
        else:
            if output_dict['env_map'] is not None:
                return {"rgb_maps": torch.cat(rgbs),
                        "depth_maps": torch.cat(depth_maps),
                        "bg_maps": torch.cat(bg_maps),
                        "env_maps": torch.cat(env_maps),
                        "alphas": torch.cat(alphas),
                        "acc_maps": torch.cat(acc_maps)}
            else:
                return {"rgb_maps": torch.cat(rgbs),
                        "depth_maps": torch.cat(depth_maps),
                        "bg_maps": None,
                        "env_maps": None,
                        "alphas": torch.cat(alphas),
                        "acc_maps": torch.cat(acc_maps)}
    else:
        if not is_train:
            print(f"elapsed time per image: {time.time() - start}")
        if use_palette:
            if env_map_np is not None:
                return {"rgb_maps": np.concatenate(rgbs),
                        "depth_maps": np.concatenate(depth_maps),
                        "bg_maps": np.concatenate(bg_maps),
                        "env_maps": np.concatenate(env_maps),
                        "alphas": np.concatenate(alphas),
                        "acc_maps": np.concatenate(acc_maps),

                        "diffuse_rgbs": np.concatenate(diffuse_rgbs),
                        "direct_rgbs": np.concatenate(direct_rgbs),
                        "final_rgbs": np.concatenate(final_rgbs),

                        "basis_rgbs": np.concatenate(basis_rgbs),
                        "final_colors": np.concatenate(final_colors),
                        "soft_colors": np.concatenate(soft_colors),
                        
                        "basis_acc": np.concatenate(omegas),
                        "radiance_maps": np.concatenate(radiance_maps),
                        
                        "omega_sparsitys": np.concatenate(omega_sparsitys),
                        "offset_norms": np.concatenate(offset_norms),
                        "view_dep_norms": np.concatenate(view_dep_norms)}
            else:
                return {"rgb_maps": np.concatenate(rgbs),
                        "depth_maps": np.concatenate(depth_maps),
                        "bg_maps": None,
                        "env_maps": None,
                        "alphas": np.concatenate(alphas),
                        "acc_maps": np.concatenate(acc_maps),

                        "diffuse_rgbs": np.concatenate(diffuse_rgbs),
                        "direct_rgbs": np.concatenate(direct_rgbs),
                        "final_rgbs": np.concatenate(final_rgbs),

                        "basis_rgbs": np.concatenate(basis_rgbs),
                        "final_colors": np.concatenate(final_colors),
                        "soft_colors": np.concatenate(soft_colors),
                        
                        "basis_acc": np.concatenate(omegas),
                        "radiance_maps": np.concatenate(radiance_maps),
                        
                        "omega_sparsitys": np.concatenate(omega_sparsitys),
                        "offset_norms": np.concatenate(offset_norms),
                        "view_dep_norms": np.concatenate(view_dep_norms)}
        else:
            if env_map_np is not None:
                return {"rgb_maps": np.concatenate(rgbs),
                        "depth_maps": np.concatenate(depth_maps),
                        "bg_maps": np.concatenate(bg_maps),
                        "env_maps": np.concatenate(env_maps),
                        "alphas": np.concatenate(alphas),
                        "acc_maps": np.concatenate(acc_maps)}
            else:
                return {"rgb_maps": np.concatenate(rgbs),
                        "depth_maps": np.concatenate(depth_maps),
                        "bg_maps": None,
                        "env_maps": None,
                        "alphas": np.concatenate(alphas),
                        "acc_maps": np.concatenate(acc_maps)}


@torch.no_grad()
def evaluation(
        test_dataset,
        model, 
        args, 
        renderer, 
        savePath=None, 
        N_vis=5, 
        prtx='', 
        n_coarse=-1, n_fine=0,
        compute_extra_metrics=True, 
        exp_sampling=False, 
        device='cuda',
        empty_gpu_cache=False, 
        envmap_only=False, 
        resampling=False, 
        use_coarse_sample=True, 
        use_palette=False,
        interval_th=False
):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    model.eval()
    PSNRs, ssims, l_alex, l_vgg = [], [], [], []
    
    # original_palette = model.basis_color.clone().clamp(0., 1.)  # [num_basis, 3]
    # print("original_palette:\n", original_palette)
    # visualize_colors(original_palette, savePath + "_Optimized_Palette.png")

    edit = False
    save = False

    if use_palette:
        if save:
            os.makedirs(savePath + "/out_dict", exist_ok=True)
        elif edit:
            os.makedirs(savePath + "/edited_frames", exist_ok=True)
            os.makedirs(savePath + "/original_frames", exist_ok=True)
        else:
            os.makedirs(savePath + "/recons", exist_ok=True)
            os.makedirs(savePath + "/depth_maps", exist_ok=True)
            os.makedirs(savePath + "/basis_imgs", exist_ok=True)
            os.makedirs(savePath + "/view_dep_imgs", exist_ok=True)
            os.makedirs(savePath + "/diffuse_imgs", exist_ok=True)
    else:
        os.makedirs(savePath + "/recons", exist_ok=True)
        os.makedirs(savePath + "/depth_maps", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far  # [0.1, 300.0]
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)  # 1
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval)) # [0, 1, 2, ..., N_frame-1]

    if use_palette and edit:
        original_palette = model.basis_color.clone().clamp(0., 1.)  # [num_basis, 3]
        print("original_palette:\n", original_palette)
        palette_file_path = os.path.join(savePath, f'palette.pth')
        torch.save(original_palette, palette_file_path)

        novel_palette = original_palette.clone()
        target_color = torch.tensor((1.0, 0.0, 0.0), dtype=novel_palette.dtype) # yellow
        novel_palette[2, :] = target_color
        print("novel_palette:\n", novel_palette)

        visualize_colors(original_palette, savePath + "_Original_Palette.png")
        visualize_colors(novel_palette, savePath + "_Novel_Palette.png")  

        h_diff, s_scale, v_scale = calculate_palette_changes(original_palette, novel_palette)

    if envmap_only:
        samples = test_dataset.all_rays[0]
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        env_map = renderer(rays, model, chunk=16384 * 4, device=device, pretrain_envmap=True)
        env_map = env_map.reshape(H, W, 3).cpu()
        env_map = (env_map.numpy() * 255).astype('uint8')
        imageio.imwrite(f'{savePath}/pretrained_envmap.png', env_map)
        return

    for idx, (samples, sample_times) in enumerate(tqdm(zip(test_dataset.all_rays[0::img_eval_interval], 
                                                           test_dataset.all_times[0::img_eval_interval]), 
                                                  file=sys.stdout)):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])
        M = rays.shape[0]  # H*W
        if use_palette:
            palette = model.basis_color.clone()
            num_basis = palette.shape[0]

        output_dict = renderer(rays=rays, 
                               times=times, 
                               model=model, 
                               chunk=4096, 
                               n_coarse=n_coarse, 
                               n_fine=n_fine,
                               exp_sampling=exp_sampling, 
                               device=device, 
                               empty_gpu_cache=empty_gpu_cache, 
                               resampling=resampling, 
                               use_coarse_sample=use_coarse_sample, 
                               use_palette=use_palette,
                               interval_th=interval_th)
        
        rgb_map = output_dict['rgb_maps']     # (M, 3)
        depth_map = output_dict['depth_maps'] # (M, )
        bg_map = output_dict['bg_maps']       # (M, 3)
        env_map = output_dict['env_maps']     # (M, 3)

        if use_palette:
            diffuse_rgb_map = output_dict['diffuse_rgbs'] # (M, 3)
            final_rgb_map = output_dict['final_rgbs']     # (M, 3)
            basis_rgb_map = output_dict['basis_rgbs']     # (M, num_basis*3)

            if save or edit:
                soft_color_map = output_dict['soft_colors']   # (M, num_basis*3)
                final_color_map= output_dict['final_colors']    # (M, num_basis*3)
                omega_map = output_dict['basis_acc']          # (M, num_basis)
                radiance_map = output_dict['radiance_maps']   # (M, 1)
        
        if empty_gpu_cache:
            rgb_map = rgb_map.clip(0., 1.)  
            rgb_map = torch.from_numpy(rgb_map.reshape(H, W, 3))
            depth_map = torch.from_numpy(depth_map.reshape(H, W))
            if env_map is not None:
                bg_map = torch.from_numpy(bg_map.reshape(H, W, 3))
                env_map = torch.from_numpy(env_map.reshape(H, W, 3))
            if use_palette:
                pred_basis_img = []
                final_rgb_map = torch.from_numpy(final_rgb_map.reshape(H, W, 3))
                diffuse_rgb_map = torch.from_numpy(diffuse_rgb_map.reshape(H, W, 3))
                for i in range(num_basis):
                    basis_img = torch.from_numpy(basis_rgb_map[..., i*3:(i+1)*3].reshape(H, W, 3))
                    pred_basis_img.append(basis_img)
                pred_basis_img = torch.cat(pred_basis_img, dim=1).clip(0., 1.)

                if save or edit:
                    soft_color_map = torch.from_numpy(soft_color_map)   # [M, num_basis*3]
                    pred_basis_img = torch.cat(pred_basis_img, dim=1).clip(0., 1.)
                    final_color_map = torch.from_numpy(final_color_map) # [M, num_basis*3]
                    radiance_map = torch.from_numpy(radiance_map)       # [M, 1]
                    omega_map = torch.from_numpy(omega_map)             # [M, num_basis]

        else:
            rgb_map = rgb_map.clamp(0.0, 1.0)   # limit the range with [0, 1]
            rgb_map = rgb_map.reshape(H, W, 3).cpu()
            depth_map = depth_map.reshape(H, W).cpu()
            if env_map is not None:
                bg_map = bg_map.reshape(H, W, 3).cpu()
                if idx == 0:
                    env_map = env_map.reshape(H, W, 3).cpu()
            
            if use_palette:
                pred_basis_img = []
                final_rgb_map = final_rgb_map.reshape(H, W, 3).cpu()
                diffuse_rgb_map = diffuse_rgb_map.reshape(H, W, 3).cpu()
                for i in range(num_basis):
                    basis_img = basis_rgb_map[..., i*3 : (i+1)*3].reshape(H, W, 3).cpu()
                    pred_basis_img.append(basis_img)
                pred_basis_img = torch.cat(pred_basis_img, dim=1).clip(0., 1.)

                if save or edit:
                    soft_color_map = soft_color_map.cpu()
                    final_color_map = final_color_map.cpu()
                    radiance_map= radiance_map.cpu()
                    omega_map = omega_map.cpu()  
        
        if use_palette and save:
            soft_color_map = soft_color_map.view(-1, 3) # [M * num_basis, 3]
            soft_color_hsv = rgb_to_hsv(soft_color_map)      # [M * num_basis, 3]

            output = {
                "soft_color": soft_color_map,          # [M * num_basis, 3]
                "radiance": radiance_map,              # [M, 1]
                "omega": omega_map,                    # [M, 4]
                "soft_color_hsv": soft_color_hsv,      # [M * num_basis, 3]
                "view_dep_color": view_dep_color_map   # [M, 3]
            }
        
        if use_palette and edit:
            file_path = os.path.join(savePath, 'out_dict', f'{(idx+1):03d}.pth')
            out_dict = torch.load(file_path)  # 'soft_color', 'radiance', 'omega', 'soft_color_hsv', 'view_dep_color'
            
            # soft_color_map = soft_color_map.view(-1, 3) # [M * num_basis, 3]
            # soft_color_hsv = rgb_to_hsv(soft_color_map) # [M * num_basis, 3]
            soft_color_map = out_dict['soft_color']
            soft_color_hsv = out_dict['soft_color_hsv']
            # radiance_map = out_dict['radiance']
            # omega_map = out_dict['omega']
            view_dep_color_map = out_dict['view_dep_color']

            edited_soft_color = apply_palette_changes(M, num_basis, soft_color_hsv, h_diff, s_scale, v_scale)

            original_rgb = compose_palette_bases(M, num_basis, 
                                                 rgb_map, 
                                                 soft_color_map, 
                                                 radiance_map, 
                                                 omega_map)
            original_rgb_map = original_rgb.reshape(H, W, 3).clip(0., 1.)
            edited_rgb = compose_palette_bases(M, num_basis, 
                                               rgb_map, 
                                               edited_soft_color, 
                                               radiance_map, 
                                               omega_map)
            edited_rgb_map = edited_rgb.reshape(H, W, 3).clip(0., 1.)

        
        if not save or not edit:
            if len(test_dataset.all_rgbs):
                gt_rgb = test_dataset.all_rgbs[idxs[idx]].view(H, W, 3)
                if use_palette:
                    mse = torch.mean((final_rgb_map - gt_rgb) ** 2)
                else:
                    mse = torch.mean((rgb_map - gt_rgb) ** 2)
                PSNRs.append(-10.0 * np.log(mse.item()) / np.log(10.0))

                if compute_extra_metrics:
                    if use_palette:
                        ssim = rgb_ssim(final_rgb_map, gt_rgb, 1)
                        l_a = rgb_lpips(gt_rgb.numpy(), final_rgb_map.numpy(), 'alex', model.device)
                        l_v = rgb_lpips(gt_rgb.numpy(), final_rgb_map.numpy(), 'vgg', model.device)
                    else:
                        ssim = rgb_ssim(rgb_map, gt_rgb, 1)
                        l_a = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'alex', model.device)
                        l_v = rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), 'vgg', model.device)
                    
                    ssims.append(ssim)
                    l_alex.append(l_a)
                    l_vgg.append(l_v)

        # Convert results to numpy 'uint8' for visualize into images
        if not save:
            if use_palette:
                if edit:
                    original_rgb_map = (original_rgb_map.numpy() * 255).astype('uint8')
                    edited_rgb_map = (edited_rgb_map.numpy() * 255).astype('uint8')
                else:
                    rgb_map = (rgb_map.numpy() * 255).astype('uint8')
                    depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
                    diffuse_rgb_map = (diffuse_rgb_map.numpy() * 255).astype('uint8')
                    final_rgb_map = (final_rgb_map.numpy() * 255).astype('uint8')
                    pred_basis_img = (pred_basis_img.numpy() * 255).astype('uint8')
            else:
                rgb_map = (rgb_map.numpy() * 255).astype('uint8')
                depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)
                if env_map is not None:
                    bg_map = (bg_map.numpy() * 255).astype('uint8')
                    if idx == 0:
                        env_map = (env_map.numpy() * 255).astype('uint8')

        
        if savePath is not None:
            if use_palette:
                if save:
                    file_path = os.path.join(savePath, 'out_dict', f'{(idx+1):03d}.pth')
                    torch.save(output, file_path)
                elif edit:
                    imageio.imwrite(f'{savePath}/edited_frames/{prtx}{(idx+1):03d}.png', edited_rgb_map)
                    imageio.imwrite(f'{savePath}/original_frames/{prtx}{(idx+1):03d}.png', original_rgb_map)
                else:
                    imageio.imwrite(f'{savePath}/diffuse_imgs/{prtx}{(idx+1):03d}.png', diffuse_rgb_map)
                    imageio.imwrite(f'{savePath}/view_dep_imgs/{prtx}{(idx+1):03d}.png', rgb_map)
                    imageio.imwrite(f'{savePath}/recons/{prtx}{(idx+1):03d}.png', final_rgb_map)
                    imageio.imwrite(f'{savePath}/basis_imgs/{prtx}{(idx+1):03d}.png', pred_basis_img)
                    imageio.imwrite(f'{savePath}/depth_maps/{prtx}{(idx+1):03d}.png', depth_map)
 
            else:
                imageio.imwrite(f'{savePath}/recons/{prtx}{(idx+1):03d}.png', rgb_map)
                imageio.imwrite(f'{savePath}/depth_maps/{prtx}{(idx+1):03d}.png', depth_map)
                if env_map is not None:
                    os.makedirs(savePath + "/envmaps", exist_ok=True)
                    os.makedirs(savePath + "/bgs", exist_ok=True)
                    if idx == 0:
                        imageio.imwrite(f'{savePath}/envmaps/{prtx}envmap.png', env_map)
                    imageio.imwrite(f'{savePath}/bgs/{prtx}{(idx+1):03d}.png', bg_map)
    gc.collect()
    torch.cuda.empty_cache()

    if not save or not edit:
        if PSNRs:
            psnr = np.mean(np.asarray(PSNRs))
            if compute_extra_metrics:
                ssim = np.mean(np.asarray(ssims))
                l_a = np.mean(np.asarray(l_alex))
                l_v = np.mean(np.asarray(l_vgg))
                with open(f"{savePath}/{prtx}mean.txt", "w") as f:
                    f.write(f"PSNR: {psnr} \n")
                    f.write(f"SSIM: {ssim} \n")
                    f.write(f"LPIPS_a: {l_a} \n")
                    f.write(f"LPIPS_v: {l_v}")
                print(f"PSNR: {psnr}, SSIM: {ssim}, LPIPS_a: {l_a}, LPIPS_v: {l_v}")
            else:
                with open(f"{savePath}/{prtx}mean.txt", "w") as f:
                    f.write(f"PSNR: {psnr}")
                print(f"PSNR: {psnr}")
        model.train()
    return PSNRs


@torch.no_grad()
def palette_extract(
        test_dataset,
        model,
        renderer, 
        savePath=None, 
        N_vis=5,
        n_coarse=-1, 
        n_fine=0,
        exp_sampling=False, 
        device='cuda',
        empty_gpu_cache=False,
        resampling=False, 
        use_coarse_sample=True, 
        interval_th=False):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    model.eval()
    all_valid_rgbs_norm = []
    all_valid_positions = [] 
    
    os.makedirs(savePath, exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far  # [0.1, 300.0]
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)  # 1

    for idx, (samples, sample_times, sample_rgbs) in enumerate(tqdm(zip(test_dataset.all_rays[0::img_eval_interval], 
                                                                        test_dataset.all_times[0::img_eval_interval],
                                                                        test_dataset.all_rgbs[0::img_eval_interval]), 
                                                                    file=sys.stdout)):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])
        rgbs = sample_rgbs.view(-1, sample_rgbs.shape[-1])

        output_dict = renderer(
            rays=rays, 
            times=times, 
            model=model, 
            chunk=4096, 
            n_coarse=n_coarse, 
            n_fine=n_fine, 
            exp_sampling=exp_sampling, 
            device=device, 
            empty_gpu_cache=empty_gpu_cache, 
            resampling=resampling,
            use_coarse_sample=use_coarse_sample, 
            interval_th=interval_th)

        depth_rendered = output_dict['depth_maps'] # (M, )
        acc_map = output_dict['acc_maps']          # (M, )

        rgbs_norm, valid_positions = get_valid(rays, rgbs, depth_rendered, acc_map)
        all_valid_rgbs_norm.append(rgbs_norm)
        all_valid_positions.append(valid_positions)

    colors_norm = torch.cat(all_valid_rgbs_norm, dim=0).detach().cpu().numpy()
    positions = torch.cat(all_valid_positions, dim=0).detach().cpu().numpy()
    palette_extraction_input = {"colors": colors_norm, "positions": positions}

    palette_path = os.path.join(savePath, "palette")
    os.makedirs(palette_path, exist_ok=True)

    palette_extraction(palette_extraction_input, 
                       palette_path, H, W,
                       normalize_input = True,
                       error_thres = 5.0 / 255.0)


@torch.no_grad()
def render_for_stabilizer(
        test_dataset,
        model,
        renderer, 
        savePath=None, 
        N_vis=5, 
        prtx='', 
        n_coarse=-1, 
        n_fine=0,
        exp_sampling=False, 
        device='cuda',
        empty_gpu_cache=False, 
        envmap_only=False, 
        resampling=False, 
        use_coarse_sample=True, 
        interval_th=False):
    """
    Evaluate the model on the test rays and compute metrics.
    """
    model.eval()
    rgb_maps, depth_maps = [], [] 
    
    os.makedirs(savePath, exist_ok=True)
    os.makedirs(savePath + "/recons", exist_ok=True)
    os.makedirs(savePath + "/depth_maps", exist_ok=True)
    os.makedirs(savePath + "/envmaps", exist_ok=True)
    os.makedirs(savePath + "/bgs", exist_ok=True)

    try:
        tqdm._instances.clear()
    except Exception:
        pass

    near_far = test_dataset.near_far  # [0.1, 300.0]
    img_eval_interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)  # 1
    idxs = list(range(0, test_dataset.all_rays.shape[0], img_eval_interval)) # [0, 1, 2, ..., N_frame-1]

    for idx, (samples, sample_times) in enumerate(tqdm(zip(test_dataset.all_rays[0::img_eval_interval], 
                                                           test_dataset.all_times[0::img_eval_interval]), 
                                                  file=sys.stdout)):
        W, H = test_dataset.img_wh
        rays = samples.view(-1, samples.shape[-1])
        times = sample_times.view(-1, sample_times.shape[-1])

        output_dict = renderer(
            rays, times, model, chunk=4096, n_coarse=n_coarse, n_fine=n_fine, exp_sampling=exp_sampling, 
            device=device, empty_gpu_cache=empty_gpu_cache, resampling=resampling,
            use_coarse_sample=use_coarse_sample, interval_th=interval_th)
        
        rgb_map = output_dict['rgb_maps']     # (M, 3)
        depth_map = output_dict['depth_maps'] # (M, )
        bg_map = output_dict['bg_maps']       # (M, 3)
        env_map = output_dict['env_maps']     # (M, 3)

        if empty_gpu_cache:
            rgb_map = rgb_map.clip(0., 1.)  
            rgb_map = torch.from_numpy(rgb_map.reshape(H, W, 3))
            depth_map = torch.from_numpy(depth_map.reshape(H, W))
            if env_map is not None:
                bg_map = torch.from_numpy(bg_map.reshape(H, W, 3))
                env_map = torch.from_numpy(env_map.reshape(H, W, 3))
        else:
            rgb_map = rgb_map.clamp(0.0, 1.0)   # limit the range with [0, 1]
            rgb_map = rgb_map.reshape(H, W, 3).cpu()
            depth_map = depth_map.reshape(H, W).cpu()
            if env_map is not None:
                bg_map = bg_map.reshape(H, W, 3).cpu()
                if idx == 0:
                    env_map = env_map.reshape(H, W, 3).cpu()
        
        if env_map is not None:
            bg_map = (bg_map.numpy() * 255).astype('uint8')
            if idx == 0:
                env_map = (env_map.numpy() * 255).astype('uint8')

        depth_map, _ = visualize_depth_numpy(depth_map.numpy(), near_far)

        rgb_map = (rgb_map.numpy() * 255).astype('uint8')
        # gt_rgb_map = (gt_rgb.numpy() * 255).astype('uint8')
        rgb_maps.append(rgb_map)
        depth_maps.append(depth_map)

        if savePath is not None:
            imageio.imwrite(f'{savePath}/recons/{prtx}{(idx+1):03d}.png', rgb_map)
            imageio.imwrite(f'{savePath}/depth_maps/{prtx}{(idx+1):03d}.png', depth_map)
            if env_map is not None:
                if idx == 0:
                    imageio.imwrite(f'{savePath}/envmaps/{prtx}envmap.png', env_map)
                imageio.imwrite(f'{savePath}/bgs/{prtx}{(idx+1):03d}.png', bg_map)
    gc.collect()
    torch.cuda.empty_cache()