import gc
import os, sys
import random
from tqdm.auto import tqdm
from opt import recursive_config_parser, export_config

from renderer import volume_renderer, evaluation, render_for_stabilizer, palette_extract
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from models.OmniPlanes import OmniPlanes
from models.YinYang_OmniPlanes import YinYang_OmniPlanes
from dataLoader import dataset_dict
from models import coordinates_dict
from sampler import SimpleSampler, ThetaImportanceSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def render_test(args):
    if args.metric_only:
        raise NotImplementedError
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "test",
        "is_stack": True,
        "use_gt_depth": args.use_gt_depth,
        "downsample": 1,
        "near_far": args.near_far,
        "roi": args.roi,
        "localization_method": args.localization_method,
        "skip": 1,
    }
    test_dataset = dataset(**load_params)

    if args.ckpt is None:
        if args.use_palette:
            ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}_palette.th')
        else:
            ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}.th')
    else:
        ckpt = args.ckpt

    if not os.path.exists(ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # Evaluation
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)
    renderer = volume_renderer

    if args.ckpt is None:
        if args.use_palette:
            evaluation_dir = f'{args.basedir}/{args.expname}/evaluation_palette'
        else:
            evaluation_dir = f'{args.basedir}/{args.expname}/evaluation'
    else:
        evaluation_dir = f'{args.basedir}/{args.expname}/evaluation_{args.ckpt.split(".")[-2].split("_")[-1]}'
    os.makedirs(evaluation_dir, exist_ok=True)

    evaluation(
        test_dataset, 
        model, 
        args, 
        renderer, 
        evaluation_dir, 
        N_vis=-1, 
        n_coarse=args.n_coarse, 
        n_fine=args.n_fine,
        device=device, 
        exp_sampling=args.exp_sampling, 
        compute_extra_metrics=True,
        resampling=args.resampling, 
        empty_gpu_cache=True, 
        use_coarse_sample=args.use_coarse_sample, 
        use_palette=args.use_palette,
        interval_th=args.interval_th,
        test=True,
        edit=False,
        recolor=False,
        lighting=False,
        texture=False,
        mask=False,
        save=False
    )


@torch.no_grad()
def render_edit(args):
    if args.metric_only:
        raise NotImplementedError
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "test",
        "is_stack": True,
        "use_gt_depth": args.use_gt_depth,
        "downsample": 1,
        "near_far": args.near_far,
        "roi": args.roi,
        "localization_method": args.localization_method,
        "skip": 1,
    }
    test_dataset = dataset(**load_params)

    if args.ckpt is None:
        ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}_palette.th')
    else:
        ckpt = args.ckpt

    if not os.path.exists(ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # Evaluation
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)
    renderer = volume_renderer

    edit_results_dir = f'{args.basedir}/{args.expname}/edit_results'
    os.makedirs(edit_results_dir, exist_ok=True)

    evaluation(
        test_dataset, 
        model, 
        args, 
        renderer, 
        edit_results_dir, 
        N_vis=-1, 
        n_coarse=args.n_coarse, 
        n_fine=args.n_fine,
        device=device, 
        exp_sampling=args.exp_sampling, 
        compute_extra_metrics=True,
        resampling=args.resampling, 
        empty_gpu_cache=True, 
        use_coarse_sample=args.use_coarse_sample, 
        use_palette=args.use_palette,
        interval_th=args.interval_th,
        test=False,
        edit=True,
        recolor=args.recolor,
        lighting=args.relighting,
        texture=args.retexture,
        mask=args.visualize_seg,
        save=args.save_edit_dict
    )

@torch.no_grad()
def stabilizer(args):
    if args.metric_only:
        raise NotImplementedError
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "test",
        "is_stack": True,
        "use_gt_depth": args.use_gt_depth,
        "downsample": 1,
        "near_far": args.near_far,
        "roi": args.roi,
        "localization_method": args.localization_method,
        "skip": 1
    }
    test_dataset = dataset(**load_params)

    if args.ckpt is None:
        ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}.th')
    else:
        ckpt = args.ckpt
    print("ckpt:", ckpt)

    if not os.path.exists(ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # Evaluation
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)
    renderer = volume_renderer

    if args.ckpt is None:
        stabilizer_dir = f'{args.basedir}/{args.expname}/stabilizer'
    else:
        stabilizer_dir = f'{args.basedir}/{args.expname}/stabilizer_{args.ckpt.split(".")[-2].split("_")[-1]}'
    os.makedirs(stabilizer_dir, exist_ok=True)

    render_for_stabilizer(
        test_dataset, 
        model,
        renderer, 
        stabilizer_dir, 
        N_vis=-1, 
        n_coarse=args.n_coarse, 
        n_fine=args.n_fine,
        device=device, 
        exp_sampling=args.exp_sampling,
        resampling=args.resampling, 
        empty_gpu_cache=True, 
        use_coarse_sample=args.use_coarse_sample, 
        interval_th=args.interval_th
    )

@torch.no_grad()
def palette_extractor(args):
    if args.metric_only:
        raise NotImplementedError
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    load_params = {
        "data_dir": args.datadir,
        "split": "test",
        "is_stack": True,
        "use_gt_depth": args.use_gt_depth,
        "downsample": 1,
        "near_far": args.near_far,
        "roi": args.roi,
        "localization_method": args.localization_method,
        "skip": 1
    }
    test_dataset = dataset(**load_params)

    if args.ckpt is None:
        ckpt = os.path.join(f'{args.basedir}/{args.expname}/{args.expname}.th')
    else:
        ckpt = args.ckpt
    print("ckpt:", ckpt)

    if not os.path.exists(ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})

    # Evaluation
    model = eval(args.model_name)(**kwargs)
    model.load(ckpt)
    renderer = volume_renderer

    palette_dir = f'{args.basedir}/{args.expname}/palette'
    os.makedirs(palette_dir, exist_ok=True)

    palette_extract(
        test_dataset=test_dataset, 
        model=model,
        renderer=renderer, 
        savePath=palette_dir, 
        N_vis=-1, 
        n_coarse=args.n_coarse, 
        n_fine=args.n_fine,
        device=device, 
        exp_sampling=args.exp_sampling,
        resampling=args.resampling, 
        empty_gpu_cache=True, 
        use_coarse_sample=args.use_coarse_sample, 
        interval_th=args.interval_th
    )

def palette_train(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]  # dataset_dict = {'llff','omnivideos','omniblender','omniscenes'}
    load_params = {
        "data_dir": args.datadir,
        "split": "train",
        "is_stack": False,
        "use_gt_depth": args.use_gt_depth,
        "downsample": args.downsample_train,
        "near_far": args.near_far,
        "localization_method": args.localization_method,
    }

    # Load train dataset (not stack)
    train_dataset = dataset(**load_params)

    # Load test dataset (stack)
    load_params["split"] = "test"
    load_params["is_stack"] = True
    load_params["downsample"] = args.downsample_test
    load_params["skip"] = args.test_skip  # only test dataset can have non one skip value
    test_dataset = dataset(**load_params)
    near_far = train_dataset.near_far  # [0.1, 300.0]

    # init resolution
    upsamp_list = args.upsamp_list  # [1000000000000000]
    update_AlphaMask_list = args.update_AlphaMask_list  # [10000000000000]
    vis_list = args.vis_list   # [10000, 50000, 100000]
    
    n_lamb_sigma = args.n_lamb_sigma # R_sigma [16, 16, 16]
    n_lamb_sh = args.n_lamb_sh       # R_c [48, 48, 48]

    # init log file (Save outputs)
    if args.add_timestamp: # the timestamp here means when run this program
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}' 
    os.makedirs(logfolder, exist_ok=True)
    export_config(args, logfolder)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # 1. Initialize bounding box
    aabb = train_dataset.scene_bbox.to(device)  # Tensor: of shape [2, 3]

    # 2. Initialize specific coordinates
    coordinates = coordinates_dict[args.coordinates_name]
    is_yinyang = args.coordinates_name == 'yinyang'  # True or False
    is_general_sphere = args.coordinates_name == 'generic_sphere'  # True or False
    # for backward compatibility
    if is_yinyang or is_general_sphere:
        coordinates = coordinates(device, 
                                  aabb, 
                                  exp_r=args.exp_sampling, 
                                  N_voxel=args.N_voxel_init, 
                                  r0=args.r0, 
                                  interval_th=args.interval_th)
    else:
        coordinates = coordinates(device, aabb)
    
    # 3. Initialize resolution
    reso_cur = coordinates.N_to_reso(args.N_voxel_init, aabb)  # list(len=3): grid size (three dims)
    if not is_yinyang:
        coordinates.set_resolution(reso_cur)
    
    # 4. Initialize the number of samplings
    n_coarse = args.n_coarse                        # 128
    n_fine = args.n_fine if args.resampling else 0  # 128
    use_coarse_sample = args.use_coarse_sample      # True
    if args.resampling:
        coarse_sigma_grid_update_rule = args.coarse_sigma_grid_update_rule
        if coarse_sigma_grid_update_rule == "conv":
            coarse_sigma_grid_update_step = 1  # This one
        elif coarse_sigma_grid_update_rule == "samp":
            raise NotImplementedError
        else:
            coarse_sigma_grid_update_step = 1000000000
    else:
        coarse_sigma_grid_update_rule = None
        coarse_sigma_grid_update_step = 1000000000

    # 5. Load the Palette Base Initialization
    # get initialized palette from the extracted palette file.
    extracted_palette = None
    extracted_hist_weights = None
    num_basis = None
    if args.use_palette:
        palette_workspace = f'{args.basedir}/{args.expname}/palette'
        assert os.path.exists(os.path.join(palette_workspace, 'palette.npz')), "Extracted palette is missing."
        extracted_palette = np.load(os.path.join(palette_workspace, 'palette.npz'))['palette']                 # (4, 3)
        extracted_hist_weights = np.load(os.path.join(palette_workspace, 'hist_weights.npz'))['hist_weights']  # (32, 32, 32, 4)
        num_basis = extracted_palette.shape[0]   # 4  

    # 6. Initialize the volume render
    renderer = volume_renderer

    # 7. Initialize the neural model
    ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.th')]
    # Create the model
    start = 0
    model = eval(args.model_name)(
        aabb = aabb,     # bounding box
        gridSize = reso_cur, # resolution
        device = device,   # cuda or cpu
        coordinates = coordinates, # 'xyz', 'sphere', 'cylinder', 'balanced_sphere', ... , 'yinyang', 'generic_sphere'
        time_grid=args.time_grid,     # time_grid: 300
        num_basis=num_basis,          # the number of palette bases: 4
        color_space=args.color_space, # 'srgb' or 'linear'
        use_palette=args.use_palette,         # True or False
        init_palette=extracted_palette,
        init_hist_weights=extracted_hist_weights,

        density_n_comp=n_lamb_sigma,  # R_sigma: [16, 16, 16]
        appearance_n_comp=n_lamb_sh,  # R_c: [48, 48, 48]
        app_dim=args.data_dim_color,  # dimension of color/appearance feature: 27
        near_far=near_far,            # [0.1, 300.0]
        densityMode=args.densityMode, # density_MLP
        shadingMode=args.shadingMode, # render_MLP
        alphaMask_thres=args.alpha_mask_thre, # threshold for creating alpha mask volume: 0.0001
        density_shift=args.density_shift,     # shift density in softplus: -10
        distance_scale=args.distance_scale,   # scaling sampling distance for computation: 25.0

        t_pe=args.t_pe,       # PE for time: -1
        pos_pe=args.pos_pe,   # PE for position: 6
        density_view_pe=args.density_view_pe, # PE for view direction (density): -1
        app_view_pe=args.app_view_pe,  # PE for view direction (appearance): 2
        fea_pe=args.fea_pe,     # PE for features: 2
        featureC=args.featureC, # hidden feature channel in MLP: 128
        density_dim=args.density_dim,  # the dimension of density channel
        
        step_ratio=args.step_ratio,     # 0.5
        fea2denseAct=args.fea2denseAct, # softplus
        init_scale=args.init_scale,     # default: 0.1, the scale of gaussian noise
        init_shift=args.init_shift,     # default: 0.0, the mean of gaussian noise
        use_envmap=args.use_envmap,     # True
        envmap_res_H=int(args.envmap_res_H / args.downsample_train,),  # 1920
        coarse_sigma_grid_update_rule=coarse_sigma_grid_update_rule,   # conv
        coarse_sigma_grid_reso=None, 
        interval_th=args.interval_th  # force minimum r-grid interval to be r0: True
    )
    if args.ckpt is not None or len(ckpts) > 0:
        # Load the learned model
        if args.ckpt is None:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'\nload ckpt from {ckpt_path}\n')
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        model.load(ckpt, strict=False)

    # 8. Initialize the learn rate settings
    if args.iter_pretrain_envmap > 0:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap_pretrain)
    else:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
    
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    lr_factor = lr_factor ** start
    print("lr factor:", lr_factor)
    print("lr decay: (1) target ratio:", args.lr_decay_target_ratio, "(2) iterations:", args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logarithmic space (300*3)
    # The number of voxels (namely the resolution of voxel, commonly 300)
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
                                                         np.log(args.N_voxel_final), 
                                                         len(upsamp_list) + 1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    # Load all rays, rgbs, and times (sometimes depths)
    allrays = train_dataset.all_rays    # [frames*h*w, 6]
    allrgbs = train_dataset.all_rgbs    # [frames*h*w, 3], range: [0.0, 1.0]
    alltimes = train_dataset.all_times  # [frames*h*w, 1], range: [-1.0, 1.0]
    if args.use_depth:
        alldepths = train_dataset.all_depths

    # Set the rays sampler, SimpleSampler (originall) or ThetaImportanceSampler
    if args.sampling_method == 'simple':
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    elif args.sampling_method == 'theta_importance':
        img_len = allrays.shape[0] // np.prod(train_dataset.img_wh)
        trainingSampler = ThetaImportanceSampler(args.theta_importance_lambda, 
                                                 img_len, 
                                                 train_dataset.img_wh,
                                                 args.batch_size, 
                                                 train_dataset.roi)
    else:
        raise ValueError('sampling method not supported')

    # Set the weight of each loss
    L1_reg_weight = args.L1_weight_initial     # 0.0
    TV_weight_density = args.TV_weight_density # 0.1
    TV_weight_app = args.TV_weight_app         # 0.01
    print(f"1 - initial TV weight for density: {TV_weight_density}")
    print(f"2 - Initial TV weight for appearance: {TV_weight_app}")

    palette_lambda = args.palette_lambda
    hsv_reg_lambda = args.hue_lambda
    weight_lambda = args.weight_lambda
    omega_sparsity_lambda = args.omega_sparsity_lambda
    offset_norm_lambda = args.offset_norm_lambda
    view_dep_norm_lambda = args.view_dep_norm_lambda
    print(f"3 - initial weight for palette supervision: {palette_lambda}")
    print(f"4 - initial weight for palette regularization in HSV space: {hsv_reg_lambda}")
    print(f"5 - initial weight for palette blending weights supervision: {weight_lambda}")
    print(f"6 - Initial weight for omega sparsity: {omega_sparsity_lambda}")
    print(f"7 - Initial weight for offset regularization: {offset_norm_lambda}")
    print(f"8 - Initial weight for view-dependent color regularization: {view_dep_norm_lambda}")

    # Define the TV loss
    tvreg = TVLoss()

    if args.use_envmap and args.iter_pretrain_envmap > 0:
        pbar_pretrain = tqdm(range(args.iter_pretrain_envmap), miniters=50, file=sys.stdout)
        print("\n\n pretrain envmap")

        for pretrain_iter in pbar_pretrain:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
            env_map = renderer(
                rays_train, model, chunk=16384 * 4, device=device, is_train=True, pretrain_envmap=True
            )
            loss_pretrain_envmap = torch.mean((env_map - rgb_train) ** 2)
            optimizer.zero_grad()
            loss_pretrain_envmap.backward()
            optimizer.step()
            if pretrain_iter % 50 == 49:
                pbar_pretrain.set_description(f'Iteration {pretrain_iter:04d}: {loss_pretrain_envmap.item()}')

        evaluation(test_dataset, model, args, renderer, 
                   f'{logfolder}/palette_imgs_vis/', N_vis=args.N_vis,
                   n_coarse=0, compute_extra_metrics=False,
                   exp_sampling=args.exp_sampling, 
                   empty_gpu_cache=True, envmap_only=True, 
                   use_palette=args.use_palette)
        # reset lr rate of envmap
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    pbar = tqdm(range(start, args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()        # [4096] indices
        rays_train = allrays[ray_idx].to(device)   # Tensor: [4096, 6]
        rgb_train = allrgbs[ray_idx].to(device)    # Tensor:[4096, 3]
        frame_time = alltimes[ray_idx].to(device)  # Tensor: [4096, 1]
        
        if args.use_depth:  # False
            depth_train = alldepths[ray_idx].to(device).squeeze()
            depth_nonezero_mask = depth_train != 0
        
        if hasattr(model, "hist_weights"):
            gt_weights = get_palette_weight_with_hist(rgb_train, model.hist_weights.to(device)).detach()
        else:
            gt_weights = None

        output_dict = renderer(
            rays_train, 
            frame_time, 
            model, 
            chunk=args.batch_size, 
            n_coarse=n_coarse, 
            n_fine=n_fine, 
            device=device, 
            is_train=True,
            exp_sampling=args.exp_sampling, 
            pivotal_sample_th=args.pivotal_sample_th,
            resampling=(args.resampling and iteration > args.iter_ignore_resampling), 
            use_coarse_sample=use_coarse_sample,
            use_palette=args.use_palette,
            interval_th=args.interval_th
        )  # rgb_map: [4096, 3], depth_map: [4096], alpha: [4096, 257]

        # Reconstruction/Photometric Loss (MSE)
        if args.use_palette:
            final_rgb = output_dict['final_rgbs'] # final_rgb = view_dep_color + radiance * omega(basis_color + offsets)
            direct_rgb = output_dict['direct_rgbs']  # direct_rgb = view_dep_color + diffuse_rgb
            recon_loss_f = torch.mean((final_rgb - rgb_train) ** 2)
            recon_loss_d = torch.mean((direct_rgb - rgb_train) ** 2)
            recon_loss = recon_loss_f + recon_loss_d
        else:
            rgb_map = output_dict['rgb_maps']  # view-dependent color
            recon_loss = torch.mean((rgb_map - rgb_train) ** 2)

        # Initialize the total loss
        total_loss = recon_loss

        # Blending Weights Supervision
        if gt_weights is not None:
            # weight_lambda = args.weight_lambda * max(0, (1 - iteration / args.weight_decay_iters))
            if weight_lambda > 0:
                pred_weights = output_dict['basis_acc']
                weight_guide_loss = torch.mean((gt_weights - pred_weights) ** 2)
                total_loss = total_loss + weight_lambda * weight_guide_loss
                summary_writer.add_scalar('palette_train/reg_palette_weights', weight_guide_loss.detach().item(), global_step=iteration)
        
        # Regularization
        if args.use_palette:
            if palette_lambda > 0:
                update_basis_color = model.basis_color
                orginal_basis_color = model.basis_color_origin
                palette_loss = torch.mean(((update_basis_color - orginal_basis_color) ** 2).sum(dim=-1))
                total_loss = total_loss + palette_lambda * palette_loss
                summary_writer.add_scalar('palette_train/palette_supervision', palette_loss.detach().item(), global_step=iteration)
            
            if hsv_reg_lambda > 0:
                basis_color_hsv = rgb_to_hsv(update_basis_color.clamp(0, 1))  # [num_basis, 3]
                h = basis_color_hsv[:, 0]  # [num_basis]
                # Calculate the L1/absolute distance between each pair of elements
                h_i = h.unsqueeze(1)  # Reshape h to [num_basis, 1] to enable broadcasting
                h_j = h.unsqueeze(0)  # Reshape h to [1, num_basis] to enable broadcasting
                diff_h = torch.abs(h_i - h_j)
                diff_h = torch.min(diff_h, 360 - diff_h)  # Circular hue differences, range [0, 180]
                # Exclude self-comparisons by setting diagonal to NaN
                diff_h[np.arange(num_basis), np.arange(num_basis)] = float('nan')
                # normalized_diff_h = diff_h / 360.0  # Normalizing to [0, 1] since max diff is 360

                # Find the minimum value for each row, ignoring NaNs
                min_h = nanmin(diff_h, dim=1)
                sigmoid_h =  1 - (1 / (1 + torch.exp(-((min_h - 15) / 5))))
                # Compute the sum of all function values to be the final loss
                palette_h_loss = torch.sum(sigmoid_h)
                
                # normalized_diff_h = diff_h / 360.0  # Normalizing to [0, 1] since max diff is 360
                # mapped_diff = 2 * torch.pi * normalized_diff_h  # Mapping differences to [0, 2π]
                # palette_h_reg = torch.cos(mapped_diff) + 1 # Apply cosine + 1
                # # Calculate the mean while ignoring NaN values (self-comparisons)
                # palette_h_loss = torch.nanmean(palette_h_reg)
                total_loss = total_loss + hsv_reg_lambda * palette_h_loss.to(device)
                summary_writer.add_scalar('palette_train/reg_palette_h', palette_h_loss.detach().item(), global_step=iteration)
                
            if omega_sparsity_lambda > 0:
                pred_omega_sparsity = output_dict['omega_sparsitys']
                omega_sparsity_loss = torch.mean(pred_omega_sparsity)
                total_loss = total_loss + omega_sparsity_lambda * omega_sparsity_loss
                summary_writer.add_scalar('palette_train/reg_omega_sparsity', omega_sparsity_loss.detach().item(), global_step=iteration)

            if offset_norm_lambda > 0:
                pred_offset_norm = output_dict['offset_norms']
                offset_norm_loss = torch.mean(pred_offset_norm)
                total_loss = total_loss + offset_norm_lambda * offset_norm_loss
                summary_writer.add_scalar('palette_train/offset_norm_sparsity', offset_norm_loss.detach().item(), global_step=iteration)
            
            if view_dep_norm_lambda > 0:
                pred_view_dep_norm = output_dict['view_dep_norms']
                view_dep_norm_loss = torch.mean(pred_view_dep_norm)
                total_loss = total_loss + view_dep_norm_lambda * view_dep_norm_loss
                summary_writer.add_scalar('palette_train/reg_view_dep', view_dep_norm_loss.detach().item(), global_step=iteration)

        # depth loss calculation
        if args.use_depth:  # False
            depth_lambda = args.depth_lambda * args.depth_rate ** (int(iteration / args.depth_step_size))
            depth_map = output_dict['depth_maps']
            depth_loss = torch.mean((depth_map[depth_nonezero_mask] - depth_train[depth_nonezero_mask]) ** 2)
            if args.depth_end_iter is not None:
                if iteration > args.depth_end_iter:
                    depth_loss = 0
            total_loss = total_loss + depth_lambda * depth_loss
        
        if L1_reg_weight > 0:    # 0.0
            loss_reg_L1 = model.density_L1()
            total_loss = total_loss + L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar('palette_train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        # TV loss calculation, iter_ignore_TV = 100000
        if TV_weight_density > 0 and iteration < args.iter_ignore_TV:
            TV_weight_density *= lr_factor
            loss_tv = model.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('palette_train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0 and iteration < args.iter_ignore_TV:
            TV_weight_app *= lr_factor
            loss_tv = model.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('palette_train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        recon_loss = recon_loss.detach().item()

        PSNRs.append(-10.0 * np.log(recon_loss) / np.log(10.0))
        summary_writer.add_scalar('palette_train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('palette_train/recon_loss', recon_loss, global_step=iteration)

        # Empty cache in each 1000 iters
        if iteration % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of loss and metric
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {(iteration):06d}:'
                + f' palette_train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' palette_test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' palette_recon_loss = {recon_loss:.6f}'
            )
            PSNRs = []

        # N_vis is the number of rendered images (-1 means equal to given images)
        if (iteration + 1) in vis_list and args.N_vis != 0:
            PSNRs_test = evaluation(
                test_dataset, 
                model, 
                args, 
                renderer, 
                f'{logfolder}/palette_imgs_vis/', 
                N_vis=args.N_vis,
                prtx=f'{(iteration + 1):06d}_', 
                n_coarse=n_coarse, 
                n_fine=n_fine,
                compute_extra_metrics=False, 
                exp_sampling=args.exp_sampling, 
                empty_gpu_cache=True,
                resampling=(args.resampling and iteration > args.iter_ignore_resampling), 
                use_coarse_sample=use_coarse_sample,
                use_palette=args.use_palette,
                interval_th=args.interval_th
            )
            summary_writer.add_scalar('palette_test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration % args.i_weights == 0 and iteration != 0:
            model.save(f'{logfolder}/{args.expname}_palette_{iteration:06d}.th', global_step=iteration)

        if args.resampling and (iteration + 1) % coarse_sigma_grid_update_step == 0 and is_yinyang:
            model.update_coarse_sigma_grid()

        if update_AlphaMask_list and iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] <= 128 ** 3:  # update volume resolution
                reso_mask = reso_cur
            new_aabb = model.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = coordinates.N_to_reso(n_voxels, model.aabb)
            model.upsample_volume_grid(reso_cur)
            coordinates.set_resolution(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = model.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale, args.lr_envmap * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # Save the last learned model
    model.save(f'{logfolder}/{args.expname}_palette.th', global_step=iteration)

def train(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]  # dataset_dict = {'llff','omnivideos', 'cpp_omnivideos', omniblender','omniscenes'}
    load_params = {
        "data_dir": args.datadir,
        "split": "train",
        "is_stack": False,
        "use_gt_depth": args.use_gt_depth,
        "downsample": args.downsample_train,
        "near_far": args.near_far,
        "localization_method": args.localization_method,
    }

    # Load train dataset (not stack)
    train_dataset = dataset(**load_params)

    # Load test dataset (stack)
    load_params["split"] = "test"
    load_params["is_stack"] = True
    load_params["downsample"] = args.downsample_test
    load_params["skip"] = args.test_skip  # only test dataset can have non one skip value
    test_dataset = dataset(**load_params)
    near_far = train_dataset.near_far  # [0.1, 300.0]

    # init resolution
    upsamp_list = args.upsamp_list  # [1000000000000000]
    update_AlphaMask_list = args.update_AlphaMask_list  # [10000000000000]
    vis_list = args.vis_list   # [10000, 50000, 100000]
    
    n_lamb_sigma = args.n_lamb_sigma # R_sigma [16, 16, 16]
    n_lamb_sh = args.n_lamb_sh       # R_c [48, 48, 48]

    # init log file (Save outputs)
    if args.add_timestamp: # the timestamp here means when run this program
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}' 
    os.makedirs(logfolder, exist_ok=True)
    export_config(args, logfolder)
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)

    # init parameters
    # 1. Initialize bounding box
    aabb = train_dataset.scene_bbox.to(device)  # Tensor: of shape [2, 3]
    # aabb: tensor([[-301.9334, -301.9334, -301.9334],
    #     [ 301.9334,  301.9334,  301.9334]], device='cuda:0')

    # 2. Initialize specific coordinates
    coordinates = coordinates_dict[args.coordinates_name]
    is_yinyang = args.coordinates_name == 'yinyang'  # True or False
    is_general_sphere = args.coordinates_name == 'generic_sphere'  # True or False
    # for backward compatibility
    if is_yinyang or is_general_sphere:
        coordinates = coordinates(device, 
                                  aabb, 
                                  exp_r=args.exp_sampling, 
                                  N_voxel=args.N_voxel_init, 
                                  r0=args.r0, 
                                  interval_th=args.interval_th)
    else:
        coordinates = coordinates(device, aabb)
    
    # 3. Initialize resolution
    reso_cur = coordinates.N_to_reso(args.N_voxel_init, aabb)  # list(len=3): grid size (three dims)
    if not is_yinyang:
        coordinates.set_resolution(reso_cur)
    
    # 4. Initialize the number of samplings
    n_coarse = args.n_coarse                        # 128
    n_fine = args.n_fine if args.resampling else 0  # 128
    use_coarse_sample = args.use_coarse_sample      # True
    if args.resampling:
        coarse_sigma_grid_update_rule = args.coarse_sigma_grid_update_rule
        if coarse_sigma_grid_update_rule == "conv":
            coarse_sigma_grid_update_step = 1  # This one
        elif coarse_sigma_grid_update_rule == "samp":
            raise NotImplementedError
        else:
            coarse_sigma_grid_update_step = 1000000000
    else:
        coarse_sigma_grid_update_rule = None
        coarse_sigma_grid_update_step = 1000000000

    # 5. Initialize the volume render
    renderer = volume_renderer

    # 6. Initialize the neural model
    ckpts = [os.path.join(args.basedir, args.expname, f) for f in sorted(os.listdir(os.path.join(args.basedir, args.expname))) if f.endswith('.th')]
    if args.ckpt is not None or len(ckpts) > 0:
        # Load the learned model
        if args.ckpt is None:
            ckpt_path = ckpts[-1]
        else:
            ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path, map_location=device)
        print(f'\nload ckpt from {ckpt_path}\n')
        kwargs = ckpt['kwargs']
        kwargs.update({'device': device})
        model = eval(args.model_name)(**kwargs)
        start = model.load(ckpt)
    else:
        # Create the model
        print("\nCreate a new model...")
        start = 0
        model = eval(args.model_name)(
            aabb,     # bounding box
            reso_cur, # resolution
            device,   # cuda or cpu
            coordinates, # 'xyz', 'sphere', 'cylinder', 'balanced_sphere', ... , 'yinyang', 'generic_sphere'
            
            time_grid=args.time_grid,     # time_grid: 300
            density_n_comp=n_lamb_sigma,  # R_sigma: [16, 16, 16]
            appearance_n_comp=n_lamb_sh,  # R_c: [48, 48, 48]
            app_dim=args.data_dim_color,  # dimension of color/appearance feature: 27
            near_far=near_far,            # [0.1, 300.0]
            densityMode=args.densityMode, # density_MLP
            shadingMode=args.shadingMode, # render_MLP
            alphaMask_thres=args.alpha_mask_thre, # threshold for creating alpha mask volume: 0.0001
            density_shift=args.density_shift,     # shift density in softplus: -10
            distance_scale=args.distance_scale,   # scaling sampling distance for computation: 25.0

            t_pe=args.t_pe,       # PE for time: -1
            pos_pe=args.pos_pe,   # PE for position: 6
            density_view_pe=args.density_view_pe, # PE for view direction (density): -1
            app_view_pe=args.app_view_pe,  # PE for view direction (appearance): 2
            fea_pe=args.fea_pe,     # PE for features: 2
            featureC=args.featureC, # hidden feature channel in MLP: 128
            density_dim=args.density_dim,  # the dimension of density channel
            
            step_ratio=args.step_ratio,     # 0.5
            fea2denseAct=args.fea2denseAct, # softplus
            init_scale=args.init_scale,     # default: 0.1, the scale of gaussian noise
            init_shift=args.init_shift,     # default: 0.0, the mean of gaussian noise
            use_envmap=args.use_envmap,     # True
            envmap_res_H=int(args.envmap_res_H / args.downsample_train,),  # 1920
            coarse_sigma_grid_update_rule=coarse_sigma_grid_update_rule,   # conv
            coarse_sigma_grid_reso=None, 
            interval_th=args.interval_th  # force minimum r-grid interval to be r0: True
        )
    
    if args.iter_pretrain_envmap > 0:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap_pretrain)
    else:
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
    
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio ** (1 / args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio ** (1 / args.n_iters)
    lr_factor = lr_factor ** start
    print("lr factor:", lr_factor)
    print("lr decay: (1) target ratio:", args.lr_decay_target_ratio, "(2) iterations:", args.lr_decay_iters)

    optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # linear in logarithmic space (300*3)
    # The number of voxels (namely the resolution of voxel, commonly 300)
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), 
                                                         np.log(args.N_voxel_final), 
                                                         len(upsamp_list) + 1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs, PSNRs_test = [], [0]

    # Load all rays, rgbs, and times (sometimes depths)
    allrays = train_dataset.all_rays    # [frames*h*w, 6]
    allrgbs = train_dataset.all_rgbs    # [frames*h*w, 3], range: [0.0, 1.0]
    alltimes = train_dataset.all_times  # [frames*h*w, 1], range: [-1.0, 1.0]
    if args.use_depth:
        alldepths = train_dataset.all_depths

    # Set the rays sampler, SimpleSampler (originall) or ThetaImportanceSampler
    if args.sampling_method == 'simple':
        trainingSampler = SimpleSampler(allrays.shape[0], args.batch_size)
    elif args.sampling_method == 'theta_importance':
        img_len = allrays.shape[0] // np.prod(train_dataset.img_wh)
        trainingSampler = ThetaImportanceSampler(args.theta_importance_lambda, 
                                                 img_len, 
                                                 train_dataset.img_wh,
                                                 args.batch_size, 
                                                 train_dataset.roi)
    else:
        raise ValueError('sampling method not supported')

    # Set the weight of each loss
    Ortho_reg_weight = args.Ortho_weight   # 0.0
    L1_reg_weight = args.L1_weight_initial # 0.0
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app # 0.1, 0.01
    entropy_weight = args.entropy_weight # 0.0
    
    # Define the TV loss
    tvreg = TVLoss()

    if args.use_envmap and args.iter_pretrain_envmap > 0:
        pbar_pretrain = tqdm(range(args.iter_pretrain_envmap), miniters=50, file=sys.stdout)
        print("\n\n pretrain envmap")

        for pretrain_iter in pbar_pretrain:
            ray_idx = trainingSampler.nextids()
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
            env_map = renderer(
                rays_train, model, chunk=16384 * 4, device=device, is_train=True, pretrain_envmap=True
            )
            loss_pretrain_envmap = torch.mean((env_map - rgb_train) ** 2)
            optimizer.zero_grad()
            loss_pretrain_envmap.backward()
            optimizer.step()
            if pretrain_iter % 50 == 49:
                pbar_pretrain.set_description(f'Iteration {pretrain_iter:04d}: {loss_pretrain_envmap.item()}')

        evaluation(
            test_dataset, model, args, 
            renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
            n_coarse=0, compute_extra_metrics=False,
            exp_sampling=args.exp_sampling, empty_gpu_cache=True, envmap_only=True
        )
        # reset lr rate of envmap
        grad_vars = model.get_optparam_groups(args.lr_init, args.lr_basis, args.lr_envmap)
        optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    pbar = tqdm(range(start, args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()        # [4096] indices
        rays_train = allrays[ray_idx].to(device)   # Tensor: [4096, 6]
        rgb_train = allrgbs[ray_idx].to(device)    # Tensor:[4096, 3]
        frame_time = alltimes[ray_idx].to(device)  # Tensor: [4096, 1]
        
        if args.use_depth:  # False
            depth_train = alldepths[ray_idx].to(device).squeeze()
            depth_nonezero_mask = depth_train != 0

        output_dict = renderer(
            rays=rays_train, 
            times=frame_time, 
            model=model, 
            chunk=args.batch_size, 
            n_coarse=n_coarse, 
            n_fine=n_fine, 
            is_train=True,
            exp_sampling=args.exp_sampling,
            device=device, 
            empty_gpu_cache=False,
            pretrain_envmap=False,
            pivotal_sample_th=args.pivotal_sample_th,
            resampling=(args.resampling and iteration > args.iter_ignore_resampling), 
            use_coarse_sample=use_coarse_sample,
            use_palette=False,
            interval_th=args.interval_th
        )  # rgb_map: [4096, 3], depth_map: [4096], alpha: [4096, 257]

        rgb_map = output_dict['rgb_maps']
        depth_map = output_dict['depth_maps']
        alpha = output_dict['alphas']

        # Reconstruction/Photometric Loss (MSE) - equation (13)
        recon_loss = torch.mean((rgb_map - rgb_train) ** 2)

        # Initialize the total loss
        total_loss = recon_loss

        # sparsity loss (ref: DirectVoxGO)
        if args.sparsity_lambda > 0:  # 0.0
            sample_points = torch.rand((args.N_sparsity_points, 3), device=device) * 2 - 1
            sp_sigma = model.compute_densityfeature(sample_points)
            sp_sigma = model.feature2density(sp_sigma)
            loss_sp = 1.0 - torch.exp(-args.sparsity_length * sp_sigma).mean()
            total_loss = total_loss + args.sparsity_lambda * loss_sp

        # depth loss calculation
        depth_lambda = args.depth_lambda * args.depth_rate ** (int(iteration / args.depth_step_size))
        if args.use_depth:  # False
            depth_loss = torch.mean((depth_map[depth_nonezero_mask] - depth_train[depth_nonezero_mask]) ** 2)
            if args.depth_end_iter is not None:
                if iteration > args.depth_end_iter:
                    depth_loss = 0
            total_loss = total_loss + depth_lambda * depth_loss

        if Ortho_reg_weight > 0:  # 0.0
            loss_reg = model.vector_comp_diffs()
            total_loss = total_loss + Ortho_reg_weight * loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)

        if L1_reg_weight > 0:    # 0.0
            loss_reg_L1 = model.density_L1()
            total_loss = total_loss + L1_reg_weight * loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)

        # TV loss calculation, iter_ignore_TV = 100000
        if TV_weight_density > 0 and iteration < args.iter_ignore_TV:
            TV_weight_density *= lr_factor
            loss_tv = model.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app > 0 and iteration < args.iter_ignore_TV:
            TV_weight_app *= lr_factor
            loss_tv = model.TV_loss_app(tvreg) * TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)

        if entropy_weight > 0 and iteration > args.iter_ignore_entropy: # 0
            entropy_weight *= lr_factor
            loss_entropy = ray_entropy_loss(alpha)
            total_loss = total_loss + loss_entropy * entropy_weight
            summary_writer.add_scalar('train/ray_entropy_loss', loss_entropy.detach().item(), global_step=iteration)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        recon_loss = recon_loss.detach().item()

        PSNRs.append(-10.0 * np.log(recon_loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/recon_loss', recon_loss, global_step=iteration)

        # Empty cache in each 1000 iters
        if iteration % 1000 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of loss and metric
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {(iteration):06d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' recon_loss = {recon_loss:.6f}'
            )
            PSNRs = []

        # N_vis is the number of rendered images (-1 means equal to given images)
        # if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
        if (iteration + 1) in vis_list and args.N_vis != 0:
            PSNRs_test = evaluation(
                test_dataset, 
                model, 
                args, 
                renderer, 
                savePath=f'{logfolder}/imgs_vis/', 
                N_vis=args.N_vis,
                prtx=f'{(iteration + 1):06d}_', 
                n_coarse=n_coarse, 
                n_fine=n_fine,
                compute_extra_metrics=False, 
                exp_sampling=args.exp_sampling, 
                empty_gpu_cache=True,
                resampling=(args.resampling and iteration > args.iter_ignore_resampling), 
                use_coarse_sample=use_coarse_sample,
                use_palette=False,
                interval_th=args.interval_th,
                test=True
            )
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration % args.i_weights == 0 and iteration != 0:
            model.save(f'{logfolder}/{args.expname}_{iteration:06d}.th', global_step=iteration)

        if args.resampling and (iteration + 1) % coarse_sigma_grid_update_step == 0 and is_yinyang:
            model.update_coarse_sigma_grid()

        if update_AlphaMask_list and iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2] <= 128 ** 3:  # update volume resolution
                reso_mask = reso_cur
            new_aabb = model.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = coordinates.N_to_reso(n_voxels, model.aabb)
            model.upsample_volume_grid(reso_cur)
            coordinates.set_resolution(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = model.get_optparam_groups(args.lr_init * lr_scale, args.lr_basis * lr_scale, args.lr_envmap * lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

    # Save the last learned model
    model.save(f'{logfolder}/{args.expname}.th', global_step=iteration)

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_train = evaluation(
            train_dataset, model, args, renderer, 
            f'{logfolder}/imgs_train_all/', N_vis=-1, N_samples=n_coarse,
            device=device, exp_sampling=args.exp_sampling, test=True
        )
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_train)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(
            test_dataset, model, args, renderer, 
            f'{logfolder}/imgs_test_all/', N_vis=-1, n_coarse=n_coarse,
            n_fine=n_fine, device=device, exp_sampling=args.exp_sampling,
            empty_gpu_cache=True, resampling=args.resampling, 
            use_coarse_sample=use_coarse_sample, test=True
        )
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')


if __name__ == '__main__':

    # Set default data type
    torch.set_default_dtype(torch.float32)

    # Fix Random Seed for Reproducibility
    random.seed(20231215)
    np.random.seed(20231215)
    torch.manual_seed(20231215)
    torch.cuda.manual_seed(20231215)

    # Set configer
    parser = recursive_config_parser()
    args = parser.parse_args()
    # print_arguments(args)

    if args.evaluation:
        # run render for test/evaluation
        render_test(args)
    elif args.stabilize:
        # run render for stabilization
        stabilizer(args)
    elif args.palette_extract:
        # run render for palette extracting
        palette_extractor(args)
    elif args.palette_train and args.use_palette:
        # run train for palette training
        palette_train(args)
    elif args.palette_edit and args.use_palette:
        # run render for palette-based editing
        render_edit(args)
    else:
        # run train
        train(args)
        # render_test(args)
