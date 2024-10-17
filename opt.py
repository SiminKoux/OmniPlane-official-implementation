import configargparse
import os
from pathlib import Path

'''
    A parser for configuration files
'''

def load_all_include(config_file):
    parser = config_parser()
    args = parser.parse_args("--config {}".format(config_file))
    path = Path(config_file)

    include = []
    if args.include:
        include.append(os.path.join(path.parent, args.include))
        return include + load_all_include(os.path.join(path.parent, args.include))
    else:
        return include


def recursive_config_parser():
    parser = config_parser()
    args = parser.parse_args()
    include_files = load_all_include(args.config)
    include_files = list(reversed(include_files))
    parser = config_parser(default_files=include_files)
    return parser


def config_parser(default_files=None):
    if default_files is not None:
        parser = configargparse.ArgumentParser(default_config_files=default_files)
    else:
        parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument('--include', type=str, default=None, help='parent config file path')

    parser.add_argument("--expname", type=str, help='experiment name')
    parser.add_argument('--model_name', type=str, default='OmniPlanes',
                        choices=['TensorVMSplit', 'TensorCP', 'OmniPlanes', 'YinYang_OmniPlanes']) # 'dyPanoTensor'
    parser.add_argument('--dataset_name', type=str, default='omnivideos',
                        choices=['omnivideos', 'cpp_omnivideos', 'stabilize_omnivideos', 
                                 'omniblender', 'omniscenes', 'openvslam', 'own_data'])
    
    parser.add_argument("--basedir", type=str, default='./log', help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/360/walking', help='input data directory')
    
    parser.add_argument("--add_timestamp", type=int, default=0, help='add timestamp to dir')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument('--test_skip', type=int, default=1, help='skip test set for fast visualization')

    
    # Running Options
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--evaluation", type=int, default=0,   # set '1' means only evaluation
                        help='render and evaluate test set')
    parser.add_argument("--stabilize", type=int, default=0,   # set '1' means only stabilize
                        help='render and evaluate test set')
    parser.add_argument("--metric_only", type=int, default=0,  # set '1' means only metric
                        help='evaluate metrics in test set from existing rendered images')
    
    
    # Depth Supervision (--with depth or not)
    parser.add_argument("--use_depth", action='store_true',
                        help="use depth supervision.")
    parser.add_argument("--depth_lambda", type=float, default=0.1,
                        help='depth lambda for loss')
    parser.add_argument("--depth_step_size", type=int, default=5000,
                        help='reducing depth every')
    parser.add_argument("--depth_rate", type=float, default=1,
                        help='reducing depth rate')
    parser.add_argument("--depth_end_iter", type=int,
                        help='when smoothing will be end')    
    parser.add_argument("--use_gt_depth", action='store_true',
                        help='use ground truth depth value')
    
    
    # Coordinates Settings
    parser.add_argument('--coordinates_name', type=str, default='yinyang',
                        choices=['xyz', 'sphere', 'cylinder', 
                                 'balanced_sphere', 'directional_sphere', 
                                 'directional_balanced_sphere', 'euler_sphere', 
                                 'yinyang', 'generic_sphere'])
    parser.add_argument('--r0', type=float, default=None, help='radius of initial spherical shell')
    parser.add_argument('--interval_th', action='store_true', help='force minimum r-grid interval to be r0')  

    
    # Downsampling (being same as NeRF)
    # used in real-world dataset (might be 1.0, 4.0 or 8.0), default 1.0 in synthesize dataset
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)

    
    # Loader Options (should be the number of sampling points)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)

    parser.add_argument('--localization_method', type=str, default='colmap',
                        choices=['colmap', 'openvslam', 'pix4d'])
    parser.add_argument('--near_far', type=float, action='append')
    parser.add_argument('--roi', type=float, action='append')

    
    # Training Options
    # Learning Rate
    parser.add_argument("--lr_init", type=float, default=0.005, help='learning rate')
    parser.add_argument("--lr_basis", type=float, default=1e-3, help='learning rate')

    parser.add_argument("--lr_envmap_pretrain", type=float, default=0.02, help='learning rate for envmap')
    parser.add_argument("--lr_envmap", type=float, default=0.005, help='learning rate for envmap')

    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help='number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')
    
    # Loss Settings
    parser.add_argument("--weight_lambda", type=float, default=0.05, help='weight of weight loss')
    parser.add_argument("--weight_decay_iters", type=int, default=5000, help='iteration number when weight_lambda drops to 0')
    parser.add_argument("--palette_lambda", type=float, default=0.001, help='weight of palette loss')
    parser.add_argument("--max_freeze_palette_iters", type=int, default=5000, help='iteration number when palette color is released')
    parser.add_argument("--omega_sparsity_lambda", type=float, default=0.0002, help='weight of omega sparsity loss')
    parser.add_argument("--offset_norm_lambda", type=float, default=0.03, help='weight of offset normalization loss')
    parser.add_argument("--view_dep_norm_lambda", type=float, default=0.1, help='weight of view dependent color regularity loss')
    parser.add_argument("--hue_lambda", type=float, default=0.0002, help='weight of hue diversity regularity loss')
    # TV Loss
    parser.add_argument("--TV_weight_density", type=float, default=0.0, help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0, help='loss weight')
    # parser.add_argument("--TV_t_s_ratio", type=float, default=2.0, 
    #                     help='ratio of TV loss along temporal and spatial dimensions')
    parser.add_argument("--iter_ignore_TV", type=int, default=1e5, 
                        help='ignore TV loss after the first few iterations')

    # L1 Loss
    parser.add_argument("--L1_weight_initial", type=float, default=0.0, help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0, help='loss weight')

    # Ortho Regularization
    parser.add_argument("--Ortho_weight", type=float, default=0.0, help='loss weight')

    # Ray Entropy Loss
    parser.add_argument("--entropy_weight", type=float, default=0.0, help='weight of ray entropy loss')
    parser.add_argument("--iter_ignore_entropy", type=int, default=0, 
                        help='ignore ray entropy loss for the first few iterations')

    # Sparsity Loss
    parser.add_argument("--sparsity_lambda", type=float, default=0.1,
                        help='sparsity lambda for loss')
    parser.add_argument("--N_sparsity_points", type=int, default=10000,
                        help='N points to sample for sparsity loss calculation')
    parser.add_argument("--sparsity_length", type=float, default=0.2,
                        help='hyper param for sparse alpha composition')
    
    
    # Model
    # Volume Options
    parser.add_argument("--n_lamb_sigma", type=int, action="append") # [16, 16, 16] - density
    parser.add_argument("--n_lamb_sh", type=int, action="append")    # [48, 48, 48] - color
    parser.add_argument("--data_dim_color", type=int, default=27)    # '27' is the dimension of appearance feature
    
    # Density Feature Settings
    parser.add_argument("--fea2denseAct", type=str, default='softplus')  # nv3d: relu
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
    parser.add_argument("--distance_scale", type=float, default=25.0,
                        help='scaling sampling distance for computation')
    
    # Alpha/Empty Mask Settings
    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')

    # Network Decoder
    parser.add_argument("--densityMode", type=str, default="density_MLP",
                        help='which shading mode to use, decode color')
    parser.add_argument("--shadingMode", type=str, default="render_MLP",
                        help='which shading mode to use, decode color')
    parser.add_argument("--t_pe", type=int, default=-1,
                        help='number of pe for time dimension')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--density_view_pe", type=int, default=-1,
                        help='number of pe for view')
    parser.add_argument("--app_view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    parser.add_argument("--density_dim", type=int, default=1,
                        help='the dimension of density channel')
    parser.add_argument("--init_scale", type=float, default=0.1,
                        help='the scale of gaussian noise for feature plane initialization')
    parser.add_argument("--init_shift", type=float, default=0.0,
                        help='the mean of gaussian noise for feature plane initialization')
    
    # Rendering Options
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)
    
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument('--ndc_ray', type=int, default=0)
    
    
    # Sampling Settings
    # Basic Settings
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)

    parser.add_argument('--N_voxel_init', type=int, default=100**3)  # initial voxel number
    parser.add_argument('--N_voxel_final', type=int, default=300**3) # final voxel number
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")

    parser.add_argument("--time_grid", type=int, default=300)
    parser.add_argument("--time_scale", type=float, default=1.0)
    
    # parser.add_argument("--fusion_one", type=str, default="multiply")
    # parser.add_argument("--fusion_two", type=str, default="concat")

    parser.add_argument('--idx_view', type=int, default=0)
    
    # Unique Sampling
    parser.add_argument('--sampling_method', type=str, default='simple',
                        choices=['simple', 'theta_importance', 'brute_force'])
    parser.add_argument('--theta_importance_lambda', type=float, default=5,
                        help='weight for theta importance sampling')
    parser.add_argument('--exp_sampling', default=False, action="store_true",
                        help='exponential sampling')
    
    # Resampling
    parser.add_argument('--resampling', default=False, action='store_true')
    parser.add_argument("--iter_ignore_resampling", type=int, default=-1, 
                        help="ignore resampling for the first few iterations")
    
    parser.add_argument('--n_coarse', type=int, default=128, 
                        help='number of coarse samples along camera ray')
    parser.add_argument('--n_fine', type=int, default=64, 
                        help='number of fine samples to resample from coarse ray weights')
    
    # About Coarse Sample
    parser.add_argument('--use_coarse_sample', action='store_true', 
                        help='use both coarse samples and fine samples for rendering')
    parser.add_argument("--coarse_sigma_grid_update_rule", type=str, default=None, choices=["conv", "samp"],
                        help="coarse sigma grid updating strategy. "
                             "conv: obtain coarse sigma grid by convolving with a kernel. "
                             "samp: obtain coarse sigma grid by sampling from fine sigma grid.")
    parser.add_argument('--ray_weight_th', type=float, default=0.01, 
                        help='ray weight threshold value for filtering pivot coarse samples')
    parser.add_argument("--pivotal_sample_th", type=float, default=0., 
                        help="weight threshold value for filtering coarse samples to obtain pivotal samples")
    parser.add_argument("--filter_ray", action='store_true', help='filter rays in bbox')
    
    
    # Envmap Options
    parser.add_argument("--use_envmap", default=False, action="store_true")  # outdoor scene - True
    parser.add_argument("--envmap_res_H", type=int, default=1000)            # the width of video frame
    parser.add_argument("--iter_pretrain_envmap", type=int, default=0)       # iteration for pretraining envmap

    
    # Blender Flags
    parser.add_argument("--white_bkgd", action='store_true',  # defalut: false (disabled)
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    

    # Palette Options
    parser.add_argument('--color_space', type=str, default='srgb', choices=['srgb', 'linear'], 
                        help="Color space, supports (linear, srgb)") 
    parser.add_argument("--use_palette", default=False, action="store_true")
    
    
    # Logging/Saving Options
    parser.add_argument("--N_vis", type=int, default=-1, help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000, help='frequency of visualize the image, deprecated!!')
    parser.add_argument("--vis_list", type=int, action="append", help='list of visualization steps')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of save the weights')

    
    return parser


def export_config(args, logdir):
    """
    Create log dir and copy the config file
    """
    f = os.path.join(logdir, 'args.txt')
    with open(f, 'w') as f:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            f.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(logdir, 'config.txt')
        with open(f, 'w') as f:
            f.write(open(args.config, 'r').read())
