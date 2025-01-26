import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from PIL import Image
import scipy.signal
import plyfile
import random
import skimage.measure
import matplotlib.pyplot as plt
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from palette.backend import _backend
from palette.rgbsg import *

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def compute_RGB_histogram(colors_rgb: np.ndarray,
                          weights: np.ndarray,
                          bits_per_channel: int):
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _backend.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        print(err)
        assert False

    return bin_weights, bin_centers_rgb

def run_kmeans(n_clusters: int,
               points: np.ndarray,
               init: np.ndarray,
               sample_weight: np.ndarray):
    print(f'running kmeans with K = {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, init=init).fit(X=points, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        center_weights[i] = np.sum(sample_weight[labels==i])

    idcs = np.argsort(center_weights * -1)
    return centers[idcs], center_weights[idcs]

def palette_extraction(inputs: dict, 
                       output_dir: str, 
                       H: int,
                       W: int,
                       tau: float = 8e-3,
                       palette_size = None,
                       normalize_input = False,
                       error_thres = 5.0 / 255.0):
    """
        Extract palettes with the RGBXY method
        Convex hull should have at least 4 vertices
    """
    assert palette_size is None or palette_size >=4
    print(f'extracting palette with {palette_size} colors')
    
    output_prefix = "%s/extract"%output_dir
    # if not os.path.exists(output_prefix):
    #     print(f'create output directionary {output_prefix}')
    #     os.makedirs(output_prefix)

    # Radiance Sampling
    colors = inputs['colors']
    weights = np.ones_like(colors[..., 0])
    colors = colors.reshape(-1, 3)
    weights = weights.flatten()

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'
    
    # Save Radiance Samples (outside timig analysis)
    n_total = H * W
    random.seed(0)
    idcs = random.sample(range(len(colors)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element if idcs should be unique'
    img = colors[idcs].reshape(H, W, 3)
    Image.fromarray((img*255).round().clip(0, 255).astype(np.uint8)).save(output_prefix + "-radiance-raw.png")

    # Radiance Sample Filtering
    # Coarse Histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = compute_RGB_histogram(colors, weights, bits_per_channel = 3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights
    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    # Fine Histogram (2^5 = 32 bins)
    bin_weights_fine, bin_centers_fine = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    # Convex Hull Simplification
    palette_rgb = Hull_Simplification_posternerf(
        centers.astype(np.double), output_prefix,
        pixel_counts=center_weights,
        error_thres=error_thres,
        target_size=palette_size)
    _, hist_rgb = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    
    if normalize_input:
        hist_rgb = hist_rgb + 0.05
        hist_rgb_norm = np.linalg.norm(hist_rgb, axis=-1, keepdims=True) #.clip(min=0.1)
        hist_rgb = hist_rgb / hist_rgb_norm

    # Generate weight
    hist_weights = Tan18.Get_ASAP_weights_using_Tan_2016_triangulation_and_then_barycentric_coordinates(hist_rgb.astype(np.double).reshape((-1,1,3)), 
                        palette_rgb, None, order=0) # N_bin_center x 1 x num_palette
    hist_weights = hist_weights.reshape([32,32,32,palette_rgb.shape[0]])
    
    # Save Palette
    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img * 255).round().clip(0, 255).astype(np.uint8)).save(output_prefix+"-palette.png")

    write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)
    np.savez(os.path.join(output_dir, 'hist_weights.npz'), hist_weights=hist_weights)

def get_valid(rays, rgbs, depths, weights):
    rays_o = rays[:, :3]
    rays_d = rays[:, 3:]
    positions = rays_o + rays_d * depths[..., None]

    rgbs_norm = rgbs + 0.05
    rgbs_norm = rgbs_norm / rgbs_norm.norm(dim=-1, p=2, keepdim=True)
    
    valid = (weights > 5e-1) # (1226830,)
    rgbs = rgbs[valid]
    rgbs_norm = rgbs_norm[valid]
    depths = depths[valid]
    positions = positions[valid]

    return rgbs_norm, positions

def extract_euler(rotation_matrix):
    ''' Extract Euler Angles (Roll, Pitch, Yaw) from rotation matrices '''
    sy = torch.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
    singular = sy < 1e-6  # If sy is close to zero, there may be gimbal lock

    if not singular:
        roll = torch.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = torch.atan2(-rotation_matrix[2, 0], sy)
        yaw = torch.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = torch.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = torch.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0
    return torch.stack([roll, pitch, yaw])

def vis_orientations(ori_poses, smoothed_poses, save_path='camera_orientations.png'):
    # Extract camera orientations for original poses
    ori_orientations = []
    for pose in ori_poses:
        rotation_matrix = pose[:3, :3]  # Extract rotation matrix
        rpy = extract_euler(rotation_matrix)
        ori_orientations.append(rpy)
    ori_orientations = torch.stack(ori_orientations)

    # Extract camera orientations for smoothed poses
    smoothed_orientations = []
    for pose in smoothed_poses:
        rotation_matrix = pose[:3, :3]  # Extract rotation matrix
        rpy = extract_euler(rotation_matrix)
        smoothed_orientations.append(rpy)
    smoothed_orientations = torch.stack(smoothed_orientations)

    # Convert to numpy arrays for easier plotting with matplotlib
    ori_orientations = ori_orientations.numpy()
    smoothed_orientations = smoothed_orientations.numpy()

    # Plot each orientation (Roll, Pitch, Yaw) in separate 2D line charts
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot Roll
    axs[0].plot(ori_orientations[:, 0], label='Original Roll', linestyle='--', color='b')
    axs[0].plot(smoothed_orientations[:, 0], label='Smoothed Roll', linestyle='-', color='r')
    axs[0].set_title('Roll Over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Roll (radians)')
    axs[0].legend()

    # Plot Pitch
    axs[1].plot(ori_orientations[:, 1], label='Original Pitch', linestyle='--', color='b')
    axs[1].plot(smoothed_orientations[:, 1], label='Smoothed Pitch', linestyle='-', color='r')
    axs[1].set_title('Pitch Over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Pitch (radians)')
    axs[1].legend()

    # Plot Yaw
    axs[2].plot(ori_orientations[:, 2], label='Original Yaw', linestyle='--', color='b')
    axs[2].plot(smoothed_orientations[:, 2], label='Smoothed Yaw', linestyle='-', color='r')
    axs[2].set_title('Yaw Over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Yaw (radians)')
    axs[2].legend()

    # Adjust layout for clarity
    plt.tight_layout()
    # Save the figure
    plt.savefig(save_path)
    plt.close(fig)

def vis_positions(ori_poses, smoothed_poses, save_path='camera_positions.png'):
    # Extract camera positions for original poses
    ori_positions = []
    for pose in ori_poses:
        position = pose[:3, 3]
        ori_positions.append(position)
    ori_positions = torch.stack(ori_positions)

    # Extract camera positions for smoothed poses
    smoothed_positions = []
    for pose in smoothed_poses:
        position = pose[:3, 3]
        smoothed_positions.append(position)
    smoothed_positions = torch.stack(smoothed_positions)

    # Convert to numpy arrays for easier plotting with matplotlib
    ori_positions = ori_positions.numpy()
    smoothed_positions = smoothed_positions.numpy()

    # Plot each axis (X, Y, Z) in separate 2D line charts
    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot X axis
    axs[0].plot(ori_positions[:, 0], label='Original X', linestyle='--', color='b')
    axs[0].plot(smoothed_positions[:, 0], label='Smoothed X', linestyle='-', color='r')
    axs[0].set_title('X Axis Over Time')
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('X Position')
    axs[0].legend()

    # Plot Y axis
    axs[1].plot(ori_positions[:, 1], label='Original Y', linestyle='--', color='b')
    axs[1].plot(smoothed_positions[:, 1], label='Smoothed Y', linestyle='-', color='r')
    axs[1].set_title('Y Axis Over Time')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Y Position')
    axs[1].legend()

    # Plot Z axis
    axs[2].plot(ori_positions[:, 2], label='Original Z', linestyle='--', color='b')
    axs[2].plot(smoothed_positions[:, 2], label='Smoothed Z', linestyle='-', color='r')
    axs[2].set_title('Z Axis Over Time')
    axs[2].set_xlabel('Time Step')
    axs[2].set_ylabel('Z Position')
    axs[2].legend()

    # Adjust layout for clarity and save the figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def vis_camera(ori_poses, smoothed_poses):
     # Extract camera positions (translations) for original poses
    ori_positions = []
    for pose in ori_poses:
        position = pose[:3, 3]  # Extract translation (camera position)
        ori_positions.append(position)
    ori_positions = torch.stack(ori_positions)

    # Extract camera positions (translations) for smoothed poses
    smoothed_positions = []
    for pose in smoothed_poses:
        position = pose[:3, 3]  # Extract translation (camera position)
        smoothed_positions.append(position)
    smoothed_positions = torch.stack(smoothed_positions)

    # Convert to numpy arrays for easier plotting with matplotlib
    ori_positions = ori_positions.numpy()
    smoothed_positions = smoothed_positions.numpy()

    # Visualize the two sets of camera positions connected with lines
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original camera positions as a connected line
    ax.plot(ori_positions[:, 0], 
            ori_positions[:, 1], 
            ori_positions[:, 2], 
            color='b',
            linestyle='--', 
            label='Original Positions')

    # Plot smoothed camera positions as a connected line
    ax.plot(smoothed_positions[:, 0], 
            smoothed_positions[:, 1], 
            smoothed_positions[:, 2], 
            color='r', 
            linestyle='-',
            label='Smoothed Positions')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Positions Over Time')
    ax.legend()
    plt.show()

def stabilize_camera_poses(c2w_matrices, kernel_size=5, sigma=2.0): 
    ''' 
    Stabilize the camera poses represented by c2w matrices. 
    ''' 
    translations = [] 
    quaternions = [] 

    # Step 1: Extract translations and convert rotation matrices to quaternions 
    for c2w in c2w_matrices:
        rot_matrix = c2w[:3, :3] # tensor - [3, 3]
        translation = c2w[:3, 3] # tensor - [3]
        quat = R.from_matrix(rot_matrix).as_quat() # numpy.ndarray - (4, )
        translations.append(translation)
        quaternions.append(quat) 

    # Step 2: Smooth translations using Gaussian filter 
    smoothed_translations = apply_gaussian_filter_to_trans(translations, kernel_size=kernel_size, sigma=sigma) # [num_poses, 3]

    # Step 3: Smooth rotations using slerp on quaternions 
    smoothed_quaternions = apply_slerp_to_quats(quaternions)  # List(numpy.ndarray), len: num_poses

    # Step 4: Reconstruct smoothed c2w matrices 
    smoothed_c2w_matrices = [] 

    for i in range(len(c2w_matrices)): 
        smoothed_rot_matrix = R.from_quat(smoothed_quaternions[i]).as_matrix()  # (3, 3) from quaternion 
        smoothed_rot_matrix = torch.from_numpy(smoothed_rot_matrix)  # Convert to tensor [3, 3]
        smoothed_translation = smoothed_translations[i] 

        # Reconstruct the c2w matrix with smoothed rotation and translation 
        smoothed_c2w = torch.eye(4)
        smoothed_c2w[:3, :3] = smoothed_rot_matrix 
        smoothed_c2w[:3, 3] = smoothed_translation 
        smoothed_c2w_matrices.append(smoothed_c2w) 
    return smoothed_c2w_matrices 

def gaussian_filter_1d(kernel_size, sigma): 
    x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1) 
    kernel = torch.exp(-0.5 * (x.float() / sigma) ** 2) 
    kernel = kernel / kernel.sum() 
    return kernel 

def apply_gaussian_filter_to_trans(translations, kernel_size=5, sigma=2.0): 
    ''' 
    Apply 1D Gaussian filter to the translation components of camera poses. 
    ''' 
    # translations = torch.tensor(translations, dtype=torch.float32)
    translations = torch.stack(translations) # [num_poses, 3]
    kernel = gaussian_filter_1d(kernel_size, sigma).view(1, 1, -1)  # [1, 1, kernel_size] 

    # Pad the translations along the sequence dimension
    translations = translations.unsqueeze(0).permute(2, 0, 1)  # [3, 1, num_poses] 
    translations = F.pad(translations, (kernel_size // 2, kernel_size // 2), mode='replicate') 

    # Apply 1D convolution (Gaussian smoothing) 
    smoothed_translations = F.conv1d(translations, kernel, padding=0).squeeze().permute(1, 0)
    return smoothed_translations

def slerp(t, q0, q1): 
    ''' 
    Perform spherical linear interpolation (slerp) between two quaternions q0 and q1 
    t is a value between 0 and 1 indicating the interpolation factor. 
    ''' 
    q0 = R.from_quat(q0)
    q1 = R.from_quat(q1) 
    
    # Define the keyframe times (0 and 1 for the two quaternions)
    key_times = np.array([0, 1])
    
    # Create a Slerp object with the two quaternions
    slerp = Slerp(key_times, R.from_quat([q0.as_quat(), q1.as_quat()]))
    
    # Interpolate at time t (where t is between 0 and 1)
    interpolated_rotation = slerp(t).as_quat()
    smoothed_rotation = torch.from_numpy(interpolated_rotation)
    return smoothed_rotation

def apply_slerp_to_quats(quaternions): 
    ''' 
    Smooth quaternions using spherical linear interpolation (slerp). 
    ''' 
    num_poses = len(quaternions) 
    smoothed_quaternions = [quaternions[0]]  # Start with the first quaternion 

    for i in range(1, num_poses): 
        prev_q = smoothed_quaternions[-1] 
        next_q = quaternions[i] 
        smoothed_q = slerp(0.5, prev_q, next_q)  # Interpolate halfway 
        smoothed_quaternions.append(smoothed_q) 

    return smoothed_quaternions

def nanmin(tensor, dim=None):
    # Mask to filter out NaN values
    mask = ~torch.isnan(tensor)
    # Use masked_fill to replace NaNs with +inf so min ignores them
    tensor = tensor.masked_fill(~mask, float('inf'))
    # Compute minimum along the specified dimension
    result = torch.min(tensor, dim=dim)
    # If a dimension is specified, result is a tuple (values, indices), return only values
    return result.values if dim is not None else result


def visualize_colors(colors, filename, color_height=50, color_width=50):
    '''
    Visualize colors and save as an image
    
    Parameters:
        colors: Tensor of shape (N, 3) where N is the number of colors
        filename (str): Path to save the image
        color_height (int): The height of each color patch in the image
        color_width (int): The width of each color patch in the image
    '''
    # Convert tensor to numpy array and scales to [0, 255]
    colors_np = (colors.numpy() * 255).astype(np.uint8)

    # OpenCV uses BGR, so convert RGB colors to BGR
    colors_bgr = colors_np[:, ::-1]

    # Create an image with each color patch
    num_colors = colors.shape[0]
    img = np.zeros((color_height, color_width * num_colors, 3), dtype=np.uint8)

    for i, color in enumerate(colors_bgr):
        img[:, i * color_width:(i+1) * color_width] = color
    
    # Save the image
    cv2.imwrite(filename, img)

def load_palette(model):
    palette = model.basis_color.clone()
    original_palette = palette.clone()
    return original_palette

def rgb_to_hsv(rgb):
    '''
    Parameter:
        'rgb' are in [0, 1], shape [N, 3]
    Return:
        'hsv' of shape [N, 3]
            (1) 'h' is in the range of [0, 360]
            (2) 's' is in the range of [0, 1]
            (3) 'v' is in the range of [0, 1]
    '''
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]    # [N]
    max_val, _ = torch.max(rgb, dim=-1, keepdim=True)  # [N, 1]
    min_val, _ = torch.min(rgb, dim=-1, keepdim=True)  # [N, 1]
    diff = max_val - min_val  # [N, 1]

    h = torch.zeros_like(max_val)
    s = torch.zeros_like(max_val)
    v = max_val

    nonzero_diff_mask = diff > 0
    s[nonzero_diff_mask] = diff[nonzero_diff_mask] / max_val[nonzero_diff_mask]
    
    # Calculate hue only where there is a color difference
    for i in torch.where(nonzero_diff_mask)[0]:
        if max_val[i] == r[i]:
            h[i] = (60 * (g[i] - b[i]) / diff[i] + 360) % 360
        elif max_val[i] == g[i]:
            h[i] = (60 * (b[i] - r[i]) / diff[i] + 120) % 360
        elif max_val[i] == b[i]:
            h[i] = (60 * (r[i] - g[i]) / diff[i] + 240) % 360

    hsv = torch.cat((h, s, v), dim=-1)
    return hsv

def hsv_to_rgb(hsv):
    # 'h' in the range of [0, 360], 's' and 'v' in the range of [0, 1]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    c = v * s
    x = c * (1 - torch.abs(torch.fmod(h / 60.0, 2) -1))
    m = v - c

    zeros = torch.zeros_like(c)
    r, g, b = zeros.clone(), zeros.clone(), zeros.clone()

    h1 = (0 <= h) & (h < 60)
    r[h1], g[h1], b[h1] = c[h1], x[h1], zeros[h1]

    h2 = (60 <= h) & (h < 120)
    r[h2], g[h2], b[h2] = x[h2], c[h2], zeros[h2]

    h3 = (120 <= h) & (h < 180)
    r[h3], g[h3], b[h3] = zeros[h3], c[h3], x[h3]

    h4 = (180 <= h) & (h < 240)
    r[h4], g[h4], b[h4] = zeros[h4], x[h4], c[h4]

    h5 = (240 <= h) & (h < 300)
    r[h5], g[h5], b[h5] = x[h5], zeros[h5], c[h5]

    h6 = (300 <= h) & (h < 360)
    r[h6], g[h6], b[h6] = c[h6], zeros[h6], x[h6]

    rgb = torch.stack([r, g, b], dim=-1) + m.unsqueeze(-1)
    return rgb


def apply_palette_changes_ori(M, num_basis, original_palette, novel_palette, soft_color):
    '''
    Parameters:
        (1) 'M' is the number of points in one image (H*W)
        (2) 'num_basis' is the number of palette bases
        (3) 'original_palette': the original palette basis colors
            A tensor shape [num_basis, 3]
        (3) 'novel_palette': : the changed palette basis colors
            A tensor shape [num_basis, 3]
        (4) 'soft_color' = 'palette_bases' + 'offset'
            A tensor of shape [M, num_basis*3]
                each element will be in the range of [0., 1.]
    Return:
        'composed_rgb' is the final edited color
            A tensor of shape [M, 3]
    '''
    # Convert palette from RGB to HSV
    original_palette_hsv = rgb_to_hsv(original_palette)
    novel_palette_hsv = rgb_to_hsv(novel_palette)

    # Calculate 'Hue'(h) difference
    h_diff = novel_palette_hsv[..., 0] - original_palette_hsv[..., 0]  # [num_basis]
    # Calculate the scale of 'Saturation'(s) and 'value'(v) 
    s_scale = novel_palette_hsv[..., 1] / original_palette_hsv[..., 1] # [num_basis]
    v_scale = novel_palette_hsv[..., 2] / original_palette_hsv[..., 2] # [num_basis]

    # Convert soft color c_p from rgb to hsv space
    soft_color = soft_color.view(-1, 3)      # [M * num_basis, 3]
    soft_color_hsv = rgb_to_hsv(soft_color)  # [M * num_basis, 3]
    soft_color_hsv = soft_color_hsv.reshape(M, num_basis, 3)
    
    # Apply changes to soft color c_p
    for i in range(num_basis):
        soft_color_hsv[:, i, 0] += h_diff[i]
        soft_color_hsv[:, i, 1] *= s_scale[i]
        soft_color_hsv[:, i, 2] *= v_scale[i]

    # Ensure HSV values remain within valid range
    soft_color_hsv[..., 0] = soft_color_hsv[..., 0] % 360
    soft_color_hsv[..., 1] = torch.clamp(soft_color_hsv[..., 1], 0, 1)
    soft_color_hsv[..., 2] = torch.clamp(soft_color_hsv[..., 2], 0, 1)

    # Convert back to RGB
    print("soft_color_hsv_after_change:", soft_color_hsv.shape)
    soft_color_hsv = soft_color_hsv.view(-1, 3)
    soft_color_prime = hsv_to_rgb(soft_color_hsv) # [M * num_basis, 3]
    print("soft_color_prime:", soft_color_prime.shape)
    return soft_color_prime


def calculate_palette_changes(original_palette, novel_palette):
    '''
    Parameters:
        (1) 'original_palette': the original palette basis colors
            A tensor shape [num_basis, 3]
        (2) 'novel_palette': : the changed palette basis colors
            A tensor shape [num_basis, 3]
    Return:
        h_diff: [num_basis]
        s_scale: [num_basis]
        v_scale: [num_basis]
    '''
    print("Calculate the palette changes...")
    # Convert palette from RGB to HSV
    original_palette_hsv = rgb_to_hsv(original_palette)
    novel_palette_hsv = rgb_to_hsv(novel_palette)

    # Calculate 'Hue'(h) difference
    h_diff = novel_palette_hsv[..., 0] - original_palette_hsv[..., 0]  # [num_basis]
    # Calculate the scale of 'Saturation'(s) and 'value'(v) 
    s_scale = novel_palette_hsv[..., 1] / original_palette_hsv[..., 1] # [num_basis]
    v_scale = novel_palette_hsv[..., 2] / original_palette_hsv[..., 2] # [num_basis]

    print("Changes on Hue offset:", h_diff)
    print("Changes on Saturation Scale:", s_scale)
    print("Changes on Value Scale:", v_scale)

    return h_diff, s_scale, v_scale

def apply_palette_changes(M, num_basis, soft_color_hsv, h_diff, s_scale, v_scale):
    '''
    Parameters:
        (1) 'M' is the number of points in one image (H*W)
        (2) 'num_basis' is the number of palette bases
        (3) 'soft_color_hsv' = 'palette_bases' + 'offset'
            A tensor of shape [M * num_basis, 3]
                each element will be in the range of [0., 1.]
        (4) 'h_diff', 's_scale', and 'v_scale' are the palette changes
            Tensors of shape [num_basis]
    Return:
        'composed_rgb' is the final edited color
            A tensor of shape [M, 3]
    '''
    # Reshape soft color c_p in the hsv space
    soft_color_hsv = soft_color_hsv.reshape(M, num_basis, 3) # [M * num_basis, 3] -> [M, num_basis, 3]
    
    # Apply changes to soft color c_p
    for i in range(num_basis):
        soft_color_hsv[:, i, 0] += h_diff[i]
        soft_color_hsv[:, i, 1] *= s_scale[i]
        soft_color_hsv[:, i, 2] *= v_scale[i]

    # Ensure HSV values remain within valid range
    soft_color_hsv[..., 0] = soft_color_hsv[..., 0] % 360
    soft_color_hsv[..., 1] = torch.clamp(soft_color_hsv[..., 1], 0, 1)
    soft_color_hsv[..., 2] = torch.clamp(soft_color_hsv[..., 2], 0, 1)

    # Convert back to RGB
    soft_color_hsv = soft_color_hsv.view(-1, 3)
    soft_color_prime = hsv_to_rgb(soft_color_hsv) # [M * num_basis, 3]
    return soft_color_prime

def compose_palette_bases(M, num_basis, view_dep_color, soft_color, radiance, omega):
    '''
    Parameters:
        (1) 'M' is the number of points in one image (H*W)
        (2) 'num_basis' is the number of palette bases
        (3) 'view_dep_color' is a tensor shape [H, W, 3] or [M, 3]
        (4) 'soft_color' = 'palette_bases' + 'offset'
            A tensor of shape [M, num_basis*3] or [M*num_basis, 3]
                each element will be in the range of [0., 1.]
        (5) 'radiance' is the intensity of palettes
            A tensor of shape [M, 1]
        (6) 'omega' is the palette blending weights
            A tensor of shape [M, num_basis]
    Return:
        'composed_rgb' is the final edited color
            A tensor of shape [M, 3]
    '''
    view_dep_color = view_dep_color.view(-1, 3)  # [H, W, 3] -> [M, 3]
    radiance = radiance.view(-1, 1, 1)           # [M, 1] -> [M, 1, 1]
    omega = omega.view(-1, num_basis, 1)         # [M, num_basis] -> [M, num_basis, 1]
    soft_color = soft_color.reshape(M, num_basis, 3)  # [M, num_basis*3] or [M*num_basis, 3] -> [M, num_basis, 3]
    
    final_color = radiance * soft_color  # [M, num_basis, 3]
    basis_rgb = omega * final_color      # [M, num_basis, 3]
    composed_rgb = basis_rgb.sum(dim=-2) + view_dep_color # [M, 3]
    
    return composed_rgb


def get_palette_weight_with_hist(rgb, hist_weights):
    assert(hist_weights.ndim == 5)
    rgb_shape = rgb.shape
    rgb = rgb.reshape(-1, 3)
    rgb = rgb[None,None,None,:,[2,1,0]]*2-1
    weight = torch.nn.functional.grid_sample(hist_weights, rgb, mode='bilinear', padding_mode='zeros', align_corners=True)
    weight = weight.squeeze().permute(1, 0)
    return weight.reshape(rgb_shape[:-1] + (-1,))

def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

def index2r(r0, ratio, index: torch.Tensor):
    non_zero_idx = index > 0
    r = torch.zeros_like(index, dtype=torch.float32)
    r[non_zero_idx] = r0 * ratio ** (index[non_zero_idx] - 1)
    r[~non_zero_idx] = 0
    return r

def sph2cpp(h, w):
    theta = np.linspace(-np.pi, np.pi, w)    # [-pi, pi]
    phi = np.linspace(-np.pi/2, np.pi/2, h)  # [-pi/2, pi/2]
    theta, phi = np.meshgrid(theta, phi)
    
    # CPP coordinates (a scale from ERP)
    C_phi = np.arcsin(phi / np.pi) * 3                  # [-pi/2, pi/2]
    C_theta = theta / (2 * np.cos((2 * C_phi) / 3) - 1) # [-1.414847550405688e+16, 1.414847550405688e+16]

    # CPP from spherical to Cartesian coordinates
    ex_float = ((C_theta + np.pi) / (2 * np.pi)) * (w - 1) # [h, w], range: -4.3212038424619894e+18~4.321203842461992e+18 
    ey_float = ((C_phi + np.pi/2) / np.pi) * (h - 1)       # [h, w], range: -6.778115421163362e-14~959.0

    # Set float to int
    ex_int = np.rint(ex_float).astype(int)  # [h, w], range: -4321203842461989376~4321203842461991936
    ey_int = np.rint(ey_float).astype(int)  # [h, w], range: 0~959
    
    # Get the valid indices for sampling valid pixels from ERP to CPP
    mask_int = (ex_int >= w) | (ex_int < 0) | (ey_int >= h ) | (ey_int < 0)  # [h, w], bool
    valid_indices_int = np.where(~mask_int)

    # Process the two poles
    new_elements_y_start = np.array([0, 0])                 # [0, 0]
    new_elements_y_end = np.array([(h - 1), (h - 1)])       # [h - 1, h - 1]
    new_elements_x = np.array([int(w / 2 - 1), int(w / 2)]) # [959, 960]

    modified_y_cpp = np.concatenate([new_elements_y_start, valid_indices_int[0], new_elements_y_end])
    modified_x_cpp = np.concatenate([new_elements_x, valid_indices_int[1], new_elements_x])
    cpp_valid_indices = (modified_y_cpp, modified_x_cpp)  # integer indices for CPP

    # ex_valid_indices = ex[valid_indices]
    # ey_valid_indices = ey[valid_indices]
    # sph_valid_indices = ey_valid_indices * w + ex_valid_indices
    # return valid_indices, sph_valid_indices
    
    return cpp_valid_indices
    

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth)  # change nan to 0
    H, W = x.shape
    
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax
    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)

    # Set the invalid region to be white
    # valid_indices = sph2cpp(H, W)
    # invalid_mask = np.ones_like(x, dtype=bool)
    # invalid_mask[valid_indices] = False
    # x_[invalid_mask] = [255, 255, 255]
    
    return x_, [mi, ma]


def init_log(log, keys):
    for key in keys:
        log[key] = torch.tensor([0.0], dtype=float)
    return log


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]


def N_to_reso(n_voxels, bbox):
    xyz_min, xyz_max = bbox
    dim = len(xyz_min)
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / dim)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso, step_ratio=0.5, is_yinyang=True):
    assert True, 'Deprecated method. Fixed number of ray samples is used.'
    if is_yinyang:
        grid_num = 2 * reso[0] * reso[1] * reso[2]
        reso_sphere = []
        reso_sphere.append(int(pow(grid_num, 1 / 3) / 2))
        reso_sphere.append(reso_sphere[0] * 2)
        reso_sphere.append(reso_sphere[1] * 2)
        return int(np.linalg.norm(reso_sphere) / step_ratio)
    else:
        return int(np.linalg.norm(reso) / step_ratio)


__LPIPS__ = {}


def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    gt = torch.from_numpy(np_gt).permute([2, 0, 1]).contiguous().to(device)
    im = torch.from_numpy(np_im).permute([2, 0, 1]).contiguous().to(device)
    return __LPIPS__[net_name](gt, im, normalize=True).item()


def findItem(items, target):
    for one in items:
        if one[:len(target)]==target:
            return one
    return None


def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


def valid_rgb_ssim(img0, img1, 
                   max_val, valid_indices,
                   filter_size=11,
                   filter_sigma=1.5,
                   k1=0.01,
                   k2=0.03,
                   return_map=False):
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    
    # Get the SSIM map's shape
    ssim_map_height, ssim_map_width, _ = ssim_map.shape

    # Extract row and column indices from the valid_indices tuple
    row_valid_indices, col_valid_indices = valid_indices
    adjusted_row_indices = row_valid_indices - hw
    adjusted_col_indices = col_valid_indices - hw
    
    # Filetr out the indices that are out of bounds
    valid_row_mask = (adjusted_row_indices >= 0) & (adjusted_row_indices < ssim_map_height)
    valid_col_mask = (adjusted_col_indices >= 0) & (adjusted_col_indices < ssim_map_width)
    valid_mask = valid_row_mask & valid_col_mask

    # Create binary mask for SSIM map
    binary_mask = np.zeros((ssim_map_height, ssim_map_width), dtype=bool)
    adjusted_row_indices = adjusted_row_indices[valid_mask]
    adjusted_col_indices = adjusted_col_indices[valid_mask]
    binary_mask[adjusted_row_indices, adjusted_col_indices] = True

    # Calculate the masked SSIM map
    masked_ssim_map = ssim_map * binary_mask[..., np.newaxis] # addding channel dimension to match ssim_map
    ssim = np.mean(masked_ssim_map[binary_mask])
    return ssim_map if return_map else ssim


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]),2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]),2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


# Ray Entropy MInimization Loss below is borrowed from https://github.com/mjmjeong/InfoNeRF
def ray_entropy_loss(alpha: torch.Tensor):
    """
    alpha: 1 - exp(-volume_density * interval_length)
        - tensor size of (N_rays, N_samples)
        - include background sigma for environment map
    """
    ray_prob = alpha / (torch.sum(alpha, -1).unsqueeze(-1) + 1e-10)
    ray_entropy = -torch.sum(ray_prob * torch.log2(ray_prob + 1e-10), -1)
    return torch.mean(ray_entropy)


def convert_sdf_samples_to_ply(pytorch_3d_sdf_tensor, ply_filename_out, bbox, level=0.5, offset=None, scale=None):
    """
    Convert sdf samples to .ply

    :param pytorch_3d_sdf_tensor: a torch.FloatTensor of shape (n,n,n)
    :voxel_grid_origin: a list of three floats: the bottom, left, down origin of the voxel grid
    :voxel_size: float, the size of the voxels
    :ply_filename_out: string, path of the filename to save to

    This function adapted from: https://github.com/RobotLocomotion/spartan
    """

    numpy_3d_sdf_tensor = pytorch_3d_sdf_tensor.numpy()
    voxel_size = list((bbox[1]-bbox[0]) / np.array(pytorch_3d_sdf_tensor.shape))

    verts, faces, normals, values = skimage.measure.marching_cubes(
        numpy_3d_sdf_tensor, level=level, spacing=voxel_size
    )
    faces = faces[...,::-1] # inverse face orientation

    # transform from voxel coordinates to camera coordinates
    # note x and y are flipped in the output of marching_cubes
    mesh_points = np.zeros_like(verts)
    mesh_points[:, 0] = bbox[0,0] + verts[:, 0]
    mesh_points[:, 1] = bbox[0,1] + verts[:, 1]
    mesh_points[:, 2] = bbox[0,2] + verts[:, 2]

    # apply additional offset and scale
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    # try writing to the ply file

    num_verts = verts.shape[0]
    num_faces = faces.shape[0]

    verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

    for i in range(0, num_verts):
        verts_tuple[i] = tuple(mesh_points[i, :])

    faces_building = []
    for i in range(0, num_faces):
        faces_building.append(((faces[i, :].tolist(),)))
    faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

    el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
    el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

    ply_data = plyfile.PlyData([el_verts, el_faces])
    print("saving mesh to %s" % (ply_filename_out))
    ply_data.write(ply_filename_out)


def print_arguments(args):
    print('#####################################################################')
    print('#############################  Options  #############################')
    print('#####################################################################\n')
    for k, v in sorted(vars(args).items()):
        print(f'{str(k)}: \t{str(v)}'.expandtabs(36))
    print('#####################################################################\n')
