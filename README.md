# OmniPlane: A Recolorable Representation for Dynamic Scenes in Omnidirectional Videos

This repository is the offical PyTorch implementation of [OmniPlanes](https://ieeexplore.ieee.org/document/10847904/media#media), which is accepted by TVCG.

## Overview
OmniPlanes is a feature-grid-based dynamic scene representation for modeling Omnidirectional (360-degree) videos, utilizing weighted ERP spherical coordinates with time dependency. Incorporating palette-based color decomposition to achieve intuitive recoloring.

## Installation
We tested the code on RTX 3090Ti GPU, with PyTorch 1.13.1 in a Python 3.9 environment.

### Clone the repository
```bash
# SSH
$ git clone git@github.com:SiminKoux/Recolorable_OmniPlanes.git
```
or
```bash
# HTTP
$ git clone https://github.com/SiminKoux/Recolorable_OmniPlanes.git
```

### Environment setup
```bash
# Create conda environment
conda create -n omniplanes python=3.9

# Activate env
conda activate omniplanes

# Install dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```
## Dataset Composition
You can download our [DyOmni](https://vuw-my.sharepoint.com/:f:/g/personal/kousi_staff_vuw_ac_nz/EsP60RZWBp1Dmn_lGAcKkTgBVqQvbKwvlNXFNvVR0bFfew?e=BymM5Q) dataset.
```text
data/
└── omni/
    ├── video1/
    │   ├── cpp_imgs/
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   ├── erp_imgs/
    │   │   ├── 1.png
    │   │   ├── 2.png
    │   │   └── ...
    │   ├── output_dir/
    │   │   └── colmap/
    │   │       ├── cameras.txt
    │   │       ├── images.txt
    │   │       └── points3D.txt
    │   ├── test.txt
    │   └── train.txt
    ├── video2/
    │   ├── cpp_imgs/
    │   ├── erp_imgs/
    │   ├── output_dir/
    │   ├── test.txt
    │   └── train.txt
    └── ...
```

## Training
Our recolorable OmniPlane involves two training stages.

**Stage 1: Train for OmniPlane**

To train OmniPlanes on individual scenes, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt
# For example
python main.py --config configs/omni/lab/default.txt
```
If you want to exclude evaluation during the training process, you can set `--N_vis 0` or comment out the following lines in the `def train()` function:
```python
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
                interval_th=args.interval_th
            )
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
```

**Stage 2: Train for palette-based color decomposition on learned OmniPlane from stage 1**

Before starting color decomposition training, run the script below to extract the palette from the learned OmniPlane, which will serve as the palette initialization for Stage 2.
```bash
python main.py --config configs/omni/specific_instance/default.txt --palette_extract 1
# For example
python main.py --config configs/omni/lab/default.txt --palette_extract 1
```
To train color decompostion based on learned OmniPlane on individual scenes, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt --palette_train 1
# For example
python main.py --config configs/omni/lab/default.txt --palette_train 1
```

## Testing
To evaluate after training, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt --evaluation 1
# For example
python main.py --config configs/omni/lab/default.txt --evaluation 1
```

## Editing
Our OmniPlane supports various editing applications, including recoloring, adjusting lighting or color textures, and enabling palette-based video segmentation. 
To generate edited videos using the learned recolorable OmniPlanes, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt --palette_edit 1 --edit_option
# For example
# Recoloring
python main.py --config configs/omni/lab/default.txt --palette_edit 1 --recolor
# Relighting
python main.py --config configs/omni/lab/default.txt --palette_edit 1 --relighting
# Retexture
python main.py --config configs/omni/lab/default.txt --palette_edit 1 --retexture
# Visualize Palette-based Segmentation
python main.py --config configs/omni/lab/default.txt --palette_edit 1 --visualize_seg
```
### Custom Recolored Video Instructions
To get the custom recolored video, please apply the following changes in the `evaluation()` function of the `renderer.py` file:
1. **Ensure Edit Mode**:
   - Verify that the parameter `edit` is set to `True` and that the `--recolor` flag is enabled.

2. **Set Target RGB Value**:
   - Replace the `target_color` variable with your desired RGB color (normalized to the range [0, 1]).

     For example:
     
     ```python
     target_color = torch.tensor((1.0, 0.0, 0.0), dtype=novel_palette.dtype)
     ```
     This sets the target color to red `(1.0, 0.0, 0.0)`. It allows setting any target color you like.
3. **Update Recoloring Novel Palette Index**:
   - Modify the palette index in the following line:
     
     ```python
     novel_palette[index, :] = target_color
     ```
     Ensure that `index` is within the valid range of the optimized palette.
     
     Common values are `0, 1, 2, 3, 4`.

**Note: If it needs to change regions with different base colors, two or more `target_color` are allowed.**

For example,
 ```python
     target_color_1 = torch.tensor((1.0, 0.0, 0.0), dtype=novel_palette.dtype)  # red
     target_color_2 = torch.tensor((0.0, 0.0, 1.0), dtype=novel_palette.dtype)  # blue
     novel_palette[0, :] = target_color_1  # change the first base into red
     novel_palette[2, :] = target_color_2  # change the third base into blue
 ```
### Instructions for Other Edits
1. **Relighting**:
   - Adjust the **view-dependent color component**:
     Multiply the predicted view-dependent colors by a custom `view_dep_scale` within the `evaluation()` function of `renderer.py`.
     
     ```python
     elif lighting:
        view_dep_scale = 6   # default: 1, options: 0, 1, 3, 6
        edited_rgb = compose_palette_bases(M, num_basis, 
                                           rgb_map * view_dep_scale, 
                                           soft_color_map, 
                                           radiance_map, 
                                           omega_map)
     ```
     
2. **Retexture**:
    - Adjust the **palette offset component**:
      Apply a custom scale factor to the predicted offset within the `forward()` function of `OmniPlanes.py`.
      
     ```python
     scaled_color = basis_color.to(device) + 3 * offset  # 3 is the scale factor, can be changed to other values (e.g.: 0, 1, 6)
     ```
     
3. **Visualize Segmentation**:
    - Leverage the **palette weights component**:
      Set a threshold (e.g., 0.5) and define `target_indices` to specify the palette base colors to visualize in the `evaluation()` function of `renderer.py`.
      
     ```python
     # soft segmentation
     visualize_segmentation(omega_map, palette, H, W, savePath, idx+1, soft=True)
     # filtered hard segmentation
     visualize_segmentation(omega_map, palette, H, W, savePath, idx+1, soft=False, 
                            threshold=0.5, target_indices=[2], bg_color=[0.9, 0.9, 0.9])
     ```
     For hard segmentation, you can specify one or more palette base colors for visualization.
     For instance, setting `target_indices=[2]` selects the third base color in the optimized palette for visualization.

## Stabilization
To get the stabilized video frames, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt --stabilize 1
# For example
python main.py --config configs/omni/lab/default.txt --stabilize 1
```

## Results Composition (include all operations)
```text
log/
├── video1/
│   └── OmniPlanes/
│       ├── edit_results/
│       │   ├── original_frames/
│       │   ├── recolored_frames/
│       │   ├── relighted_frames/
│       │   ├── retextured_frames/
│       │   ├── segmentation/
│       │   ├── Novel_Palette.png
│       │   ├── Original_Palette.png
│       │   └── palette.pth
│       ├── evaluation/
│       │   ├── depth_maps/
│       │   ├── erp_recons/
│       │   └── mean.txt
│       ├── evaluation_palette/
│       │   ├── basis_imgs/
│       │   ├── depth_maps/
│       │   ├── diffuse_imgs/
│       │   ├── erp_recons/
│       │   ├── view_dep_imgs/
│       │   ├── palette.pth
│       │   └── mean.txt
│       ├── palette/
│       │   ├── extract-mesh_obj_files.obj
│       │   ├── extract-org-final.obj
│       │   ├── extract-palette.png
│       │   ├── extract-palette.txt
│       │   ├── extract-radiance-raw.png
│       │   ├── hist_weights.npz
│       │   └── palette.npz
│       ├── OmniPlanes.th
│       ├── OmniPlanes_palette.th
│       ├── args.txt
│       └── config.txt
├── video2/
│   └── OmniPlanes/
│       ├── evaluation/
│       ├── palette/
│       ├── OmniPlanes.th
│       ├── args.txt
│       └── config.txt
└── ...
```

## Acknowledgement
Our code is hugely influenced by [EgoNeRF](https://github.com/changwoonchoi/EgoNeRF), [HexPlane](https://github.com/Caoang327/HexPlane)，[PaletteNeRF](https://github.com/zfkuang/PaletteNeRF?tab=readme-ov-file) and many other projects. We would like to acknowledge them for making great code available for us.

## Copyright and license

Code and documentation copyright the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).
