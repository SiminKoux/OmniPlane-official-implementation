# OmniPlane: A Recolorable Representation for Dynamic Scenes in Omnidirectional Videos

This repository is the offical PyTorch implementation of [OmniPlanes](https://ieeexplore.ieee.org/document/10847904/media#media).

## Overview
OmniPlanes is a feature-grid-based dynamic scene representation for modeling Omnidirectional (360-degree) videos, utilizing weighted ERP spherical coordinates with time dependency. Incorporating palette-based color decomposition to achieve intuitive recoloring.

## Installation
We tested the code on RTX 3090Ti GPU, PyTorch 1.10.1 in a Python 3.9 environment.

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
pip install torch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1
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
To train OmniPlanes on individual scenes, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt
# For example
python main.py --config configs/omni/lab/default.txt
```
If you want to exclude evaluation during the training process, you can comment out the following lines in the `def train()` function:
```
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

To train color decompostion based on learned OmniPlanes on individual scenes, run the script below.
```bash
python palette_main.py --config configs/omni/specific_instance/default.txt
# For example
python palette_main.py --config configs/omni/lab/default.txt
```

## Testing
To evaluate after training, run the script below.
```bash
python main.py --config configs/omni/specific_instance/default.txt --evaluation 1
# For example
python main.py --config configs/omni/lab/default.txt --evaluation 1
```

## Editing/Recoloring
To get the recolored videos based on the learned recolorable OmniPlanes, Run the script below.
```bash
python palette_main.py --config configs/omni/specific_instance/default.txt --evaluation 1
# For example
python palette_main.py --config configs/omni/lab/default.txt --evaluation 1
```
### Custom Recolored Video Instructions
To get the custom recolored video, please apply the following changes in the `def evaluation()` function of the `renderer.py` file:
1. **Enable Edit Mode**:
   - Replace the line:
     ```python
     edit = False
     ```
     with:
     ```python
     edit = True
     ```
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
│       ├── evaluation/
│       │   ├── basis_imgs/
│       │   ├── depth_maps/
│       │   ├── diffuse_imgs/
│       │   ├── edited_frames/
│       │   ├── erp_recons/
│       │   ├── original_frames/
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
│       ├── args.txt
│       ├── config.txt
│       ├── evaluation_Novel_Palette.png
│       └── evaluation_Original_Palette.png
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
Our code is hugely influenced by [EgoNeRF](https://github.com/changwoonchoi/EgoNeRF), [HexPlane](https://github.com/Caoang327/HexPlane) and many other projects. We would like to acknowledge them for making great code openly avaliable for us to use.

## Copyright and license

Code and documentation copyright the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).
