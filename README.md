# OmniPlane: A Recolorable Representation for Dynamic Scenes in Omnidirectional Videos

This repository is the official PyTorch implementation of [OmniPlane](https://ieeexplore.ieee.org/document/10847904/media#media), which is accepted by TVCG. You can watch our [video demo](https://vimeo.com/1066790171?share=copy) here.

## Overview
OmniPlane is a feature-grid-based dynamic scene representation for modeling Omnidirectional (360-degree) videos, utilizing weighted ERP spherical coordinates with time dependency. Incorporating palette-based color decomposition to achieve intuitive recoloring.

## Installation
We tested the code on RTX 3090Ti GPU, with PyTorch 1.13.1 in a Python 3.9 environment.

### Clone the repository
```bash
git clone https://github.com/SiminKoux/OmniPlane-official-implementation.git
```

### Environment setup
```bash
cd OmniPlane-official-implementation

# Step1: Create and activate conda environment
conda create -n omniplanes python=3.9 -y
conda activate omniplanes

# Step2: Install dependencies
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# Step3: Ensure a compatible GCC version
# If your system's GCC version is newer than 10 (e.g., GCC 11 or 12), install GCC 10.4:
conda install gcc=10.4 gxx=10.4 -y
# If using GCC 9 or 10 in your system, you can skip this step.

# Step 4: Ensure a compatible CUDA version
# If your system's CUDA version is newer than 11.x (e.g., GCC 12.2 or 12.4), install CUDA 11.6:
conda install -c nvidia cudatoolkit=11.6 cuda-cccl=11.6 cuda-nvcc=11.6 cuda-cudart=11.6 cuda-cudart-dev=11.6 libcusparse=11.6 libcusparse-dev=11.6 libcusolver=11.6 libcusolver-dev=11.6 libcublas=11.6 libcublas-dev=11.6
# If using CUDA 11.6 or 11.8 in your system, you can skip this step.

# Step 5: Build the C++ extension
cd palette
pip install .
cd .. 
```
**Notes:**
The recommended setup is to install the appropriate GCC and CUDA versions directly on your system. Both can be installed in a local directory, and you can set the environment variables accordingly. If this setup is not convenient for you, you can safely follow Steps 3 and 4 to create the Conda environment.

## Dataset
You can download our _**Dyomni**_ dataset from [OneDrive](https://vuw-my.sharepoint.com/:f:/g/personal/kousi_staff_vuw_ac_nz/EsP60RZWBp1Dmn_lGAcKkTgBVqQvbKwvlNXFNvVR0bFfew?e=BymM5Q).<br>
<sub>**Note:** This OneDrive link is hosted by my university. However, access might be lost after I graduate, depending on the university’s policies.</sub>

To ensure long-term availability, you can also download the dataset from [Hugging Face](https://huggingface.co/datasets/SiminKou/DyOmni) using:
```bash
git clone https://huggingface.co/datasets/SiminKou/DyOmni.git
```
**Example Scene Provided in This Repository**

For your convenience in testing, we have also uploaded an example indoor scene, _**Lab**_, which allows you to directly run ``bash run.sh`` to obtain recolored results (e.g. representative_figure.png) along with the corresponding models.

**Dataset Composition**
```text
data/
└── DyOmni/
    ├── scene1/
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
    │   ├── time.txt
    │   └── train.txt
    ├── scene2/
    │   ├── erp_imgs/
    │   ├── output_dir/
    │   ├── test.txt
    │   ├── time.txt
    │   └── train.txt
    └── ...
```

**Note:**
The `train.txt` and `test.txt` both include all frames, aiming to evaluate its complete reconstruction performance, not focusing on novel view synthesis capacity. That said, our model does possess a certain generalization ability to generate novel views, so it also supports splitting out some test frames from the entire set of video frames. If you require a complete separation between training and testing images, you can simply adjust the numbers in the `train.txt` and `test.txt` files. These two files directly control which images from the scene are used for training and testing. For example, if you want to alternate frames—using one for training and the next for testing—the `train.txt` would include frames like `1, 3, 5, ..., 99`, while the `test.txt` would contain `2, 4, 6, ..., 100`. This example of train and test splits can be found in the `example` folder of this repository.

## Training
Our recolorable OmniPlane involves two training stages. 

For detailed timing estimates for each stage, refer to the **🕒 Timings** section later in this document.

**Stage 1: Train for OmniPlane**

To train OmniPlanes on individual scenes, run the script below.

*For indoor scenes:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt
# For example
python main.py --config configs/DyOmni/Lab/default.txt
```
*For outdoor scenes:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --time_grid 100 --r0 0.05 --distance_scale 10.0
# For example
python main.py --config configs/DyOmni/Campus/default.txt --time_grid 100 --r0 0.05 --distance_scale 10.0
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
python main.py --config configs/DyOmni/specific_instance/default.txt --palette_extract 1
# For example
python main.py --config configs/DyOmni/Lab/default.txt --palette_extract 1
```
To train color decomposition based on learned OmniPlane on individual scenes, run the script below.

*For indoor scenes:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --palette_train 1 --use_palette --n_iters 30000
# For example
python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1 --use_palette --n_iters 30000
```
*For outdoor scenes:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --palette_train 1 --use_palette --n_iters 30000 --time_grid 100 --r0 0.05 --distance_scale 10.0
# For example
python main.py --config configs/DyOmni/Campus/default.txt --palette_train 1 --use_palette --n_iters 30000 --time_grid 100 --r0 0.05 --distance_scale 10.0
```

## Skip Training with Pretrained Models ##
If you'd like to skip part or all of the training process, we've got you covered! You can download [pretrained models](https://vuw-my.sharepoint.com/:f:/g/personal/kousi_staff_vuw_ac_nz/El-mcQMQiCJLmdidP3-2bwkBim-12pKyAZ0zl0LVaMPl3g?e=RfOgNe) from OneDrive. These models are trained for the ``Lab`` scene. 

All necessary files should be placed in the directory: ``log/Lab/OmniPlanes``.

✅ What to prepare based on what you want to skip:
- **To skip Stage 1:**
  + Place ``OmniPlanes.th``
  + Then perform *palette extraction* and continue training for *palette-based color decomposition*
- **To skip Stage 1 and Palette Extraction:**
  + Place ``OmniPlanes.th`` and the ``palette`` folder
  + Then train for *palette-based color decomposition*
- **To skip the entire training process:**
  + Place ``OmniPlanes.th``, ``OmniPlanes_palette.th`` and the ``palette`` folder
  + ✨ Once set up, just follow the provided commands for *testing* and *editing* — no training required!

## Testing
To evaluate after training, run the script below.

*Evaluate the Stage 1's resulting model:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --evaluation 1
# For example
python main.py --config configs/DyOmni/Lab/default.txt --evaluation 1
```
*Evaluate the Stage 2's resulting model:*
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --evaluation 1 --use_palette
# For example
python main.py --config configs/DyOmni/Lab/default.txt --evaluation 1 --use_palette
```

## Editing
Our OmniPlane supports various editing applications, including recoloring, adjusting lighting or color textures, and enabling palette-based video segmentation. 
To generate edited videos using the learned recolorable OmniPlanes, run the script below.
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --palette_edit 1 --use_palette --edit_option
# For example
# Recoloring
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --recolor
# Relighting
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --relighting
# Retexture
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --retexture
# Visualize Palette-based Segmentation
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --visualize_seg
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

4. **Color Transfer**:
    - Leverage the **palette component**:
      Please follow the steps below to transfer the palette from the target scene:
      + Copy the `palette.npz` file from the target scene and place it at the path `log/current_scene/OmniPlanes`.
      + Rename the file as `transferred_palette_from_target_scene_name.npz` (replace `target_scene_name` with the actual name of the target scene).
      + Then, proceed by following the prompt in the `forward()` function within the `OmniPlanes.py` file.
      For reference, here’s an example that demonstrates how to transfer the palette from the target scene "Basketball" to the current scene "Lab" in the `forward()` function:

     ```python
     basis_color = self.basis_color[None, :, :].clamp(0, 1)  # [1, num_basis, 3]
     '''
     If performing color transfer, please annotate the row above, and unannotate the next three rows. 
     Also, update the load path and modify the .npz file by copying from your target's 'palette.npz' and renaming it.
     '''
     # exchanged_palette = np.load('./log/Lab/OmniPlanes/transfered_palette_from_Basketball.npz')['palette']
     # exchanged_palette = torch.from_numpy(exchanged_palette).to(device)
     # basis_color = exchanged_palette[[1, 0, 2, 3]][None, :, :].clamp(0, 1)
     ```

## Stabilization
To get the stabilized video frames, run the script below.
```bash
python main.py --config configs/DyOmni/specific_instance/default.txt --stabilize 1
# For example
python main.py --config configs/DyOmni/Lab/default.txt --stabilize 1
```

## Results Composition (include all operations)
```text
log/
├── scene1/
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
├── scene2/
│   └── OmniPlanes/
│       ├── evaluation/
│       ├── palette/
│       ├── OmniPlanes.th
│       ├── args.txt
│       └── config.txt
└── ...
```

## 🕒 Timings
This section provides the estimated timings for each step of the process based on our test environment. These timings are approximate and may vary depending on your specific setup.

| Step                      | Estimated Time | Notes                                         |
|---------------------------|----------------|-----------------------------------------------|
| **Stage 1: Training OmniPlane**                   | 3-4 hours                  | Time for 100,000 iterations.                   |
| **Stage 1 Inference**                             | 18-22 seconds per frame    | Time to render each frame with `OmniPlanes.th`. |
| **Palette Extraction (from `OmniPlanes.th`)**     | 25-30 minutes              | Time to extract a global palette across the video. |
| **Stage 2: Training for Palette-based Decomposition** | 1-2 hours              | Time for 30,000 iterations.                   |
| **Palette-based Editing Inference**              | 245-260 seconds per frame   | Time to render each frame with `OmniPlanes_palette.th` and modify components. |

## Citation
Cite as below if you find this paper, dataset, and repository helpful to you:
```
@article{kou2025omniplane,
  title={OmniPlane: A Recolorable Representation for Dynamic Scenes in Omnidirectional Videos},
  author={Kou, Simin and Zhang, Fang-Lue and Nazarenus, Jakob and Koch, Reinhard and Dodgson, Neil A},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  year={2025},
  publisher={IEEE}
}
```

## Acknowledgement
Our code is hugely influenced by [EgoNeRF](https://github.com/changwoonchoi/EgoNeRF), [HexPlane](https://github.com/Caoang327/HexPlane)，[PaletteNeRF](https://github.com/zfkuang/PaletteNeRF?tab=readme-ov-file) and many other projects. We would like to acknowledge them for making great code available for us.

## Copyright and license

Code and documentation copyright the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).
