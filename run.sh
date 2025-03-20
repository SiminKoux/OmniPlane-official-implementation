### Please unannotate corresponding commands to achieve respective functions ###

# Training - Stage 1 (Reconstruction) 
# For Indoor scenes (Basketball, CableCar, Corridor, Floor, Lab, Shop, Studio, Tunnel)
python main.py --config configs/DyOmni/Lab/default.txt 
# For outdoor scenes (Ayutthaya, Campus, City, Garden, GreatWall, Square, Stage, Street)
# python main.py --config configs/DyOmni/Campus/default.txt --time_grid 100 --r0 0.05 --distance_scale 10.0
# For erp coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --sampling_method "simple"
# For erp coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0
# For yin-yang coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --coordinates "yinyang" --sampling_method "simple"
# For yin-yang coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --coordinates "yinyang" --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0
# For cpp coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --dataset_name "cpp_omnivideos" --sampling_method "simple"
# For cpp coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --dataset_name "cpp_omnivideos" --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0

# Only Evaluation and Inference Reconstruction Frames based on the stage 1's model
# python main.py --config configs/DyOmni/Lab/default.txt --evaluation 1

# Extract initialized palette based on the stage 1's model
python main.py --config configs/DyOmni/Lab/default.txt --palette_extract 1

# Stabilization based on the stage 1's model
# python main.py --config configs/DyOmni/Lab/default.txt --stabilize 1

# Training - Stage 2 (Color Decomposion) 
# For indoor scenes
python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1 --use_palette --n_iters 30000
# For outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --palette_train 1 --use_palette --n_iters 30000 --time_grid 100 --r0 0.05 --distance_scale 10.0
# For erp coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1 --use_palette --n_iters 30000 --sampling_method "simple" 
# For erp coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --palette_train 1 --use_palette --n_iters 30000 --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0
# For yin-yang coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1 --use_palette --n_iters 30000 --coordinates "yinyang" --sampling_method "simple"
# For yin-yang coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --palette_train 1 --use_palette --n_iters 30000 --coordinates "yinyang" --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0
# For cpp coordinate, indoor scenes
# python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1 --use_palette --n_iters 30000 --dataset_name "cpp_omnivideos" --sampling_method "simple"
# For cpp coordinate, outdoor scenes
# python main.py --config configs/DyOmni/Campus/default.txt --palette_train 1 --use_palette --n_iters 30000 --dataset_name "cpp_omnivideos" --sampling_method "simple" --time_grid 100 --r0 0.05 --distance_scale 10.0

# Only Evaluation and Inference Reconstruction Frames based on the stage 2's model
# python main.py --config configs/DyOmni/Lab/default.txt --evaluation 1 --use_palette

# Editing (Inference for recolored videos based on the stage 2's model) 
# There are different editing options:
# 1. Recoloring
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --recolor
# 2. Relighting
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --relighting
# 3. Retexturing
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --retexture
# 4. soft and hard segmentation
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --visualize_seg
# 5. color transfer
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --use_palette --color_transfer