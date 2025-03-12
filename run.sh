# Training - Stage 1 (Reconstruction) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_recon_indoor.txt' renamed into 'common.txt' 
python main.py --config configs/DyOmni/Lab/default.txt 

# Only Evaluation and Inference Reconstruction Frames based on the stage 1's model
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_recon_indoor.txt' renamed into 'common.txt' 
# please uncomment the following command if you need to evaluate and rendering the final images
# python main.py --config configs/DyOmni/Lab/default.txt --evaluation 1

# Extract initialized palette based on the stage 1's model
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_recon_indoor.txt' renamed into 'common.txt' 
python main.py --config configs/DyOmni/Lab/default.txt --palette_extract 1

# Stabilization based on the stage 1's model 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_stabilize_indoor.txt' renamed into 'common.txt'
# please uncomment the next command to achieve stablization
# python main.py --config configs/DyOmni/Lab/default.txt --stabilize 1

# Training - Stage 2 (Color Decomposion) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_palette_indoor.txt' renamed into 'common.txt' 
python main.py --config configs/DyOmni/Lab/default.txt --palette_train 1

# Editing (Inference for recolored videos based on the stage 2's model) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_palette_indoor.txt' renamed into 'common.txt'
# please uncomment the following commands to activate their respective editing functions
# There are different editing options:
# 1. Recoloring
python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --recolor
# 2. Relighting
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --relighting
# 3. Retexturing
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --retexture
# 4. soft and hard segmentation
# python main.py --config configs/DyOmni/Lab/default.txt --palette_edit 1 --visualize_seg
