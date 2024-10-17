# Training - Stage 1 (Reconstruction) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_recon_indoor.txt' renamed into 'common.txt' 
python main.py --config configs/omni/lab/default.txt 

# Only Evaluation and Inference Reconstruction Frames based on the stage 1's model
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_recon_indoor.txt' renamed into 'common.txt' 
python main.py --config configs/omni/lab/default.txt --evaluation 1

# Stabilization based on the stage 1's model 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_stabilize_indoor.txt' renamed into 'common.txt' 
ython main.py --config configs/omni/lab/default.txt --stabilize 1

# Training - Stage 2 (Color Decomposion) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_palette_indoor.txt' renamed into 'common.txt' 
python palette_main.py --config configs/omni/lab/default.txt

# Editing/Recoloring (Inference for recolored videos based on the stage 2's model) 
# change the corresponding .txt name
# For example, if train for indoor scene 'Lab' in werp grids, 
# please let the 'common_werp_palette_indoor.txt' renamed into 'common.txt' 
# where replace 'edit = False' in the ./renderer/def evaluation function with 'edit = True'
python palette_main.py --config configs/omni/lab/default.txt --evaluation 1
