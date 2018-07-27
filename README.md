# 2DVoxelmorph

## Notes
 - Python code written in 3.6.4
 - Several elements of the code such as 2D_datagenerators will need to be changed for other datasets, depending on direcotry locations, volume size, etc.
 - The atlas used can be found undere data/atlas_2D.npz

## Instructions

### Training:
 1. Change base_data_dir in train.py to the location of your image files.
 2. Change the paramaneters at the bottom oftrain.py as you see fit.
 2. Run train.py

### Testing:
1. Put test filenames in data/test_examples.txt, and anatomical labels in data/test_labels.mat.
2. Run test.py [model_name] [gpu-id] [iter-num]
3. The program will display the images for a visual check of the code.

## Credits
- The code in this repository both features and is based on voxelmorph code, which can be found at https://github.com/voxelmorph/voxelmorph
