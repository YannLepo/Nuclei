# Nuclei
The idea here is to get different labels for each nuclei, not only a segmentation of them.

As each nuclei, even if overlapping, have to be identified as separate ones 3 labels are used to train the classifier.
Label 0 for background, label 1 for boundaries of the nuclei and label 1 for the interior of the cells.

## Data augmentation
Images are cropped to the same size to form batches. 
They are randomly flipped or not.
Following the idea of Mixup : https://arxiv.org/abs/1710.09412, images and labels are mixed to improve learning.

## Cost function
The cost function computes the binary cross entropy for the classes and contains a constraint on the neighbouring regions.
If background directly touches the interior of the cell, this constraints is non zero.

## Make submission
Lables for the nuclei are computed here. Each interior gets a label. Boundaries are then associated to the closest interior.
Finally, the output is formated into a run-length format (for kaggle submission).



