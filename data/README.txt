Place your dataset images in these folders:

data/train/no_tumor/
data/train/tumor/
data/test/no_tumor/
data/test/tumor/

Use labelled MRI images and keep the same folder names. The Keras loader maps
alphabetically: no_tumor = 0 and tumor = 1. Keep patient-level separation
between train and test sets to reduce data leakage.

The dataset is not included in this ZIP. Obtain it from a permitted source and
check its licence/usage conditions before use or redistribution.
