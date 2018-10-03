
## Setup

install cocotools

## Input data

the input data consists of (1) raw Aperio SVS images and (2) ROI outlines stored as XML files.

the challenge is that SVS files are of huge size (~ 3e4 x 5e4),
while the tissue occupies less than a quarter of that area.

the set of scripts has been created to (1) sample tissue and specific tissue features and (2) convert ROI outlines to
masks and manipulate the masks.

the masks can be efficiently stored in MS-COCO format. This format dramatically compresses binary masks allowing to
store them in JSON files, preserving original label in free text form.

these MS-COCO JSON masks can be converted to one-hot [ height x width x classes] or sparse [height x width ] format. As a rule we store them in sparse format in png files.

## Functions



## Pipeline

    # download SVS file and sample patches from it
    pull_n_chop.sh

    # subsample if needed
    FACTOR=2 # produces 512x512
    FACTOR=4 # produces 256x256 

    DATADIR="/repos/data/data_1024/fullsplit/all"
    subsample.py $DATADIR $FACTOR

    # link inflammation vs everything else classes
    # BASEDIR="/repos/data/data_1024/"
    BASEDIR="/repos/data/data_128_subsample_8x/"
    ./link_binary_infl_norm.sh $BASEDIR

    # split into train and test set
    makesets.sh

    # create sparse png masks from COCO JSON files
    json_to_png_csv.py
