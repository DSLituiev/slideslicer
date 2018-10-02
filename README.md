
## Setup

install cocotools

## Input data

the input data comes as
 (1) raw Aperio SVS images and 
 (2) ROI outlines stored as XML files.

the challenge is that SVS files are of huge size (~ 3e4 x 5e4),
while the tissue occupies less than a quarter of that area.

the set of scripts has been created to 
 (1) sample tissue and specific tissue features and
 (2) convert ROI outlines to masks and manipulate the masks.

The masks can be efficiently stored in MS-COCO format. 
This format dramatically compresses binary masks allowing to
store them in JSON files, preserving original label in free text form.

These MS-COCO JSON masks can be converted to one-hot [ height x width x classes] or sparse [height x width ] format. As a rule we store them in sparse format in png files when needed.

## Pipeline

    # download SVS file from Google Cloud Storage and sample patches from it
    pull_n_chop.sh

    # subsample if needed
    DATADIR="../data/......."
    subsample.py $DATADIR

    # split into train and test set
    makesets.sh

    # create sparse png masks from COCO JSON files
    json_to_png_csv.py
