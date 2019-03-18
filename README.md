## Background

The challenge of whole slide imaging is that the files are of huge size (~ 3e4 x 5e4 pixels, ~300MB),
while the tissue often occupies less than a quarter of that area, especially in core biopsy slides.

## Functionality
This package provides tools to sample and read slides and annotations together at different resolutions and locations.

This package comes with a set of scripts to 
 1. sample tissue and specific tissue features and
 2. convert ROI outlines to masks and manipulate the masks.

The masks can be efficiently stored in run-length encoding MS-COCO format. 
This format dramatically compresses binary masks allowing to
store them in JSON files, preserving original label in free text form.

These MS-COCO JSON masks can be converted to one-hot `[height x width x classes]` or sparse `[height x width]` format. As a rule we store them in sparse format in png files when needed.

## Examples
- [reading a file](notebooks/demo-read-slide.ipynb)
- [train a keras model on patches](notebooks/demo-feed-keras.ipynb)

## Setup


Option A: Use a docker image:

    docker pull dslituiev/slideslicer # approx 2GB
    docker run -it -p 8899:8899 slideslicer:v1 # run docker with a jupyter notebook on port 8899

Option B: Native installation under Mac or Ubuntu/Debian:

**Step 1.** download and install `openslide` (a C library) 

+ OPTION 1 (fast): use a package manager
  * on MacOS with `brew`

        # install openslide on MacOS
        brew install openslide

  * on Debian / Ubuntu

        sudo apt-get install openslide-tools

  * [other platforms](https://openslide.org/download/)
 
+ OPTION 2 (slow but robust): build from source

      curl -LOk https://github.com/openslide/openslide/releases/download/v3.4.1/openslide-3.4.1.tar.gz
      tar xzvf openslide-3.4.1.tar.gz
      cd openslide-3.4.1
      ./configure && make && make install

**Step 2.** install the python package
  
    # install dependencies
    pip install cython
    pip install numpy
    # install slideslicer
    pip install git+https://github.com/DSLituiev/slideslicer

## Input data

_Currently `slideslicer` is created to handle Aperio SVS + associated XML annotation files. Please feel free to raise an
issue to request support / offer pull request for other formats_

the input data comes as
 1. a whole slide image (WSI)
 2. ROI outlines file (in XML format -- currently Leica SVS style XML only)


## Pipeline scripts

Use following command line tools for slicing multiple slides in command line:

    # download SVS file from Google Cloud Storage and sample patches from it
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
