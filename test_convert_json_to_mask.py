import os
os.environ["PYTHONPATH"] = "/usr/local/lib/python3.6/site-packages/"
import yaml
import json
from glob import glob
from pycocotools.mask import encode, decode
import numpy as np
import matplotlib.pyplot as plt
from cocohacks import construct_dense_mask, construct_sparse_mask, dense_to_sparse
# load dictionary (str -> int)

from PIL import Image

fn =  "./tissuedict.yaml"
with open(fn) as fh:
    tissuedict = yaml.load(fh)
tissuedict

# for each json file
#    load the json file
#    for each roi in json file:
#       convert roi entry to binary mask
#       multiply entry by correspondent int from tissuedict

jsonfiles = glob("../data/data_1024/fullsplit/*/70bb3032750d09e7549928c0dbf79afc30d7cb68*json")

jsonfiles = list(filter(lambda ff: not ff.endswith("-summary.json"), jsonfiles))
#for ff in jsonfiles:
ff = jsonfiles[5]

with open(ff) as fh:
    rois = json.load(fh)

#
#maskarr = construct_dense_mask(rois, tissuedict)
#mask = dense_to_sparse(maskarr)

mask = construct_sparse_mask(rois, tissuedict)

fmask = ff.replace(".json", "-mask.png")

plt.imshow(mask)


fpng = ff.replace(".json", ".png")
img = np.asarray(Image.open(fpng))
plt.imshow(img)

maskarr

from PIL import Image
Image.fromarray(maskarr)
