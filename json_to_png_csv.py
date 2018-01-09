#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:38:43 2018

@author: dlituiev
"""

import pandas as pd
import os

fpaths = []
fnames = []
indir = "../../data/data_256_subsample_4x/fullsplit"
for dd in os.scandir(indir):
    for ff in os.scandir(dd):
        fpaths.append(ff.path)
#        fname.append(ff.path)
        

dffiles = pd.DataFrame({"files":fpaths})

#dffiles["fn"] = dffiles.files.map(os.path.basename)
dffiles["base"] = dffiles.files.map(lambda x: ".".join(x.split(".")[:-1]))
dffiles["ext"] = dffiles.files.map(lambda x: (x.split(".")[-1]))

dffiles = pd.pivot_table(dffiles, values='files', index='base', columns='ext', aggfunc=sum)
dffiles = dffiles.dropna()

dffiles = dffiles.reset_index(drop=True)
dffiles.to_csv("filepairs.csv", index=None)
#############################


from cocohacks import read_roi_to_sparse
from PIL import Image
import yaml
fn_roidict = "/Users/dlituiev/repos/kidney_histopath/data/tissuedict.yaml"
with open(fn_roidict) as fh:
    roidict = yaml.load(fh)
    
dffiles["json"]
outdir = "../../data/data_256_subsample_4x/fullsplit-masks"
os.makedirs(outdir, exist_ok=True)
for dd in os.scandir(indir):
    os.makedirs(os.path.join(outdir, dd.name), exist_ok=True)

def json_to_png_mask(fn, roidict=roidict):
    fnout = (os.path.join(outdir,*fn.split(os.sep)[-2:])).replace(".json", "") + "-mask.png"
    
    mask = read_roi_to_sparse(fn, roidict)
    mask = Image.fromarray(mask)
    mask.save(fnout)
    return fnout

dffiles["mask"] = dffiles["json"].map(json_to_png_mask)
#fn = dffiles["json"][4]
#dffiles["mask"] = fnout
#
#plt.imshow(
#    read_roi_to_sparse(fn, roidict)
#    )

dffiles[["png", "mask"]]
dffiles = dffiles.reset_index(drop=True)
dffiles[["png", "mask"]].to_csv("filepairs_png.csv", index=None)
