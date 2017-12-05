
# coding: utf-8
import os
import sys
import re
import json
import openslide
import numpy as np
import pandas as pd
from collections import Counter
from bs4 import BeautifulSoup
#import cv2
import numpy as np

from slideutils import (get_vertices, get_roi_dict, get_median_color,
                        get_chunk_masks, get_contours_from_mask,
                        get_thumbnail_magnification)

## Read XML ROI, convert, and save as JSON


def extract_rois_svs_xml(fnxml, outdir=None, keeplevels=1):
    """
    extract and save rois

    Inputs:
    fnxml      -- xml path
    outdir     -- (optional); save into an alternative directory
    keeplevels -- number of path elements to keep 
                  when saving to provided `outdir`
                  (1 -- filename only; 2 -- incl 1 directory)
    """
    fnsvs = re.sub("\.xml$", ".svs", fnxml)
    fnjson = re.sub(".xml$", ".json", fnxml)
    if outdir is not None and os.path.isdir(outdir):
        fnjson = fnjson.split('/')[-keeplevels]
        fnjson = os.path.join(outdir, fnjson)
        os.makedirs(os.path.dirname(fnjson), exist_ok = True)

    with open(fnxml) as fh:
        soup = BeautifulSoup(fh, 'lxml')
    regions = soup.find_all("region")

    # fine-parse and format the extracted rois:
    roilist = []
    for rr in regions:
    #     name = rr.get("text").lower().rstrip('.')
        attrs_ = rr.attrs.copy()
        if ("text" in attrs_) and not ("name" in attrs_):
            attrs_["name"] = attrs_.pop("text").lower().rstrip('.')
        for kk,vv in attrs_.items():
            if isinstance(vv,str) and vv.isdigit():
                attrs_[kk] = int(vv)
            else:
                try:
                    attrs_[kk] = float(vv)
                except:
                    if attrs_[kk]=='':
                        attrs_[kk]=None
                    continue
        attrs_["vertices"] = get_vertices(rr)
        roilist.append(attrs_)

    # for an ellipse, 
    #    area = $\pi \times r \times R$

    roi_name_counts = pd.Series([rr["name"] for rr in roilist]).value_counts()
    print("counts of roi names")
    print(roi_name_counts)

    #with open(fnjson, 'w+') as fh:
    #    json.dump(roilist, fh)


    ## Extract tissue chunk ROIs

    slide = openslide.OpenSlide(fnsvs)
    img = np.asarray(slide.associated_images["thumbnail"])

    median_color = get_median_color(slide)

    ## Extract mask and contours
    mask = get_chunk_masks(img, color=False, filtersize=7)
    contours = get_contours_from_mask(mask, minlen = 100)

    ratio = get_thumbnail_magnification(slide)

    sq_micron_per_pixel = np.median([roi["areamicrons"] / roi["area"] for roi in roilist])

    tissue_roilist = [get_roi_dict(cc*ratio, name='tissue', id=nn+len(roilist), sq_micron_per_pixel=sq_micron_per_pixel) 
                          for nn,cc in enumerate(contours)]

    ## Save both contour lists together
    with open(fnjson, 'w+') as fh:
        json.dump(roilist + tissue_roilist, fh)

    return fnjson

if __name__ == "__main__":
    fnxml = "examples/6371/6371 1.xml"

    outfile = extract_rois_svs_xml(fnxml)
    print(outfile)
