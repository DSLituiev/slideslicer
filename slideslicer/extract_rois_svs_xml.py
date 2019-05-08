
# coding: utf-8
import os
import sys
import re
import json
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import numpy as np
from shapely.geometry import Polygon
from warnings import warn

from .parse_leica_xml import parse_xml2annotations

from .slideutils import (get_vertices, get_roi_dict, get_median_color,
                        get_threshold_tissue_mask, convert_mask2contour,
                        get_thumbnail_magnification)

from .roi_reader import remove_empty_tissue_chunks

## Read XML ROI, convert, and save as JSON
def _shapely_polygon_from_roi_(roi):
    return Polygon(roi["vertices"])


def extract_rois_svs_xml(fnxml, remove_empty=True, outdir=None, minlen=50, keeplevels=1):
    """
    extract and save rois

    Inputs:
    fnxml         -- xml path
    remove_empty  -- remove empty chunks of tissue
    outdir        -- (optional); save into an alternative directory
    minlen        -- minimal length of tissue chunk contour in thumbnail image
    keeplevels    -- number of path elements to keep 
                  when saving to provided `outdir`
                  (1 -- filename only; 2 -- incl 1 directory)
    """
    fnsvs = re.sub("\.xml$", ".svs", fnxml)
    fnjson = re.sub(".xml$", ".json", fnxml)
    if outdir is not None and os.path.isdir(outdir):
        fnjson = fnjson.split('/')[-keeplevels]
        fnjson = os.path.join(outdir, fnjson)
        os.makedirs(os.path.dirname(fnjson), exist_ok = True)


    ############################
    # parsing XML
    ############################
    try:
        roilist = parse_xml2annotations(fnxml)
    except OSError as ee:
        roilist = []
        warn(str(ee))

    for roi in roilist:
        roi["name"] = roi.pop("text").lower().rstrip('.')
    ############################

    # for an ellipse, 
    #    area = $\pi \times r \times R$

    ############################
    ## Extract tissue chunk ROIs
    ############################

    slide = openslide.OpenSlide(fnsvs)
    img = np.asarray(slide.associated_images["thumbnail"])

    median_color = get_median_color(slide)

    ## Extract mask and contours
    mask = get_threshold_tissue_mask(img, color=False, filtersize=7)
    contours = convert_mask2contour(mask, minlen = minlen)

    ratio = get_thumbnail_magnification(slide)

    sq_micron_per_pixel = np.median([roi["areamicrons"] / roi["area"] for roi in roilist])

    tissue_roilist = [get_roi_dict(cc*ratio, name='tissue',
                                   id=1+nn+len(roilist),
                                   sq_micron_per_pixel=sq_micron_per_pixel) 
                          for nn,cc in enumerate(contours)]

    roilist = roilist + tissue_roilist

    roi_name_counts = pd.Series([rr["name"] for rr in roilist]).value_counts()
    print("counts of roi names")
    print(roi_name_counts)

    if remove_empty:
        roilist = remove_empty_tissue_chunks(roilist)
        roi_name_counts = pd.Series([rr["name"] for rr in roilist]).value_counts()
        print("counts of roi names after removing empty chunks")
        print(roi_name_counts)
    ## Save both contour lists together
    with open(fnjson, 'w+') as fh:
        json.dump(roilist, fh)

    return fnjson

if __name__ == "__main__":
    fnxml = sys.argv[1]

    outfile = extract_rois_svs_xml(fnxml)
    print(outfile)
