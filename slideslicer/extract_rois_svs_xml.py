
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
from shapely.geometry import Polygon
from parse_leica_xml import parse_xml2annotations

from slideutils import (get_vertices, get_roi_dict, get_median_color,
                        get_chunk_masks, get_contours_from_mask,
                        get_thumbnail_magnification)

## Read XML ROI, convert, and save as JSON

def _shapely_polygon_from_roi_(roi):
    return Polygon(roi["vertices"])


def find_chunk_content(roilist):
    """finds features (gloms, infl, etc) contained within tissue chunks.
    Returns a dictionary:
    {tissue_chunk_1_id: [feature_1_id, ..., feature_n_id],
     tissue_chunk_1_id: [...]
    }
    Requires `shapely` package
    """
    pgs_tissue = {}
    pgs_feature = {}
    for roi in roilist:
        if roi["name"]=="tissue":
            pgs_tissue[roi['id']] = Polygon(roi["vertices"])
        else:
            pgs_feature[roi['id']] = Polygon(roi["vertices"])

    tissue_contains = dict(zip(pgs_tissue.keys(), [[] for _ in range(len(pgs_tissue))]))
    remove_items = []
    for idt, pt in pgs_tissue.items():
        for idf in remove_items:
            pgs_feature.pop(idf)
        remove_items = []
        for idf, pf in pgs_feature.items():
            if pt.intersects(pf):
                remove_items.append(idf)
                tissue_contains[idt].append(idf)
    return tissue_contains


def remove_empty_tissue_chunks(roilist):
    """removes tissue chunks that contain no annotation contours within"""
    chunk_content = find_chunk_content(roilist)
    empty_chunks = set([kk for kk,vv in chunk_content.items() if len(vv)==0])
    return [roi for roi in roilist if roi['id'] not in empty_chunks]


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
    roilist = parse_xml2annotations(fnxml)
    for roi in roilist:
        roi["name"] = roi.pop("text").lower().rstrip('.')
    #import ipdb; ipdb.set_trace()

    #with open(fnxml) as fh:
    #    soup = BeautifulSoup(fh, 'lxml')
    #regions = soup.find_all("region")

    ## fine-parse and format the extracted rois:
    #roilist = []
    #for rr in regions:
    ##     name = rr.get("text").lower().rstrip('.')
    #    attrs_ = rr.attrs.copy()
    #    if ("text" in attrs_) and not ("name" in attrs_):
    #        attrs_["name"] = attrs_.pop("text").lower().rstrip('.')
    #    for kk,vv in attrs_.items():
    #        if isinstance(vv,str) and vv.isdigit():
    #            attrs_[kk] = int(vv)
    #        else:
    #            try:
    #                attrs_[kk] = float(vv)
    #            except:
    #                if attrs_[kk]=='':
    #                    attrs_[kk]=None
    #                continue
    #    attrs_["vertices"] = get_vertices(rr)
    #    roilist.append(attrs_)

    ############################
    ############################

    # for an ellipse, 
    #    area = $\pi \times r \times R$


    #with open(fnjson, 'w+') as fh:
    #    json.dump(roilist, fh)


    ## Extract tissue chunk ROIs

    slide = openslide.OpenSlide(fnsvs)
    img = np.asarray(slide.associated_images["thumbnail"])

    median_color = get_median_color(slide)

    ## Extract mask and contours
    mask = get_chunk_masks(img, color=False, filtersize=7)
    contours = get_contours_from_mask(mask, minlen = minlen)

    ratio = get_thumbnail_magnification(slide)

    sq_micron_per_pixel = np.median([roi["areamicrons"] / roi["area"] for roi in roilist])

    tissue_roilist = [get_roi_dict(cc*ratio, name='tissue', id=1+nn+len(roilist), sq_micron_per_pixel=sq_micron_per_pixel) 
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
