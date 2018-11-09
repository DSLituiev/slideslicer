# coding: utf-8

from PIL import Image
from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import pandas as pd
import os
import re
import json
import openslide
import cv2
from pycocotools.mask import encode, decode

from slideslicer.extract_rois_svs_xml import extract_rois_svs_xml
from slideslicer.slideutils import (plot_contour, get_median_color, 
                        get_thumbnail_magnification,
                        get_img_bbox, get_rotated_highres_roi,
                        get_uniform_tiles, 
                        get_threshold_tissue_mask, 
                        convert_contour2mask,
                        convert_mask2contour,
                        CropRotateRoi,
                        get_contour_centre, read_roi_patches_from_slide,
                        clip_roi_wi_bbox, sample_points)


def get_img_id(svsname):
    imgid = re.sub("\.svs$","", 
                   os.path.basename(svsname)
                   ).replace(" ", "_").replace("-","_")
    return imgid

def get_prefix(imgid, pos, name, tissueid, id, parentdir = "data", suffix=''):
    prefix = "{parentdir}/{typ}/{imgid}-{pos}-t{tissue}-r{roiid}-{typ}{suffix}".format(**{
                                        "tissue":tissueid,
                                        "pos": "x{}-y{}".format(*pos),
                                        "parentdir":parentdir,
                                        "imgid":imgid,
                                        "roiid":id,
                                        "typ": (name.replace(" ","_")),
                                        "suffix":suffix,
                                        })
    return prefix


def summarize_rois_wi_patch(rois, bg_names = ["tissue"]):
    names = []
    areas = []
    ids = []
    
    tissue_info = []
    for rr in rois:
        if rr['name'] in bg_names:
            tissue_info.append(rr)
            continue
        names.append(rr['name'])
        areas.append(rr['area'])
        ids.append(rr['id'])
#     assert (len(tissue_info)==1)
    tissue_id = "+".join(sorted(["%s"%tt['id'] for tt in tissue_info]))
    dfareas = (pd.DataFrame(dict(area=areas, name=names, id=ids))
                     .sort_values("area", ascending=False)
               )
    areasum = (dfareas.groupby('name')
                     .agg({"area":sum, "id": "first"})
                     .sort_values("area", ascending=False)
              )
    if len(areasum)==1:
        name = areasum.index[0]
        id = areasum["id"][0]
    elif areasum["area"][0]/areasum["area"][1] > 3:
        name = areasum.index[0]
        id = areasum["id"][0]
    else:
        name = '+'.join(areasum.index.tolist())
        id = '+'.join(sorted(areasum["id"].astype(str).tolist()))
    return {"name":name,
            "id": str(id),
            "tissue_id": tissue_id,
            "stats": dfareas.to_dict(orient='records')}


# Rewrite for generator if needed:
def visualise_chunks_and_rois(img_arr, roi_cropped_list,
                              nrows = 5, figsize=(15,15)
                             ):
    fig, axs = plt.subplots(nrows,len(img_arr)//nrows, figsize=figsize)
    for ax, reg, rois in zip(axs.ravel(), img_arr, roi_cropped_list):
        ax.imshow(reg)
        for rr in rois:
            if rr['name'] == 'tissue':
                continue
            plot_contour(rr["vertices"], ax=ax)
        xlab = "\n".join(["{}: {}".format(rr['id'], rr['name']) \
                          for rr in rois if rr['name'] !='tissue'])
        ax.set_xlabel(xlab)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        

def get_tissue_rois(slide,
                    roilist,
                    vis = False,
                    step = 1024,
                    magnlevel = 0,
                    target_size = None,
                    maxarea = 1e7,
                    random=False,
                    normal_only=True,
                    shift_factor = 2, 
                   ):

    print("NORMAL_ONLY", normal_only)
    if target_size is None:
        target_size = [step]*2

    tissue_rois = [roi for roi in roilist if roi['name']=='tissue']

    for roi in tissue_rois:
        print("id", roi["id"])
        cont = roi["vertices"]
        points = sample_points(cont,
                              spacing = step,
                              shift = -step//shift_factor,
                              mode = 'random' if random else 'grid')

        print("roi {} #{}:\t{:d} points sampled".format(roi["name"], roi["id"],len(points), ))
        pointroilist = [{"vertices":[pp], "area":0} for pp in points]
        
#         img_arr, roi_cropped_list, msk_arr, = \
        imgroiiter = read_roi_patches_from_slide(slide, 
                                        pointroilist,
                                        but_list = roilist,
                                        target_size = target_size,
                                        magnlevel = magnlevel,
                                        maxarea = maxarea,
                                        color=1,
                                        nchannels=3,
                                        allcomponents = True,
                                        nomask=True,
                                       )
#         if vis:
#             plt.scatter(points[:,0], points[:,1],c='r')
#             plot_contour(cont)
        # filter for rois with only normal tissue 
        def filter_(x):
            return all(roi['name']=='tissue' for roi in x[1])
        if normal_only:
            imgroiiter = filter(filter_, imgroiiter)
        yield imgroiiter


def save_tissue_chunks(imgroiiter, imgid, parentdir="data",
                       lower = [0, 0, 180],
                       upper = [179, 10, 255],
                       close=50,
                       open_=30,
                       filtersize = 20,
                       ):
    for ii, (reg, rois, _, start_xy) in enumerate(imgroiiter):
        sumdict = summarize_rois_wi_patch(rois, bg_names = [])
        prefix = get_prefix(imgid, start_xy, sumdict["name"], sumdict["id"], ii,
                            parentdir=parentdir,)

        #fn_summary_json = prefix + "-summary.json"
        fn_json = prefix + ".json"
        fnoutpng = prefix + '.png'
        print(fnoutpng)

        os.makedirs(os.path.dirname(fn_json), exist_ok=True)
        #with open(fn_summary_json, 'w+') as fhj: json.dump(sumdict, fhj)
        if isinstance(reg, Image.Image):
            reg.save(fnoutpng)
        else:
            Image.fromarray(reg).save(fnoutpng)

        rois = add_roi_bytes(rois, np.asarray(reg),
                lower=lower, upper=upper,
                open=open_, close=close,
                filtersize=filtersize)
        with open(fn_json, 'w+') as fhj: json.dump(rois, fhj)


def add_roi_bytes(rois, reg,
                  lower = [0, 0, 180],
                  upper = [179, 25, 255],
                  filtersize=25,
                  close=True,
                  open=False,
                  minlen = -1):
    if minlen==-1:
        minlen=filtersize
    rois = rois.copy()
    tissue_roi = None
    other_mask_ = 0
    
    for roi_ in rois:
        if roi_["name"] == "tissue":
            tissue_roi = roi_
            continue
        mask_ = convert_contour2mask(roi_["vertices"], reg.shape[1], reg.shape[0],
                             fill=1, order='F')

        cocomask = encode(np.asarray(mask_, dtype='uint8'))
        cocomask["counts"] = cocomask["counts"].decode('utf-8')
        roi_.update(cocomask)
        if isinstance(roi_["vertices"], np.ndarray):
            roi_["vertices"] = roi_["vertices"].tolist()
        other_mask_ = np.maximum(other_mask_, mask_)
    
    for roi_ in [tissue_roi]:
        if roi_ is None:
            print("Someting strange is going on. Make sure no tissue chunks are missing")
        if reg is not None:
            mask_ = get_threshold_tissue_mask(reg, color=True, filtersize=filtersize, dtype=bool,
                                    open=open, close=close,
                                    lower = lower, upper = upper)
            if mask_.sum()==0:
                roi_ = None
                print("skipping empty mask", roi_['name'], roi_['id'])
                continue
            verts = convert_mask2contour(mask_.astype('uint8'), minlen=minlen)
            # print("verts", len(verts))
            if len(verts)>0:
                roi_["vertices"] = verts[np.argmax(map(len,verts))]
            else:
                #print("verts", len(verts), roi_["vertices"])
                pass
            mask_ = np.asarray(mask_, order='F')
        else:
            mask_ = convert_contour2mask(roi_["vertices"], reg.shape[1], reg.shape[0], 
                                 fill=1, order='F')
            if mask_.sum()==0:
                roi_ = None
                continue
        if isinstance(other_mask_, np.ndarray):
            mask_ = mask_.astype(bool) & ~other_mask_.astype(bool)
        cocomask = encode(np.asarray(mask_, dtype='uint8'))
        cocomask["counts"] = cocomask["counts"].decode('utf-8')
        roi_.update(cocomask)
        if isinstance(roi_["vertices"], np.ndarray):
            roi_["vertices"] = roi_["vertices"].tolist()   

    return rois


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data-root',
      type=str,
      default='../data',
      help='The directory where the input data will be stored.')

    parser.add_argument(
      '--json-dir',
      type=str,
      default='../data/roi-json',
      help='The directory where the roi JSON files will be stored.')

    parser.add_argument(
      '--keep-empty',
      action='store_true',
      default=False,
      help='keep empty tissue chunks (with no annotations within)')

    parser.add_argument(
      '--target-side',
      type=int,
      default=1024,
      help='The directory where the input data will be stored.')

    parser.add_argument(
      '--max-area',
      type=float,
      default=1e7,
      help='maximal area of a roi')

    parser.add_argument(
      'fnxml',
      type=str,
      help='The XML file for ROI.')

    parser.add_argument(
      '--all-grid',
      action='store_true',
      default=False,
      help='store all grid patches (by defaut grid patches that overlap features will be removed)')

    parser.add_argument(
      '--keep-levels',
      type=int,
      default=3,
      help='.')

    parser.add_argument(
      '--magnlevel',
      type=int,
      default=0,
      help='.')

    parser.add_argument(
      '--frac-stride',
      type=int,
      default=1,
      help='.')

    prms = parser.parse_args()
    VISUALIZE = False

    lower = [0, 0, 180]
    upper = [179, 10, 255]
    close=50
    open_=30
    filtersize = 20

    fnsvs = re.sub(".xml$", ".svs", prms.fnxml)

    outdir = os.path.join(prms.data_root, "data_{}/fullsplit".format(prms.target_side))

    ## setup
    imgid = get_img_id(fnsvs)

    target_size = [prms.target_side, prms.target_side,]
    #os.makedirs(outdir)

    # ## Read XML ROI, convert, and save as JSON
    fnjson = extract_rois_svs_xml(prms.fnxml, outdir=prms.json_dir,
                                  remove_empty = ~prms.keep_empty,
                                  keeplevels=prms.keep_levels)

    with open(fnjson,'r') as fh:
        roilist = json.load(fh)

    print("ROI type counts")
    print(pd.Series([roi["name"] for roi in roilist]).value_counts())

    # read slide
    slide = openslide.OpenSlide(fnsvs)

    # load the thumbnail image
    img = np.asarray(slide.associated_images["thumbnail"])

    median_color = get_median_color(slide)
    ratio = get_thumbnail_magnification(slide)

    print("full scale slide dimensions: w={}, h={}".format(*slide.dimensions))

    if VISUALIZE:
        from matplotlib import pyplot as plt
        colordict = {'open glom': 'b',
                     'scler glom': 'm',
                     'infl':'r',
                     'tissue':'w',
                     'other tissue':'y',
                     'art':'olive',
                     'fold':'y'}

        #cell#

        plt.figure(figsize = (18,10))
        plt.imshow(img)
        for roi in roilist:
            plot_contour(roi["vertices"]/ratio, c=colordict[roi['name']])

        #cell#
        vert = roilist[19]["vertices"]
        target_size = [1024]*2
        x,y,w,h = cv2.boundingRect(np.asarray(vert).round().astype(int))
        mask, cropped_vertices = get_region_mask(vert, [x,y], (w,h), color=(255,))

        plt.imshow(mask)
        plot_contour(cropped_vertices, c='r')
        print(mask.max())

    #############################
    print("READING TARGETED ROIS", file=sys.stderr)

    imgroiiter = read_roi_patches_from_slide(slide, roilist,
                            target_size = target_size,
                            maxarea = prms.max_area,
                            nchannels=3,
                            allcomponents=True,
                           )

    print("READING AND SAVING SMALLER ROIS (GLOMERULI, INFLAMMATION LOCI ETC.)",
          file=sys.stderr) 

    for reg, rois,_, start_xy in imgroiiter:
        sumdict = summarize_rois_wi_patch(rois, bg_names = ["tissue"])
        prefix = get_prefix(imgid, start_xy, sumdict["name"], sumdict["tissue_id"],
                            sumdict["id"], parentdir=outdir, suffix='-targeted')
        #fn_summary_json = prefix + "-summary.json"
        fn_json = prefix + ".json"
        fnoutpng = prefix + '.png'
        print(fnoutpng)
        os.makedirs(os.path.dirname(fn_json), exist_ok=True)
        
        #with open(fn_summary_json, 'w+') as fhj: json.dump(sumdict, fhj)
        if isinstance(reg, Image.Image):
            reg.save(fnoutpng)
        else:
            Image.fromarray(reg).save(fnoutpng)
        
        rois = add_roi_bytes(rois, reg, lower=lower, upper=upper,
                             close=close,
                             open=open_,
                             filtersize = filtersize)
        with open(fn_json, 'w+') as fhj: json.dump( rois, fhj)

    print("READING AND SAVING _FEATURELESS_ / NORMAL TISSUE", file=sys.stderr)

    magnification = 4**prms.magnlevel
    real_side = prms.target_side * magnification

    for tissue_chunk_iter in get_tissue_rois(slide,
                                            roilist,
                                            vis = False,
                                            step = real_side // prms.frac_stride,
                                            target_size = [real_side]*2,
                                            maxarea = 1e7,
                                            random=False,
                                            normal_only = not prms.all_grid,
                                           ):
            # save
            save_tissue_chunks(tissue_chunk_iter, imgid, parentdir=outdir,
                               close=close,
                               open_=open_,
                               filtersize = filtersize)
