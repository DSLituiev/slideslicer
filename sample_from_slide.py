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
from uuid import uuid1
from pycocotools.mask import encode, decode

from extract_rois_svs_xml import extract_rois_svs_xml
from slideutils import (plot_contour, get_median_color, 
                        get_thumbnail_magnification,
                        get_img_bbox, get_rotated_highres_roi,
                        get_uniform_tiles, 
                        get_chunk_masks, 
                        get_roi_mask,
                        get_contours_from_mask,
                        CropRotateRoi,
                       get_contour_centre, read_roi_patches_from_slide,
                       clip_roi_wi_bbox, sample_points_wi_contour)


def get_img_id(svsname):
    imgid = re.sub("\.svs$","", os.path.basename(svsname)).replace(" ", "_").replace("-","_")
    return imgid

def get_prefix(imgid, name, tissueid, id, parentdir = "data", uid=False):
    if uid not in (None, False):
        if uid is True:
            uid = uuid1().hex
        prefix = '{parentdir}/{uid}-{imgid}/{typ}/{uid}-t{tissue}-roi{roiid}-{typ}'.format(**{
                                        "tissue":tissueid,
                                        "parentdir":parentdir,
                                        "imgid":imgid,
                                        "roiid":id,
                                        "typ": (name.replace(" ","_")),
                                        "uid":uid})

    else:
        prefix = "{parentdir}/{typ}/{imgid}-t{tissue}-roi{roiid}-{typ}".format(**{
                                        "tissue":tissueid,
                                        "parentdir":parentdir,
                                        "imgid":imgid,
                                        "roiid":id, 
                                        "typ": (name.replace(" ","_"))})
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
    tissue_id = "+".join(["%s"%tt['id'] for tt in tissue_info])
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
        id = '+'.join(areasum["id"].astype(str).tolist())
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
        ax.set_xlabel("\n".join(["{}: {}".format(rr['id'], rr['name']) for rr in rois if rr['name'] !='tissue']))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        

def get_tissue_rois(slide,
                    roilist,
                    vis = False,
                    step = 1024,
                    target_size = None,
                    maxarea = 1e7,
                    random=False,
                   ):

    target_size = [step]*2

    tissue_rois = [roi for roi in roilist if roi['name']=='tissue']

    for roi in tissue_rois:
        print("id", roi["id"])
        cont = roi["vertices"]
        points = sample_points_wi_contour(cont,
                                      step = step,
                                      shift = -step//2,
                                      random=random)

        pointroilist = [{"vertices":[pp], "area":0} for pp in points]
        
#         img_arr, roi_cropped_list, msk_arr, = \
        imgroiiter = read_roi_patches_from_slide(slide, 
                                        pointroilist,
                                        but_list = roilist,
                                        target_size = target_size,
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
        normal_tissue_only_iter = filter(lambda x: all(roi['name']=='tissue' for roi in x[1]), imgroiiter )
        yield normal_tissue_only_iter


def save_tissue_chunks(imgroiiter, imgid, uid=False, parentdir="data",
                       lower = [0, 0, 180],
                       upper = [179, 10, 255],
                       close=50,
                       open_=30,
                       filtersize = 20,
                       ):
    for ii, (reg, rois,_) in enumerate(imgroiiter):
        sumdict = summarize_rois_wi_patch(rois, bg_names = [])
        prefix = get_prefix(imgid, sumdict["name"], sumdict["id"], ii, uid=uid, parentdir=parentdir)

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
        mask_ = get_roi_mask(roi_["vertices"], reg.shape[1], reg.shape[0], fill=1, order='F')

        cocomask = encode(np.asarray(mask_, dtype='uint8'))
        cocomask["counts"] = cocomask["counts"].decode('utf-8')
        roi_.update(cocomask)
        if isinstance(roi_["vertices"], np.ndarray):
            roi_["vertices"] = roi_["vertices"].tolist()
        other_mask_ = np.maximum(other_mask_, mask_)
    
    for roi_ in [tissue_roi]:
        if reg is not None:
            mask_ = get_chunk_masks(reg, color=True, filtersize=filtersize, dtype=bool,
                                    open=open, close=close,
                                    lower = lower, upper = upper)
            if mask_.sum()==0:
                roi_ = None
                continue
            verts = get_contours_from_mask(mask_.astype('uint8'), minlen=minlen)
            # print("verts", len(verts))
            if len(verts)>0:
                roi_["vertices"] = verts[np.argmax(map(len,verts))]
            else:
                #print("verts", len(verts), roi_["vertices"])
                pass
            mask_ = np.asarray(mask_, order='F')
        else:
            mask_ = get_roi_mask(roi_["vertices"], reg.shape[1], reg.shape[0], 
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
      '--uid',
      action='store_true',
      default=False,
      help='generate uid.')

    parser.add_argument(
      '--keep-levels',
      type=int,
      default=3,
      help='.')

    prms = parser.parse_args()
    VISUALIZE = False

    if prms.uid:
        uid = uuid1().hex
    else:
        uid=False

    lower = [0, 0, 180]
    upper = [179, 10, 255]
    close=50
    open_=30
    filtersize = 20

    #fnxml = "../data/raw/70bb3032750d09e7549928c0dbf79afc30d7cb68.xml"
    #fnxml = sys.argv[1]
    fnsvs = re.sub(".xml$", ".svs", prms.fnxml)

    outdir = os.path.join(prms.data_root, "data_{}/fullsplit".format(prms.target_side))

    ## setup
    imgid = get_img_id(fnsvs)

    target_size = [prms.target_side, prms.target_side,]
    #os.makedirs(outdir)

    # ## Read XML ROI, convert, and save as JSON
    fnjson = extract_rois_svs_xml(prms.fnxml, outdir=prms.json_dir,
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
    print("READING TARGETED ROIS")

    imgroiiter = read_roi_patches_from_slide(slide, roilist,
                            target_size = target_size,
                            maxarea = prms.max_area,
                            nchannels=3,
                            allcomponents=True,
                           )

    print("READING AND SAVING SMALLER ROIS (GLOMERULI, INFLAMMATION LOCI ETC.)")

    for reg, rois,_ in imgroiiter:
        sumdict = summarize_rois_wi_patch(rois, bg_names = ["tissue"])
        prefix = get_prefix(imgid, sumdict["name"], sumdict["tissue_id"],
                            sumdict["id"], parentdir=outdir, uid=uid)
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
        # mask_ = decode(rois[-1]).astype(bool)
        # plt.imshow(mask_ )
        with open(fn_json, 'w+') as fhj: json.dump( rois, fhj)

    print("READING AND SAVING _FEATURELESS_ / NORMAL TISSUE")
    for tissue_chunk_iter in get_tissue_rois(slide,
                                            roilist,
                                            vis = False,
                                            step = 1024,
                                            target_size = None,
                                            maxarea = 1e7,
                                            random=False,
                                           ):
            # save
            save_tissue_chunks(tissue_chunk_iter, imgid, uid=uid, parentdir=outdir,
                               close=close,
                               open_=open_,
                               filtersize = filtersize)
