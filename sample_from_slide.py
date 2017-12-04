
# coding: utf-8

from PIL import Image
import json


from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import pandas as pd
import os
import re
import json
import openslide
import cv2
#cell#

from extract_rois_svs_xml import extract_rois_svs_xml
from slideutils import (plot_contour, get_median_color, get_thumbnail_magnification,
        CropRotateRoi, get_img_bbox, get_rotated_highres_roi, get_uniform_tiles,
                       get_contour_centre, read_roi_patches_from_slide,
                       clip_roi_wi_bbox, sample_points_wi_contour)


def rectangle_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return None
    return (x, y, w, h)

def get_img_id(svsname):
    imgid = re.sub("\.svs$","", os.path.basename(svsname)).replace(" ", "_").replace("-","_")
    return imgid

def get_prefix(imgid, name, tissueid, id, parentdir = "data"):
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
    return {"name":name, "id": str(id), "tissue_id": tissue_id, "stats": dfareas.to_dict(orient='records')}


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
                                        nchannels=None,
                                        allcomponents = True,
                                        nomask=True,
                                       )
#         if vis:
#             plt.scatter(points[:,0], points[:,1],c='r')
#             plot_contour(cont)
        # filter for rois with only normal tissue 
        normal_tissue_only_iter = filter(lambda x: all(roi['name']=='tissue' for roi in x[1]), imgroiiter )
        yield normal_tissue_only_iter


def save_tissue_chunks(imgroiiter, imgid):
    for ii, (reg, rois,_) in enumerate(imgroiiter):
        sumdict = summarize_rois_wi_patch(rois, bg_names = [])
        prefix = get_prefix(imgid, sumdict["name"], sumdict["id"], ii)

        fnjson = prefix + ".json"
        fnoutpng = prefix + '.png'
        print(fnoutpng)

        os.makedirs(os.path.dirname(fnjson), exist_ok=True)
        with open(fnjson, 'w+') as fhj: json.dump( sumdict, fhj)
        if isinstance(reg, Image.Image):
            reg.save(fnoutpng)
        else:
            Image.fromarray(reg).save(fnoutpng)


if __name__ == '__main__':
    VISUALIZE = False

    prms = dict(
        target_size=[1024]*2,
        maxarea = 1e7,
        )

    fnxml = "examples/6371/6371 1.xml"
    fnsvs = re.sub(".xml$", ".svs", fnxml)

    imgid = get_img_id(fnsvs)
    outdir = "data1/"

    #os.makedirs(outdir)

    # ## Read XML ROI, convert, and save as JSON
    fnjson = extract_rois_svs_xml(fnxml)

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
    print("reading targeted rois")

    imgroiiter = read_roi_patches_from_slide(slide, roilist,
                            target_size = prms["target_size"],
                            maxarea = prms["maxarea"],
                            nchannels=3,
                            allcomponents=True,
                           )

    print("reading and saving smaller rois (glomeruli, inflammation loci etc.)")

    for reg, rois,_ in imgroiiter:
        sumdict = summarize_rois_wi_patch(rois, bg_names = ["tissue"])
        prefix = get_prefix(imgid, sumdict["name"], sumdict["tissue_id"],
                            sumdict["id"], parentdir=outdir)
        fnjson = prefix + ".json"
        fnoutpng = prefix + '.png'
        print(fnoutpng)
        os.makedirs(os.path.dirname(fnjson), exist_ok=True)
        
        with open(fnjson, 'w+') as fhj: json.dump( sumdict, fhj)
        if isinstance(reg, Image.Image):
            reg.save(fnoutpng)
        else:
            Image.fromarray(reg).save(fnoutpng)


    print("reading and saving _featureless_ / normal tissue")
    for tissue_chunk_iter in get_tissue_rois(slide,
                                            roilist,
                                            vis = False,
                                            step = 1024,
                                            target_size = None,
                                            maxarea = 1e7,
                                            random=False,
                                           ):
            # save
            save_tissue_chunks(tissue_chunk_iter, imgid)
