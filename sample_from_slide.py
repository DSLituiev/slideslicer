
# coding: utf-8

#cell#

from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import pandas as pd
import os
import re
import json
import openslide
from matplotlib import pyplot as plt
import cv2
get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

#cell#

from extract_rois_svs_xml import extract_rois_svs_xml
from slideutils import (plot_contour, get_median_color, get_thumbnail_magnification,
        CropRotateRoi, get_img_bbox, get_rotated_highres_roi, get_uniform_tiles,
                       get_contour_centre, read_roi_patches_from_slide,
                       clip_roi_wi_bbox, sample_points_wi_contour)

# ## Read XML ROI, convert, and save as JSON

#cell#

fnxml = "examples/6371/6371 1.xml"
fnsvs = re.sub(".xml$", ".svs", fnxml)

#cell#

fnjson = extract_rois_svs_xml(fnxml)

#cell#

with open(fnjson,'r') as fh:
    roilist = json.load(fh)

#cell#

# pd.Series([roi["name"] for roi in roilist]).value_counts().index

#cell#

slide = openslide.OpenSlide(fnsvs)
img = np.asarray(slide.associated_images["thumbnail"])

median_color = get_median_color(slide)
ratio = get_thumbnail_magnification(slide)

#cell#

slide.dimensions

#cell#

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

#cell#

vert = roilist[19]["vertices"]
target_size = [1024]*2
x,y,w,h = cv2.boundingRect(np.asarray(vert).round().astype(int))
mask, cropped_vertices = get_region_mask(vert, [x,y], (w,h), color=(255,))

plt.imshow(mask)
plot_contour(cropped_vertices, c='r')
print(mask.max())

#cell#

print("reading targeted rois")

#cell#

# imgroiiter = read_roi_patches_from_slide(slide, roilist,
#                         target_size = [1024]*2,
#                         maxarea = 1e7,
#                         nchannels=3,
#                         allcomponents=True,
#                        )
# imgroiiter

# nomask = True
# img_arr, roi_cropped_list, msk_arr = [],[],[]
# for img, roi, msk in imgroiiter:
#     img_arr.append(img)
#     msk_arr.append(msk)
#     roi_cropped_list.append(roi)
    
# img_arr = np.stack(img_arr)
# if not nomask:
#     msk_arr = np.stack(msk_arr)

#cell#

def rectangle_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return None
    return (x, y, w, h)

#cell#

# for roi in roilist:
#     roi['bbox'] = cv2.boundingRect(np.asarray(roi["vertices"]).round().astype(int))

#cell#

# rect_list = [roi['bbox'] for roi in roilist]

#cell#

# [roi['id'] for roi in roilist]

#cell#

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

#cell#

imgid = get_img_id(fnsvs)

prefix = get_prefix(imgid, rr["name"], "39", rr["id"])
fnoutpng = prefix + '.png'
fnoutpng

#cell#

# ind = np.argmax([rr["area"] for rr in rois if rr['name'] !='tissue'])
# rois[ind]['name']

#cell#

from PIL import Image
import json

#cell#

for nn in set([rr["name"] for rr in  roilist]):
    nn = nn.replace(" ", "_")
    print(nn)
    os.makedirs(f"data/{nn}", exist_ok=True)

#cell#

def summarize_rois_wi_patch(rois, bg_names = ["tissue"]):
#     ind = np.argmax([rr["area"] for rr in rois if rr['name'] not in ignore_names])
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
#     print("tissue_info", tissue_id)
    
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
#         name = None
        name = '+'.join(areasum.index.tolist())
        id = '+'.join(areasum["id"].astype(str).tolist())
    return {"name":name, "id": str(id), "tissue_id": tissue_id, "stats": dfareas.to_dict(orient='records')}

#cell#

# roi_cropped_list

#cell#

# for reg, rois in zip(img_arr, roi_cropped_list):
#     sumdict = summarize_rois_wi_patch(rois, bg_names = ["tissue"])
# #     print(sumdict)
#     prefix = get_prefix(imgid, sumdict["name"], sumdict["tissue_id"], sumdict["id"])
#     print(prefix)

#cell#

#cell#

imgroiiter = read_roi_patches_from_slide(slide, roilist,
                        target_size = [1024]*2,
                        maxarea = 1e7,
                        nchannels=3,
                        allcomponents=True,
                       )

for reg, rois,_ in imgroiiter:
    sumdict = summarize_rois_wi_patch(rois, bg_names = ["tissue"])
#     name = sumdict["name"]
#     id = sumdict["id"]
    prefix = get_prefix(imgid, sumdict["name"], sumdict["tissue_id"], sumdict["id"])
    fnjson = prefix + ".json"
    os.makedirs(os.path.dirname(fnjson), exist_ok=True)
    
    with open(fnjson, 'w+') as fhj: json.dump( sumdict, fhj)
    fnoutpng = prefix + '.png'
    Image.fromarray(reg).save(fnoutpng)
    print(fnoutpng)

#cell#

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

#cell#

#cell#

#cell#

outdir = "data/"
os.makedirs(outdir)

#cell#

def save_tissue_chunks(imgroiiter, imgid):
    for ii, (reg, rois,_) in enumerate(imgroiiter):
        sumdict = summarize_rois_wi_patch(rois, bg_names = [])
        prefix = get_prefix(imgid, sumdict["name"], sumdict["id"], ii)
        fnjson = prefix + ".json"
        fnoutpng = prefix + '.png'
        print(prefix)
        os.makedirs(os.path.dirname(fnjson), exist_ok=True)
        with open(fnjson, 'w+') as fhj: json.dump( sumdict, fhj)
        if isinstance(reg, Image.Image):
            reg.save(fnoutpng)
        else:
            Image.fromarray(reg).save(fnoutpng)

#cell#

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

#         print(next(normal_tissue_only_iter))
        # save
        save_tissue_chunks(normal_tissue_only_iter, imgid)

#cell#

print("reading targeted rois")

get_tissue_rois(slide,
                    roilist,
                    vis = False,
                    step = 1024,
                    target_size = None,
                    maxarea = 1e7,
                    random=False,
                   )

#cell#

#cell#

# nrows = 5
# nimg = len(roi_cropped_list)
# fig, axs = plt.subplots(nrows, nimg//nrows, figsize=(15,15))
# for ax, reg, rois in zip(axs.ravel(),
#                          img_arr,
#                          roi_cropped_normal_list):
#     ax.imshow(reg)
#     ax.set_xlabel(", ".join(list(set([roi["name"] for roi in rois]))))
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     for roi in rois:
#         plot_contour(roi["vertices"], ax=ax)
# #         print(roi["name"], round(roi["areafraction"],2),roi["area"], sep='\t')

#cell#

#cell#

# nn = 17
# plt.imshow(img_arr[nn])
# for roi in roi_cropped_list[nn]:
#     plot_contour(roi["vertices"])
#     print(roi["name"], round(roi["areafraction"],2),roi["area"], sep='\t')
