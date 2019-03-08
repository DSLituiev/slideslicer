
# coding: utf-8
import numpy as np
from itertools import product
from collections import Counter
import pandas as pd
import os
import re
import json
import openslide
from matplotlib import pyplot as plt
import cv2

#cell#

from extract_rois_svs_xml import extract_rois_svs_xml
from slideutils import (plot_contour, get_median_color, get_thumbnail_magnification,
                        CropRotateRoi, get_img_bbox, get_rotated_highres_roi,
                        get_uniform_tiles,
                        get_contour_centre, read_roi_patches_from_slide,
                        clip_roi_wi_bbox, sample_points_wi_contour,
                        remove_outlier_vertices)

#cell#

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

pd.Series([roi["name"] for roi in roilist]).value_counts().index

#cell#

slide = openslide.OpenSlide(fnsvs)
img = np.asarray(slide.associated_images["thumbnail"])

median_color = get_median_color(slide)
ratio = get_thumbnail_magnification(slide)

#cell#

colordict = {'open glom': 'b',
             'scler glom': 'm',
             'infl':'r',
             'tissue':'w',
             'art':'olive',
             'fold':'y'}

#cell#

slide.dimensions

#cell#

plt.figure(figsize = (18,10))
plt.imshow(img)
for roi in roilist:
    plot_contour(roi["vertices"]/ratio, c=colordict[roi['name']])

#cell#

#cell#

# vert = roilist[19]["vertices"]
# target_size = [1024]*2
# x,y,w,h = cv2.boundingRect(np.asarray(vert).round().astype(int))
# mask, cropped_vertices = get_region_mask(vert, [x,y], (w,h), color=(255,))

# plt.imshow(mask)
# plot_contour(cropped_vertices, c='r')
# print(mask.max())

#cell#

def rectangle_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return None
    return (x, y, w, h)

# ## crop and rotate the image and a chunk roi

#cell#

for roi in roilist:
    if roi["name"] == "tissue":
        co = roi["vertices"]/ratio
# select a chunk roi

# create a transformer object
cr = CropRotateRoi(img, co, enlarge=1.0, borderValue=median_color)

# apply transformation
crimg = cr(img, crop=False)
crroi = cr(co)

#cell#

plt.figure(figsize=(15,2))
plt.imshow(crimg)
plt.plot(crroi[:,0], crroi[:,1])

# ## Now we can slice the chunk horizonatally, and stack highres images from the slices
# 
# As chunks are often huge and oriented diagonally, reading the bounding box around the ROI naively would involve reading huge number of pixels most of which are blank anyways. It is often not feasible in terms of RAM.

#cell#

np.ceil(crimg.shape[1]/int(np.ceil(crimg.shape[1]/100)))

#cell#

stepsize = 100
nsteps = int(np.ceil(crimg.shape[1]/stepsize))
print(nsteps)

# ## Read and stitch slices

#cell#

# for step in range(nsteps):
regions = []
reg_rois = []

chunk_height_tr_large = int(np.ceil(crimg.shape[0]*max(ratio)))

tight = False

for step in range(2):
    currentchunk = crimg[:,stepsize*step:stepsize*(step+1)+1,:]
    if tight:
        "contour per se"
        chunk_contours = get_chunk_countours(currentchunk, 
                                   minlen = 10 , color=False,
                                   filtersize=7)
    else:
        "compatible for stitching with other slices"
        chunk_contours = [get_img_bbox(currentchunk)]
    slice_offset = np.r_[stepsize*step,0]
    chunk_contours += slice_offset
    assert len(chunk_contours) ==1
    chunk_contour = chunk_contours[0]
    
    chunk_slice_contour_origcoord = np.linalg.solve(CropRotateRoi._pad_affine_matrix_(cr.affine_matrix), 
                                                    CropRotateRoi._pad_vectors_(chunk_contour).T).T[:,:2]
    
    trnsf, region, chunk_slice_roi_origcoord_mag, rois_within_chunk = get_rotated_highres_roi(slide,
                                                                                   chunk_slice_contour_origcoord,
                                                                                   roilist,
                                                                                   angle=cr.angle
                                                                                   )
    regions.append(region)
    for roi_ in rois_within_chunk:
        roi = roi_.copy()
        roi["vertices"] += slice_offset*ratio
        reg_rois.append(roi)

#cell#

[rr.shape for rr in regions]

#cell#

fig, axs = plt.subplots(1,len(regions), figsize=(16, 8))
for ax,reg in zip(axs, regions):
    ax.imshow(reg)

#cell#

#cell#

reg.shape

#cell#

def _get_uniform_tile_inds_(img, shape):
    """produce indices for splitting an image into tiles of defined shape.
    The tiles are  potentially overlapping."""
    inshape = np.asarray(img.shape[:2])
    shape = np.asarray(shape[:2])
    halfoutshape = (shape/2).astype(int)
#     numrows, numcols = np.ceil(inshape/shape).astype(int)
    numtiles = np.ceil(inshape/shape).astype(int)
    print(numtiles)
    
    start = halfoutshape
    end = inshape - (shape-halfoutshape)
    center_range = [np.linspace(s,e,n).astype(int) for s,e,n in zip(start,end,numtiles)]
    
    start_range_y = [cc - halfoutshape[0] for cc in center_range[0]]
    start_range_x = [cc - halfoutshape[1] for cc in center_range[1]]
    print(start_range_y[0], start_range_x[0])
    tilesinds = []
    for yy, xx, in product(start_range_y, start_range_x):
        tilesinds.append((slice(yy, yy+shape[0]) , slice(xx, xx+shape[1])))
#     return numrows, numcols
    return tilesinds

def get_uniform_tiles(img, shape):
    """split an image into potentially overlapping tiles of defined shape"""
    tilesinds = _get_uniform_tile_inds_(img, shape)
    return np.stack([reg[ind] for ind in tilesinds])

#cell#

#cell#

print(reg.shape)
outshape = (512, 512)
tiles = get_uniform_tiles(reg, outshape)
tiles.shape

#cell#

fig, axs = plt.subplots(9, 10, figsize=(15,5))
for im, ax in zip(tiles, axs.ravel()):
    ax.imshow(im)

#cell#

# plt.imshow(reg[tilesinds[15]])

#cell#

# height = int(img.shape[0] / numrows)
# width = int(img.shape[1] / numcols)
# for row in range(numrows):
#     for col in range(numcols):
#         y0 = row * height
#         y1 = y0 + height
#         x0 = col * width
#         x1 = x0 + width

#cell#

regmerg = np.hstack(*[regions],)
regmerg.shape

#cell#

fig, axs = plt.subplots(1, figsize=(16, 8))

plt.imshow(regmerg)
for roi in reg_rois:
    plot_contour(remove_outlier_vertices(roi["vertices"], regmerg.shape))

#cell#

# np.bincount(mask.ravel())
# regmerg

#cell#

# mask = cv2.drawContours(img, [remove_outlier_vertices(roi["vertices"], regmerg.shape) for roi in reg_rois], -1, (0,255,0), 3)
mask = cv2.fillPoly(np.zeros_like(regmerg[:,:,:3]),
                    pts =[remove_outlier_vertices(roi["vertices"], regmerg.shape) for roi in reg_rois][1:],
                    color=(255,255,255))
plt.imshow(mask)

#cell#

for roi in reg_rois:
    roi=roi.copy()
    roi.pop('vertices')
    print(roi)

#cell#

# mask = cv2.drawContours(img, [remove_outlier_vertices(roi["vertices"], regmerg.shape) for roi in reg_rois], -1, (0,255,0), 3)
mask = cv2.fillPoly(np.zeros_like(regmerg[:,:,:3]),
                    pts =[remove_outlier_vertices(roi["vertices"], regmerg.shape) for roi in reg_rois],
                    color=(255,255,255))
plt.imshow(mask)

#cell#

from targeted_sample_segm_pair import obj_bboxes, obj_bboxes, visualize_bboxes_mask

#cell#

from targeted_sample_segm_pair import (joint_crop_mask_centre,
                                       get_bboxes_from_mask,
                                       convert_box_range_start_to_centre,
                                      get_shifted_bboxes_from_mask)
# (img, mask)

#cell#

mask.shape

#cell#

bboxiter = get_shifted_bboxes_from_mask((mask>0)[:,:,0],
                shift_range=0.2, classes=(1,))

#cell#

list(bboxiter)

#cell#

img.shape, mask.shape

#cell#

img_arr, msk_arr = joint_crop_mask_centre(regmerg[:,:,:3], 
                                          (mask>0)[:,:,0].astype(int),
                                          target_size=1024,
                                          nshifts=2,
                                         shift_range=0.5)
msk_arr.shape

#cell#

[m_.mean() for m_ in msk_arr]

#cell#

# plt.imshow(img_arr[0])

#cell#

img_arr.shape

#cell#

fig, axs = plt.subplots(2,len(msk_arr), figsize=(14,5))
for ax, m_ in zip(axs[0,:], msk_arr):
    ax.imshow(m_, vmin=0, vmax=4)
    print(m_.mean())
    
for ax, m_ in zip(axs[1,:], img_arr):
    ax.imshow(m_, )
    print(m_.mean()) 

#cell#

bboxes = obj_bboxes(mask[:,:,0].astype('uint8'))
bboxes

#cell#

convert_box_range_start_to_centre(bboxes[0])

#cell#

# cbboxes = get_central_bboxes_from_mask(mask, classes=None)

#cell#

# list(cbboxes)

#cell#

# visualize_bboxes_mask(mask, bboxes)
# pass

#cell#

# def visualize_bboxes_mask(mask, bboxes):
#     fig, ax = plt.subplots(1)
#     ax.imshow(mask)
#     for nn, bbox in enumerate(bboxes):
#         x,y,w,h = bbox 
#         ax.add_patch(
#             plt.Rectangle(
#                 np.r_[x,y].astype(float),
#                 w,
#                 h,
#                 fill=False,      # remove background
#                 edgecolor='r',
#                 lw=3.0,
#             )
#         )
#     plt.axis('tight')
#     return fig

#cell#

# # Draw detected blobs as red circles.
# # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob

# # Show keypoints
# # cv2.imshow("Keypoints", im_with_keypoints)
# plt.imshow(im_with_keypoints)

#cell#

np.log10(mask.sum()/4)

#cell#

['open glom', 'scler glom', 'infl', 'art', 'fold']

#cell#

plt.imshow(img)
plot_contour( chunk_slice_contour_origcoord )

#cell#

plt.imshow(currentchunk)
plot_contour(chunk_contour-slice_offset)

#cell#

# plot_contour(chunk_slice_contour_origcoord)

#cell#
