
# coding: utf-8

#cell#

from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import pandas as pd
import re
import json
# import pyaml

#cell#

# see https://github.com/bgilbert/anonymize-slide

#cell#

#cell#

def get_ellipse_points(verticeslist, num=200):
    a,b = tuple(abs(verticeslist[1] - verticeslist[0])/2)
    xc, yc = tuple(abs(verticeslist[1] + verticeslist[0])/2)
    tt = np.arange(0,2*np.pi, 2*np.pi/num)
    x = xc + a*np.cos(tt)
    y = yc + b* np.sin(tt)
    return np.c_[x,y].tolist()
#     return ((x),list(y))

#cell#

def get_vertices(region):
    verticeslist = [cc for cc in region.vertices.children if cc!='\n']
    verticeslist = [(vv.get('x'), vv.get('y')) for vv in verticeslist]
    verticeslist = [(float(x), float(y)) for x,y in verticeslist]
#     verticeslist = np.asarray(verticeslist)
    if region["type"] == '2':
        verticeslist = get_ellipse_points(np.asarray(verticeslist), num=200)
    return verticeslist

# ## functions for rotating, slicing, and stitching the picture

#cell#

import openslide
from PIL import Image, ImageDraw
import cv2
import numpy as np

#cell#

def convert_contour2mask(roi, width, height, fill=1, shape='polygon', radius=3):
    img = Image.new('L', (width, height), 0)
    if len(roi)>1 and shape=='polygon':# roi.shape[0]>1:
        roi = [tuple(x) for x in roi]
        ImageDraw.Draw(img).polygon(roi, outline=fill, fill=fill)
        mask = np.asarray(img, dtype='uint8')
    else:
        ImageDraw.Draw(img).point(roi, fill=255)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        thr = np.exp(-0.5)*img.getextrema()[1]
        mask = fill*np.asarray(np.asarray(img,)>thr, dtype='uint8')
    return mask

#cell#

def map_countour_bbox(contour, slide_dimensions,
                     SUBSAMPLE_RATE=8):
    
    xmin_macro, ymin_macro = np.min(contour, axis=0)
    xmax_macro, ymax_macro = np.max(contour, axis=0)
    
    xminf, xmaxf = 1.0*xmin_macro/macro.shape[1], 1.0*xmax_macro/macro.shape[1]
    yminf, ymaxf = 1.0*ymin_macro/macro.shape[0], 1.0*ymax_macro/macro.shape[0]
    xtotf = xmaxf - xminf
    ytotf = ymaxf - yminf
    print("xtotf", xtotf)
    print("ytotf", ytotf)

    # xminf*slide.dimensions[1]
    ymin_hr = int(round(yminf*slide_dimensions[1]))
    xmin_hr =  int(round(xminf*slide_dimensions[0]))
    ymax_hr = int(round(ymaxf*slide_dimensions[1]))
    xmax_hr =  int(round(xmaxf*slide_dimensions[0]))
    
    ytot_hr = ymax_hr - ymin_hr
    xtot_hr = xmax_hr - xmin_hr
    print(xtot_hr, ytot_hr)
    
    
    loc_hr = (xmin_hr, ymin_hr)
    size_hr = (xtot_hr, ytot_hr)
    size_hr = [SUBSAMPLE_RATE*(ss//SUBSAMPLE_RATE) for ss in size_hr]
    level_hr = 0
    
    img_hr = slide.read_region(loc_hr, level_hr, size_hr)
    target_size = [s//SUBSAMPLE_RATE for s in img_hr.size]
    print("target_size", target_size)

    img_hr.thumbnail(target_size, Image.ANTIALIAS)
    return img_hr

#cell#

def get_median_color(slide):
    return np.apply_over_axes(np.median, 
                              np.asarray(slide.associated_images["thumbnail"]),
                              [0,1]).ravel()

#cell#

def get_threshold_tissue_mask(img, color=False, filtersize=7,
                   lower = [0, 0, 180],
                   upper = [179, 20, 255]):
    
    kernel = (filtersize,filtersize)
    if color:
        # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
        imghsv = cv2.cvtColor(img,  cv2.COLOR_BGR2HSV)
        imghsv[:,:,-1] = cv2.GaussianBlur(imghsv[:,:,-1],kernel,0)
        imghsv[:,:,1] = cv2.GaussianBlur(imghsv[:,:,1],kernel,0)
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        mask = cv2.inRange(imghsv, lower, upper)
    else:
        imgavg = np.mean(img, axis=-1).astype('uint8')
        blur = cv2.GaussianBlur(imgavg,kernel,0)
        ret3, mask = cv2.threshold(blur,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return (~mask.astype(bool)).astype('uint8')

def convert_mask2contour(mask, minlen = 50):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if minlen is not None:
        contours = [np.squeeze(x) for x in contours if x.shape[0]>minlen]
    return contours

def get_chunk_countours(img, color=False, filtersize=7, minlen = 100):
    mask = get_threshold_tissue_mask(img, color=color, filtersize=filtersize)
    contours = convert_mask2contour(mask, minlen = minlen)
    return contours

#cell#

def get_thumbnail_magnification(slide):
    """get ratio of magnified / thumbnail dimension
    assumes no isotropic scaling (indeed it is slightly anisotropic)"""
    ratio = np.asarray(slide.dimensions) / np.asarray(slide.associated_images["thumbnail"].size)
     # np.sqrt(np.prod(ratio))
    return ratio

def roi_loc(roi):
    xmin, ymin = roi.min(0)
    xmax, ymax = roi.max(0)
    return np.r_[xmin, ymin], np.r_[xmax-xmin, ymax-ymin]

#cell#

def get_contour_centre(vertices):
    mmnts = cv2.moments(np.asarray(vertices, dtype='int32'))
    cX = int(mmnts["m10"] / mmnts["m00"])
    cY = int(mmnts["m01"] / mmnts["m00"])
    return (cX, cY)

#cell#

class CropRotateRoi():
    def __init__(self, img, co, enlarge=1.00,
                 use_offset=None, borderValue=None,
                 rotation_matrix=None, angle=None):
        
        self.borderValue = np.asarray(borderValue,dtype=int) if borderValue is not None else None
        co = np.asarray(co, dtype=int)
        self.use_offset = use_offset
        if use_offset:
            self.offset = co.min(0)
            co -= self.offset
            print("offset", self.offset)
        # calculate the affine transformation
        self.enlarge = enlarge
        if isinstance(img, np.ndarray):
            height = img.shape[0]
            width = img.shape[1]
        elif isinstance(img, Image.Image):
            width = img.size[0]
            height = img.size[1]
        elif isinstance(img, (tuple,list)):
            width, height = img
        #self.img_size = (int(self.enlarge*height),int(self.enlarge*width), )
        if rotation_matrix is None:
            self.rotation_matrix, self.angle = CropRotateRoi.get_rotation_matrix(co, angle=angle)
        else:
            self.rotation_matrix = rotation_matrix
            self.angle = angle
            
        self.affine_matrix = self.rotation_matrix
        # calculate offset
        offset = (self.apply_roi(co, use_offset=False).min(0))#.astype(int)
 
        matrix_full = CropRotateRoi._pad_affine_matrix_(self.rotation_matrix)
        # Combine matrices
        self.transl_matrix = np.eye(3) - np.pad(offset[:,np.newaxis], ((0,1),(2,0)),
                               mode='constant', constant_values=0)
        self.affine_matrix = self.transl_matrix.dot(matrix_full)[:2,:]
        self.img_size = tuple(1+(self.apply_roi(co, use_offset=False).max(0)).astype(int)) #[::-1]

    @classmethod
    def get_rotation_matrix(self, co, angle=None):
        xdim, ydim, angle_ = cv2.minAreaRect(co)
        if angle is not None:
            angle_ = angle
        
        box = cv2.boxPoints((xdim, ydim, angle_)) 
        xmin, ymin = box.min(0)
        xmax, ymax = box.max(0)
        rect_center = ((xmin+xmax)/2,(ymin+ymax)/2)

        return cv2.getRotationMatrix2D(rect_center, angle_, 1.0), angle_
    @classmethod
    def _pad_affine_matrix_(cls, matrix):
        matrix_full = np.pad(matrix, ((0,1),(0,0)),
                       mode='constant', constant_values=0)
        matrix_full[-1,-1] = 1
        return matrix_full
    @classmethod
    def _pad_vectors_(cls, vectors):
        return np.pad(vectors, 
                      ((0,0), (0,1)),
                      mode='constant',
                      constant_values=1)
        
    def apply_img(self, img, crop=True, borderValue=None):
        if borderValue:
            self.borderValue = np.asarray(borderValue,dtype=np.uint8)#.tolist()

        borderMode = cv2.BORDER_TRANSPARENT
#         img_size = tuple(self.maxbound[::-1])
        print("img_size", self.img_size)
        "perform the affine transformation on an image"
        rotated = cv2.warpAffine(np.asarray(img), self.affine_matrix, 
                                 self.img_size, cv2.INTER_CUBIC,
                                 borderMode = borderMode,
                                )
        "fill the void"
        transparentmask = rotated[:,:,3] == 0
        transparentmask = np.broadcast_to(transparentmask[:,:,np.newaxis], rotated.shape)
        rotated = rotated+ self.borderValue.astype('uint8').reshape(1,1,-1) * transparentmask.astype('uint8')
        return rotated
    
    def apply_roi(self, co, use_offset=True):
        "perform the affine transformation on a contour"
        if use_offset and self.use_offset and hasattr(self, 'offset'):
            co -= self.offset
#         pad_contour = np.pad(co, 
#                               ((0,0), (0,1)),
#                               mode='constant',
#                               constant_values=1)
        
        pad_contour = CropRotateRoi._pad_vectors_(co)
        crop_contour = pad_contour.dot(self.affine_matrix.T)
        return crop_contour
    
    def __call__(self, *args, crop=True):
        out = []
        for aa in args:
            if isinstance(aa, Image.Image):
                out.append(self.apply_img(aa, crop=crop))
            elif len(aa.shape)==2 and aa.shape[1]==2:
                out.append(self.apply_roi(aa))
            elif len(aa.shape)>=2 and aa.shape[1]>2:
                out.append(self.apply_img(aa, crop=crop))
            else:
                out.append(None)
        if len(args)==1:
            return out[0]
        else:
            return out

#cell#

def plot_roi(roi, **kwargs):
    roi = np.asarray(roi)
    return plt.plot(roi[:,0], roi[:,1], **kwargs)

#cell#

def within_roi(vertices, roi_start, roi_size,):
    left = (vertices>=roi_start).all(0).all()
    right = (vertices<=(roi_start + roi_size)).all(0).all()
    return left and right

def transform_roi_to_rotated_chunk(transform_mag, rr, roi_start, roi_size,):
    vertices = np.asarray(rr["vertices"])
#     print(rr["name"], left, right)
    if within_roi(vertices, roi_start, roi_size,):
#         print(rr["name"])
        roi = rr.copy()
        centroid = get_contour_centre(roi["vertices"])
        roi["centroid_within_slice"] = within_roi(centroid, roi_start, roi_size,)
        roi["vertices"] = transform_mag.apply_roi(np.asarray(rr["vertices"]), use_offset=True)
        if (roi["vertices"]<0).all(0).any():
            return None
        else:
            return roi

#cell#

def get_rotated_highres_roi(slide, chunkroi_small, 
                    feature_rois = [],
                    color=True, filtersize=35, minlen=500,
                    median_color=None, angle=None
                    ):
    if median_color is None:
        median_color = get_median_color(slide)
    ratio = get_thumbnail_magnification(slide)
    
    roi_start, roi_size = roi_loc(chunkroi_small)
    region = slide.read_region((roi_start*ratio).astype(int), 0, 
                               (roi_size*ratio).astype(int))

    transform_mag = CropRotateRoi(region, chunkroi_small*ratio,
                          angle=angle,
                          use_offset=True,
                          borderValue=median_color)
    region_ = transform_mag(region,
#                             crop=True
                           )
    # transform feature rois:
    rois_within_chunk = []
    for rr in feature_rois:
        rr = transform_roi_to_rotated_chunk(transform_mag, rr, 
                                            roi_start*ratio, roi_size*ratio,)
        if rr is not None:
            print(rr["name"])
            rois_within_chunk.append(rr)
    # re-estimate the contour
    mask_ = get_threshold_tissue_mask(region_, color=color, filtersize=filtersize)
    chunkroi_large_refined = convert_mask2contour(mask_, minlen = minlen)
    assert len(chunkroi_large_refined)>0
    # take the longest contour
    maxidx = np.argmax([len(x) for x in chunkroi_large_refined])
    chunkroi_large_refined = chunkroi_large_refined[maxidx]
    return transform_roi_to_rotated_chunk, region_, chunkroi_large_refined, rois_within_chunk

#cell#

def get_roi_dict(contour, name='tissue', id=0, sq_micron_per_pixel=None):
    """input: 
        contour: numpy array
    """
    cdict = {'id':id, 
            'name': name,
            'vertices':contour.tolist(),
           'area': cv2.contourArea(np.asarray(roi["vertices"], dtype='int32'))
           }
    if sq_micron_per_pixel:
        cdict['areamicrons'] = micron_per_pixel*cdict['area']
    return cdict

# ## Read XML ROI, convert, and save as JSON

#cell#
if __name__ == '__main__':

    fnxml = "examples/6371/6371 1.xml"
    fnjson = re.sub(".xml$", ".json", fnxml)
# fnyaml = re.sub(".xml$", ".yaml", fnxml)
    with open(fnxml) as fh:
        soup = BeautifulSoup(fh, 'lxml')

#cell#

    regions = soup.find_all("region")

#cell#

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
#     dict(name=name, vertices = get_vertices(rr))
        roilist.append(attrs_)

# for an ellipse, 
# 
#    area = $\pi \times r \times R$

#cell#

# ellipses = [{"vertices":get_vertices(rr), "length": rr["length"], "area":rr["area"]} for rr in regions if rr["type"]=='2']
# ell = ellipses[1]

#cell#

    pd.Series([rr["name"] for rr in roilist]).value_counts()

#cell#

    with open(fnjson, 'w+') as fh:
        json.dump(roilist, fh)

#cell#

# with open(fnyaml, 'w+') as fh:
#     pyaml.dump(roilist, fh)

#cell#

    from matplotlib import pyplot as plt
    get_ipython().magic('matplotlib inline')

#cell#

    for rr in roilist:
        plt.plot([x for x,_ in  rr["vertices"]],
                 [y for _,y in  rr["vertices"]])

# ## Extract tissue chunk ROIs

#cell#

    fnsvs = "examples/6371/6371 1.svs"

#cell#

    slide = openslide.OpenSlide(fnsvs)
    img = np.asarray(slide.associated_images["thumbnail"])

    median_color = get_median_color(slide)

#cell#

    slide.associated_images.keys()

#cell#

    for kk,vv in slide.associated_images.items():
        print(kk, vv.size)

#cell#

    plt.imshow(img)

# ## Extract mask and contours

#cell#

    mask = get_threshold_tissue_mask(img, color=False, filtersize=7)
    contours = convert_mask2contour(mask, minlen = 100)

#cell#

    sq_micron_per_pixel = np.median([roi["areamicrons"] / roi["area"] for roi in roilist])

    tissue_roilist = [get_roi_dict(cc, name='tissue', id=nn+len(roilist), sq_micron_per_pixel=sq_micron_per_pixel) 
                          for nn,cc in enumerate(contours)]

#cell#

# Save both contour lists together

    with open(fnjson, 'w+') as fh:
        json.dump(roilist + tissue_roilist, fh)

# ### At this point we are done with basic contour extraction. It is enough for the first script 

#cell#

    plt.imshow(mask)

#cell#

#cell#

    plt.figure(figsize = (18,10))
    plt.imshow(img)
    for co in contours:
        plt.plot(co[:,0], co[:,1])

#cell#

    img.shape

#cell#

# objmask = np.zeros((height,width), np.uint8)
# objmask = cv2.fillPoly(objmask, [co], 1)

#cell#

    height, width, _ = img.shape
# objmask = convert_contour2mask(co, width, height, fill=1, shape='polygon', radius=3)

# # Rotating, slicing, and stitching the picture
# ## crop and rotate the image and a chunk roi

#cell#

# select a chunk roi
    co = contours[2]

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

#cell#

    def get_img_bbox(img):
        h = img.shape[0]
        w = img.shape[1]
        return np.c_[[0,0],[w,0],[w,h], [0,h]].T

# ## Read and stitch slices

#cell#

    ratio = get_thumbnail_magnification(slide)
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

    fig, axs = plt.subplots(1, figsize=(16, 8))

    plt.imshow(np.hstack(*[regions],))
#.shape
    for roi in reg_rois:
        plot_roi(roi["vertices"] )

#cell#

    plt.imshow(img)
    plot_roi( chunk_slice_contour_origcoord )

#cell#

    plt.imshow(currentchunk)
    plot_roi(chunk_contour-slice_offset)

#cell#

# plot_roi(chunk_slice_contour_origcoord)

#cell#

    plt.imshow(region)
    plot_roi(chunk_slice_roi_origcoord_mag)
    for rr in rois_within_chunk:
        plot_roi(rr["vertices"])

#cell#
