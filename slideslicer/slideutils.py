
# coding: utf-8
import sys
import numpy as np
from collections import Counter
from itertools import product
from copy import deepcopy

import pandas as pd
import re
import json
import openslide
from PIL import Image, ImageDraw
import cv2
import numpy as np
from warnings import warn
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, LineString
from shapely.geometry import Point, MultiPoint, asMultiPoint, box
from shapely.affinity import rotate
from .geom_tools import get_contour_centre
from .geom_tools import resolve_selfintersection

# move to parse_leica_xml.py 
def get_ellipse_points(verticeslist, num=200):
    a,b = tuple(abs(verticeslist[1] - verticeslist[0])/2)
    xc, yc = tuple(abs(verticeslist[1] + verticeslist[0])/2)
    tt = np.arange(0,2*np.pi, 2*np.pi/num)
    x = xc + a*np.cos(tt)
    y = yc + b* np.sin(tt)
    return np.c_[x,y].tolist()
#     return ((x),list(y))

# delete 
def get_vertices(region):
    "convert xml structured vertices to list of lists"
    verticeslist = [cc for cc in region.vertices.children if cc!='\n']
    verticeslist = [(vv.get('x'), vv.get('y')) for vv in verticeslist]
    verticeslist = [(float(x), float(y)) for x,y in verticeslist]
#     verticeslist = np.asarray(verticeslist)
    if region["type"] == '2':
        verticeslist = get_ellipse_points(np.asarray(verticeslist), num=200)
    return verticeslist

# ## functions for rotating, slicing, and stitching the picture

# delete ? 
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
    xmin_hr = int(round(xminf*slide_dimensions[0]))
    ymax_hr = int(round(ymaxf*slide_dimensions[1]))
    xmax_hr = int(round(xmaxf*slide_dimensions[0]))
    
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


def get_median_color(slide):
    return np.apply_over_axes(np.median, 
                              np.asarray(slide.associated_images["thumbnail"]),
                              [0,1]).ravel()


def get_threshold_tissue_mask(img, color=False, filtersize=7,
                   lower = [0, 0, 180],
                   upper = [179, 20, 255],
                     close=True,
                     open=False,
                     dtype='uint8'):
    """Returns masks of tissue chunks by
    1) thresholding in grayscale or color space and 
    2) morphological open/close operations
    """
    filtersize = filtersize if filtersize % 2 == 1 else filtersize+1
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

    if open:
        if isinstance(open, int) and not isinstance(open, bool):
            fsopen = open
        else:
            fsopen = filtersize
        kernel = np.zeros((fsopen,fsopen), np.uint8)
        kernel = cv2.circle(kernel, (fsopen//2, fsopen//2),
                            fsopen//2, 1, thickness=-1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if close:
        if isinstance(close, int) and not isinstance(close, bool):
            fsclose = close
        else:
            fsclose = filtersize
        kernel = np.zeros((fsclose, fsclose), np.uint8)
        kernel = cv2.circle(kernel, (fsclose//2, fsclose//2),
                            fsclose//2, 1, thickness=-1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = (~mask.astype(bool))
    if dtype not in (bool, 'bool'):
        mask = mask.astype('uint8')
    return mask

# rename to : construct_dict_from_verts
def get_roi_dict(contour, name='tissue', id=0, sq_micron_per_pixel=None):
    """input: 
        contour: numpy array
    """
    cdict = {'id':id, 
            'name': name,
            'vertices':contour.tolist(),
            'area': cv2.contourArea(np.asarray(contour, dtype='int32'))
           }
    if sq_micron_per_pixel:
        cdict['areamicrons'] = cdict['area'] * sq_micron_per_pixel
    return cdict


def convert_contour2mask(roi, width, height, fill=1, shape='polygon', radius=3, order = 'C'):
    """
    convert_contour2mask(roi, width, height, fill=1, shape='polygon', radius=3, order = 'C')
    """
    img = Image.new('L', (width, height), 0)
    if len(roi)>1 and shape=='polygon':# roi.shape[0]>1:
        roi = [tuple(x) for x in roi]
        ImageDraw.Draw(img).polygon(roi, outline=fill, fill=fill)
        mask = np.asarray(img, dtype='uint8', order=order)
    else:
        ImageDraw.Draw(img).point(roi, fill=fill)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        thr = np.exp(-0.5)*img.getextrema()[1]
        mask = fill*np.asarray(np.asarray(img,)>thr, dtype='uint8', order=order)
    return mask


# merge with convert_contour2mask
def get_region_mask(vertices, start_xy=0, size_xy=0, color=(1)):
    vertices = shift_vertices(vertices, start_xy, size_xy)
    mask = cv2.fillPoly(np.zeros(size_xy[::-1]),
                        pts =[ vertices ],
                        color=color)
    return mask, vertices



def convert_mask2contour(mask, minlen = 50):
    mask = mask.astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if minlen is not None:
        contours = [np.squeeze(x) for x in contours if x.shape[0]>minlen]
    return contours


def get_chunk_countours(img, color=False, filtersize=7, minlen = 100):
    mask = get_threshold_tissue_mask(img, color=color, filtersize=filtersize)
    contours = convert_mask2contour(mask, minlen = minlen)
    return contours


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


class CropRotateRoi():
    def __init__(self, co, enlarge=1.00,
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
        #if isinstance(img, np.ndarray):
        #    height = img.shape[0]
        #    width = img.shape[1]
        #elif isinstance(img, Image.Image):
        #    width = img.size[0]
        #    height = img.size[1]
        #elif isinstance(img, (tuple,list)):
        #    width, height = img
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

    @property
    def full_affine_matrix(self):
        return CropRotateRoi._pad_affine_matrix_( self.affine_matrix )


    @property
    def full_rotation_matrix(self):
        return CropRotateRoi._pad_affine_matrix_( self.rotation_matrix )

    @classmethod
    def get_rotation_matrix(self, co, angle=None):
        center, sz, angle_ = cv2.minAreaRect(co)
        if angle is not None:
            angle_ = angle
        
        box = cv2.boxPoints((center, sz, angle_))
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

def within_roi(vertices, roi_start, roi_size,):
    left = (vertices>=roi_start).all(0).all()
    right = (vertices<=(roi_start + roi_size)).all(0).all()
    return left and right

def transform_roi_to_rotated_chunk(transform_mag, rr, roi_start, roi_size, 
                                   final_size=None):
    vertices = np.asarray(rr["vertices"])
#     print(rr["name"], left, right)
    if within_roi(vertices, roi_start, roi_size,):
        roi = rr.copy()
        centroid = get_contour_centre(roi["vertices"])
        roi["centroid_within_slice"] = within_roi(centroid, roi_start, roi_size,)
        roi["vertices"] = transform_mag.apply_roi(np.asarray(rr["vertices"]), use_offset=True)
        if (roi["vertices"]<0).all(0).any():
            return None
        if (final_size is not None) and (roi["vertices"]>final_size).all(0).any():
            return None
        else:
            return roi


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
                                            roi_start*ratio, roi_size*ratio,
                                            final_size = (region_.shape[1], region_.shape[0]))
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

def get_img_bbox(img):
    h = img.shape[0]
    w = img.shape[1]
    return np.c_[[0,0],[w,0],[w,h], [0,h]].T

def _get_uniform_tile_inds_(inshape, shape):
    """produce indices for splitting an image into tiles of defined shape.
    The tiles are  potentially overlapping."""
    inshape = np.asarray(inshape[:2])
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
    return tilesinds, numtiles

def get_uniform_tiles(img, shape):
    """split an image into potentially overlapping tiles of defined shape.
    Given   an image of shape     [H x W x C], 
    Returns an array of shape [N x H x W x C]
    Where:

        N = nrows x ncols
        nrows = ceil(img_shape[1] / shape[1])
        ncols = ceil(img_shape[0] / shape[0])

    """
    tilesinds, numtiles = _get_uniform_tile_inds_(img.shape[:2], shape)
    return np.stack([img[ind] for ind in tilesinds])

def read_roi_patches_from_slide(slide, roilist,
                        and_list = [],
                        but_list = [],
                        excludenames = [],
                        target_size = [1024]*2,
                        maxarea = 1e7,
                        color=1,
                        magnlevel = 0,
                        nchannels=None,
                        allcomponents = False,
                        nomask=False,
                        verbose=False,
                       ):
    """
    Input:
    + slide        -- openslide object
    + roilist      -- list of rois
    + and_list     -- list of rois to add when producing patch mask
    + but_list     -- list of rois to include when producing patch mask instead of rois from `roilist`
    + target_size  -- size of the patches (y, x)
    + maxarea      -- maximal area to remove too big rois
    + color        -- (int, tuple(int)) color to fill in the mask
    + nchannels    -- max number of channels (set to 3 to remove 4' transparancy channel)
    
    Yields (iterator):
    
    + img_arr
    + mask_arr
    + roi_cropped_list 
    
    """

    if and_list:
        roilist = deepcopy(roilist) + deepcopy(and_list)
        checklist = roilist
    elif but_list:
        checklist = deepcopy(but_list)
    else:
        checklist = deepcopy(roilist)

    if allcomponents:
        for roi in checklist:
            roi['bbox'] = cv2.boundingRect(np.asarray(roi["vertices"]).round().astype(int))
            
    magnification = 4**-magnlevel
    size_xy = (target_size[1],target_size[0])
    size_xy_magn = (int(target_size[1] * magnification), int(target_size[0]*magnification))
    slide_w, slide_h = slide.dimensions
    for roi in roilist:
        if maxarea is not None and (roi['area'] > maxarea):
            continue
        xc, yc = get_contour_centre(roi["vertices"])
        x = min(slide_w - target_size[1], max(0, xc - target_size[1]//2))
        y = min(slide_h - target_size[0], max(0, yc - target_size[0]//2))
        start_xy = (x,y)
        reg = slide.read_region(start_xy, magnlevel, size_xy_magn)
        if nchannels is not None:
            reg = np.asarray(reg)[:,:,:nchannels]
        # Mask and main roi vertices
        if not nomask or not allcomponents:
            msk, vert = get_region_mask(roi["vertices"],
                                        start_xy, size_xy, 
                                        color=color)
        else:
            msk = None
        if allcomponents:
            bbox = start_xy + size_xy
            sublist = []
            for roi in checklist:
                #print("clipping", roi['id'], roi['name'], 'bbox', roi["bbox"])
                #print(len(roi["vertices"]))
                vert = clip_roi_wi_bbox(bbox,
                                        roi["vertices"],
                                        roi["bbox"]) 
                if vert is not None:
                    if len(vert) ==0:
                        msg = 'no vertices found; skipping; roi name: %s, id: %s' %(roi['name'], roi['id'])
                        if verbose:
                            print(msg, file=sys.stderr, sep='\t')
                        continue
                    if verbose:
                        print('adding', roi['name'], roi['id'], file=sys.stderr, sep='\t')
                    vert = (np.asarray(vert) * magnification).astype(int)
                    area = cv2.contourArea(vert)
                    if area>0.0:
                        roi = deepcopy(roi)
                        roi["areafraction"] = area / roi["area"]
                        roi["area"] = area
                        roi["vertices"] = vert
                        roi.pop("areamicrons")
                        sublist.append(roi)
            #roi_cropped_list.append(sublist)
        else:
            roi = deepcopy(roi)
            vert = (np.asarray(vert) * magnification).astype(int)
            roi["vertices"] = vert
            sublist = [roi]
            #roi_cropped_list.append(roi)
        yield reg, sublist, msk, start_xy


def remove_outlier_vertices(vertices, size_xy, verbose=True):
    size_xy = np.asarray(size_xy[:2])
    rectangle = Polygon(np.asarray([[0,0],
                                    [ size_xy[0], 0],
                                    size_xy,
                                    [0, size_xy[1]]
                                    ]))
    vertices = vertices.copy()
    roi_polygon = Polygon(np.asarray(vertices))
    roi_polygon = resolve_selfintersection(roi_polygon)
    intersection_ = rectangle.intersection(roi_polygon)
    
    if not isinstance(intersection_, Polygon):
        if len(intersection_) == 0:
            return []
        warn("multiple pieces after intersection; using the largest piece")
        intersection_ = intersection_[np.argmax([x.area for x in intersection_])]
    if isinstance(intersection_.boundary, MultiLineString):
        warn("multiple boundary segments after intersection {}; using the largest segment"
             .format(str([len(x.coords) for x in intersection_.boundary])))
        #print(intersection_.boundary[0])
        boundary = intersection_.boundary[np.argmax([len(x.coords) for x in intersection_.boundary])]
    else:
        boundary = intersection_.boundary
    if intersection_.area==0.0:
        return []
    return np.asarray(boundary).astype(int)
#    #vertices[vertices<0] = -1
#    mask = (vertices>=0).all(1)
#    vertices = vertices[mask, :]
#    if vertices.shape[1] < 2:
#        return []
#    shape_yx = np.flipud(shape)
#    for nn in range(len(shape)):
#        mask = vertices[:,nn]>shape_yx[nn]
#        #vertices[vertices[:,nn]>shape_yx[nn], nn] = shape_yx[nn]
#        vertices = vertices[~mask, :]
#    return vertices.astype(int)


def shift_vertices(vertices, start_xy, size_xy):
    """shift and crop vertices to a new image patch"""
    shifted_verices = vertices - np.asarray(start_xy)
    cropped_vertices = remove_outlier_vertices(shifted_verices, 
                                               size_xy)
    return cropped_vertices


def clip_roi_wi_bbox(patch_bbox, other_roi, other_bbox=None):
    """checks wether roi is within a bounding box and returns clipped coordinates
    within the new patch if it falls in it, otherwise returns None
    """
    other_roi = np.asarray(other_roi)

    if other_bbox is None:
        other_bbox = cv2.boundingRect(other_roi.round().astype(int))

    if rectangle_intersection(patch_bbox, other_bbox) is not None:
        px, py, pw, ph = patch_bbox
        other_roi = shift_vertices(other_roi, [px, py], [pw, ph])
        return other_roi
    else:
        return None


def rectangle_intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return None
    return (x, y, w, h)



def plot_contour(roi, ax=None, name = None, fontsize=12, **kwargs):
    from matplotlib import pyplot as plt
    roi = np.asarray(roi)
    if ax is None:
        ax = plt
    line = ax.plot(roi[:,0], roi[:,1], **kwargs)[0]
    if name is not None:
        c = line.get_color()
        ind = np.argmax(roi[:,0])
        ax.text(roi[ind,0], roi[ind,1], name, fontsize=fontsize, color=c)
    return line


class RectangleCornerWH(Polygon):
    def __init__(self, x0, y0, w, h):
        x1 = x0 + w
        y1 = y0 + h
        super(RectangleCornerWH, self).__init__(
                [(x0, y0),
                 (x0, y1),
                 (x1, y1),
                 (x1, y0),
                ]
        )
        return

class CentredRectangle(Polygon):
    def __init__(self, xc, yc, w, h):
        halfw = w/2
        halfh = h/2
        super(CentredRectangle, self).__init__(
                [(xc - halfw, yc - halfh),
                 (xc - halfw, yc + halfh),
                 (xc + halfw, yc + halfh),
                 (xc + halfw, yc - halfh),
                ]
        )
        return

def simulate_patch_sampling(points, size, n=None):
    '''returns rectangles of given size sampled with centres at given by points'''
    if not isinstance(points, MultiPoint):
        points = MultiPoint(asMultiPoint(points))
    if isinstance(size, int):
        w = h = size
    else:
        w, h = size
    return MultiPolygon([CentredRectangle(*np.asarray(p.centroid).tolist(), w,h) \
                         for p in points[:n]])


def sample_grid(w,h, n_points=None, spacing=None, angle=None):
    if (n_points is None) and (spacing is None):
        raise ValueError('either `spacing` or `n_points` must be specified')
    ar = w/h
    
    if spacing is not None:
        w_ = h_ = spacing
#         n_w = int(np.round(w/dw))
#         n_h = int(np.round(h/dh))
#         if angle:
#             dw *= np.cos(angle)
#         if angle:
#             fh =  -np.sin(angle)
#             fw = (np.cos(angle))
#         else:
#             fh = 1.0
# #             fw = 1.0
        
#         print('fh', fh)
#         print('h_/fh', h_/fh)
#         print('fw', fw)
#         print('w_/fw', w_/fw)
        xs = np.arange(0, w, int(w_))
        ys = np.arange(0, h, int(h_))
        n_w = len(xs)
        n_h = len(ys)
    elif (n_points is not None):
        n_w = int(np.round(np.sqrt( n_points * ar)))
        n_h = int(np.round(np.sqrt( n_points/ ar) ))
        #print(n_w, n_h)
        xs = np.linspace(0, w, n_w)
        ys = np.linspace(0, h, n_h)
        w_ = w/n_w
        h_ = h/n_h
        
    X, Y = np.meshgrid(xs, ys)
    
    if angle:
        rot = np.asarray([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])
        dw =  w_ *(1-np.cos(angle))
        dh =  w_ * np.sin(angle)
        X =  np.mod(dw * np.arange(n_h), w_)[..., np.newaxis,] + X
    points = np.c_[X.ravel(), Y.ravel()]
    return points


def intersect_contour_points(contour, points):
    '''select points within a contour'''
    contour = np.asarray(contour, dtype=int)
    # binary mask for clipping
    mask = np.asarray(
            [cv2.pointPolygonTest(contour, tuple(pp), False) for pp in points])>0
    # clip
    points = points[mask]
    return points


def _sample_points_(cnt, n_points=None, spacing=None, 
                  mode='uniform_random', random_seed=None):
    '''sample point within an oblong contour with shapely
    [deprecated]
    '''
    if (n_points is None) and (spacing is None):
        raise ValueError('either `spacing` or `n_points` must be specified')
    
    if isinstance(cnt, Polygon):
        pg = cnt
    else:
        pg = Polygon(cnt)

    center, sz, angle_ = cv2.minAreaRect(np.asarray(pg.boundary, dtype=int))
    w, h = sz
    
    if n_points is None:
        n_points = h*w/spacing**2 
    
    pg_rot = rotate(pg, -angle_)
    x0, y0, x1, y1 = pg_rot.bounds
    
    # calculate area ratio and increment number of sampled points accordingly
    area_ratio = pg_rot.area/Polygon([(x0,y0), (x0,y1), (x1,y1), (x1, y0)]).area
    n_points = int(np.ceil(n_points/area_ratio))
    
    if mode == 'grid':
        points_rot = sample_grid(w,h, n_points=n_points, spacing=spacing, angle=angle_)
        points_rot = points_rot + np.r_[x0, y0]
    elif mode == 'rotated_grid':
        points_rot = sample_grid(w,h, n_points=n_points, spacing=spacing)
        points_rot = points_rot + np.r_[x0, y0]
    elif mode == 'uniform_random':
        np.random.seed(random_seed)
        points_rot = np.random.rand(n_points, 2,) * np.r_[w, h] + np.r_[x0, y0]
    else:
        raise ValueError('unknown mode:%s' % mode)
        
    points_rot = MultiPoint(asMultiPoint(points_rot))
    points_rot = points_rot.intersection(pg_rot)
    points = rotate(points_rot, angle_)
    return points


def sample_points(contour,
                  n_points=None,
                  spacing=200,
                  shift=0,
                  mode='grid',
                  random_seed=None):
    """
    sample points within a roi

    Inputs:
    contour
    n_points
    spacing
    shift
    random: generates random uniform sample; otherwise grid
    """
    if (n_points is None) and (spacing is None):
        raise ValueError('either `spacing` or `n_points` must be specified')
    # get bounding box of the roi
    if isinstance(contour, Polygon):
        contour = contour.boundary
    contour = np.asarray(contour).astype('int32')
    x0, y0, w, h = cv2.boundingRect(contour)
    dimensions = np.r_[w, h]

    if n_points is not None:
        # calculate ratio of contour and bounding box areas 
        # and increment number of sampled points accordingly
        contour_area = cv2.contourArea(contour)
        area_ratio = contour_area/(w*h)
        n_points = int(np.ceil(n_points/area_ratio))
        #ar = w/h
        #n_w = int(np.round(np.sqrt( n_points * ar)))
        #n_h = int(np.round(np.sqrt( n_points/ ar) ))
        spacing = int(np.round(np.sqrt(h*w / n_points)))

    # sample
    if mode == 'grid':
        x_ = np.arange(shift, w, spacing)
        y_ = np.arange(shift, h, spacing)
        x_,y_ = np.meshgrid(x_,y_)
        points = np.vstack([x_.ravel(), y_.ravel()]).T
        # shift to the start of bbox
        points += np.r_[x0, y0]
    elif mode == 'rotated_grid':
        # find tightest rotated bounding box
        center, sz, angle_ = cv2.minAreaRect(contour.astype(int))
        wrot, hrot = sz
        rot_matr = cv2.getRotationMatrix2D(center, angle_,1)
        inv_rot_matr = cv2.getRotationMatrix2D(center, -angle_,1)
        # rotate the contour
        contour_rot = cv2.transform(np.asarray([contour]), rot_matr)[0]
        x0, y0, x1, y1 = cv2.boundingRect(contour_rot.astype(int))
        contour_rot = (contour_rot - np.r_[x0, y0])
        # sample points within the rotated rectange
        points_rot = sample_grid(wrot, hrot, spacing=spacing)
        points_rot = np.r_[x0, y0] + intersect_contour_points(contour_rot.astype(int), points_rot)
        # rotate back
        points = cv2.transform(np.asarray([points_rot]), inv_rot_matr)[0]
    elif mode == 'uniform_random':
        # find tightest rotated bounding box
        center, sz, angle_ = cv2.minAreaRect(contour.astype(int))
        wrot, hrot = sz
        rot_matr = cv2.getRotationMatrix2D(center, angle_,1)
        inv_rot_matr = cv2.getRotationMatrix2D(center, -angle_,1)
        # rotate the contour
        contour_rot = cv2.transform(np.asarray([contour]), rot_matr)[0]
        x0, y0, x1, y1 = cv2.boundingRect(contour_rot.astype(int))
        contour_rot = (contour_rot - np.r_[x0, y0])
        # sample points within the rotated rectange
        if random_seed is not None:
            np.random.seed(random_seed)
        if n_points is None:
            n_points = np.prod(sz) / (spacing**2)
        points_rot = (np.asarray(sz)*np.random.rand(*(int(n_points), 2))).astype(int)
        points_rot = np.r_[x0, y0] + intersect_contour_points(contour_rot.astype(int), points_rot)
        # rotate back
        points = cv2.transform(np.asarray([points_rot]), inv_rot_matr)[0]
    else:
        raise ValueError('unknown mode:%s' % mode)

    # binary mask for clipping
    mask = np.asarray(
            [cv2.pointPolygonTest(contour, tuple(pp), False) for pp in points])>0
    # clip
    points = points[mask]
    return points
