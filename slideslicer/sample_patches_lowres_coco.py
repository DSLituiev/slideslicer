
# coding: utf-8
import numpy as np
from collections import Counter
import pandas as pd
import os
import re
import json
import yaml
import openslide
import cv2
from shapely.geometry import Polygon, MultiPolygon

from pycocotools.mask import encode, decode
from copy import deepcopy
from slideslicer.extract_rois_svs_xml import extract_rois_svs_xml 
from slideslicer.slideutils import (plot_contour, get_median_color, get_thumbnail_magnification,
                       CropRotateRoi, get_img_bbox, get_rotated_highres_roi, get_uniform_tiles,
                       get_contour_centre, read_roi_patches_from_slide,
                       convert_mask2contour, get_roi_dict,
                       clip_roi_wi_bbox,  convert_contour2mask)
from sample_from_slide import get_tissue_rois


def resolve_selfintersection(pp, areathr=1e-3, fraction=3):
    pbuf = pp.buffer(0)#.interiors
    areadiff = abs(pp.area- pbuf.area)/pp.area
#     print("areadiff", areadiff)
    if (areadiff > areathr):
        pp = _permute_polygon_(pp, fraction=fraction)
        pp = resolve_selfintersection(pp, areathr=1e-3, fraction=3)
    else:
        try:
            pp = Polygon(pbuf)
        except NotImplementedError as ee:
            print(ee)
            ind = np.argmax([x.area for x in pbuf])
            return pbuf[ind]
        
    assert pp.is_valid
    return pp


def _permute_polygon_(pp, fraction=3):
    if fraction>1:
        fraction = 1/fraction
    vertices = np.asarray(pp.boundary)
    pp = Polygon(np.flipud(np.vstack([vertices[int(vertices.shape[0]*fraction):],
                                      vertices[:int(vertices.shape[0]*fraction)]])))
    return pp


def convert_roi_to_coco(roi, image_id, mask_id, img_size=None, rle=False,
                        slide_name=None, category_lookup={}):

    cocoroi = {'segmentation': [np.asarray(roi['vertices']).ravel().astype(int).tolist()], 
               'category_name' : roi['name'],
               'slide_name':slide_name,
              }

    if category_lookup is None or len(category_lookup)==0:
        cocoroi['category_id'] = 'some_tissue'
    else:
        cocoroi['category_id'] = category_lookup[roi['name']]

    polygon = Polygon(roi['vertices'])
    cocoroi['area'] = polygon.area
    cocoroi['bbox'] = polygon.bounds
    cocoroi['iscrowd'] = 0
    # !!!!!!!!!!!
    cocoroi['image_id'] = image_id
    # !!!!!!!!!!!
    cocoroi['id'] = mask_id

    if rle:
        mask_ = convert_contour2mask(roi["vertices"], img_size[0], img_size[1], fill=1, order='F')
        cocomask = encode(np.asarray(mask_, dtype='uint8'))
        cocomask["counts"] =  cocomask["counts"].decode('utf-8')
        for kk,vv in cocomask.items():
            cocoroi[kk] = vv
    return cocoroi


def get_image_coco_json(slide_name, filename, image_id, image_size, location, ):
    return {
    "id": image_id,
    "file_name": filename,
    "width": image_size[0],
    "height": image_size[1],
    "location-x": int(location[0]),
    "location-y": int(location[1]),
    'slide_name': slide_name
    }


def process_patch(rois, start_xy, img_size, image_id=-1,
                  category_lookup=CATEGORY_LOOKUP, rle=False):
    slide_name = os.path.basename(fnsvs)
    slide_id = slide_name.replace('.svs', '')
    filename = '{}-x{}-y{}.png'.format(slide_id, start_xy[0], start_xy[1] )
    img_json = get_image_coco_json(slide_name, filename, image_id, img_size, start_xy, )
    mask_json = []
    for ii, roi in enumerate(rois):
        if roi['name']!='tissue':
            if image_id<0:
                ii *= -1
            mask_json.append(convert_roi_to_coco(roi, image_id, ii, 
                             img_size=img_size,
                             slide_name=slide_name,
                             category_lookup=CATEGORY_LOOKUP,
                             rle=rle)
                            )
    return filename, {'annotations': mask_json, 'images':img_json}

# ## MS COCO format:
# 
#     ['category_id', 'area', 'bbox', 'image_id', 'id', 'iscrowd', 'segmentation']
#     
#     category_id	<'int'>
#     image_id	<'int'>
#     id	<'int'>
#     iscrowd	<'int'>
#     area	<'float'>
#     bbox	<'list'>
#     segmentation	<'list'>
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
      'fnxml',
      type=str,
      help='The XML file for ROI.')

    parser.add_argument(
      '--out-root',
      type=str,
      default='/repos/data/coco/gloms/img_level2_rle/',
      help='The directory where the MSCOCO-formatted data will be stored.')

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
      default=512,
      help='Requested side (in pixels) of the image patches')

    parser.add_argument(
      '--max-area',
      type=float,
      default=1e7,
      help='maximal area of a roi')

    parser.add_argument(
      '--all-grid',
      action='store_true',
      default=False,
      help='''store all grid patches 
      (by defaut grid-sampled patches that overlap ROI/features will be removed)''')

    parser.add_argument(
      '--rle',
      action='store_true',
      default=False,
      help='save run-length encoding')

    parser.add_argument(
      '--keep-levels',
      type=int,
      default=3,
      help='.')

    parser.add_argument(
      '--magnlevel',
      type=int,
      default=2,
      help='.')

    parser.add_argument(
      '--frac-stride',
      type=int,
      default=3,
      help='.')

    parser.add_argument(
      '--open',
      type=int,
      default=30,
      help='morphological open kernel size')

    parser.add_argument(
      '--close',
      type=int,
      default=50,
      help='morphological close kernel size')

    prms = parser.parse_args()
    VISUALIZE = False

    lower = [0, 0, 180]
    upper = [179, 10, 255]
    filtersize = 20

    #category_lookup = {'tissue':1, 'other':2, 'infl':3, 'glom':4, 'scler_glom':5, }
    with open("tissuedict.yaml") as fh:
        CATEGORY_LOOKUP = yaml.load(fh)

    # Read the slide
    fnsvs = re.sub(".xml$", ".svs", prms.fnxml)
    slide = openslide.OpenSlide(fnsvs)

    # extract annotations
    rreader = RoiReader(prms.fnxml, remove_empty=True)
    roilist = rreader.rois
    #fnjson = extract_rois_svs_xml(prms.fnxml, remove_empty=True)
    #with open(fnjson, 'r') as fhj: 
    #    roilist = json.load(fhj)

    #print("ROI type counts")
    #print(pd.Series([roi["name"] for roi in roilist]).value_counts())
    #cell#

    magnification = 4**prms.magnlevel
    target_side_magn = prms.target_side*magnification
    IMGDIR = prms.out_root 
    #"/repos/data/coco/gloms/img_level2/"
    ANNDIR = IMGDIR

    print("magnification: %d" % magnification)
    print("SAVING CHUNKS")
    nrois = 0
    print("RLE:", prms.rle)
    for tissue_chunk_iter in get_tissue_rois(slide,
                                            roilist,
                                            vis = False,
                                            step = int(round(target_side_magn / prms.frac_stride)),
                                            target_size = [target_side_magn] * 2,
                                            magnlevel = prms.magnlevel,
                                            maxarea = 1e7,
                                            random=False,
                                            normal_only = False,
                                            shift_factor = 1,
                                           ):
        print('='*40)
        for ii, (reg, rois, _, start_xy) in enumerate(tissue_chunk_iter):
            print('_'*20)
            print("patch #{}\twith {} rois".format(ii, len(rois)))
            nrois +=1
            filename, json_ = process_patch(rois, start_xy,
                                            img_size=[prms.target_side]*2, image_id=-1,
                                            rle=prms.rle)
            cv2.imwrite(os.path.join(IMGDIR, filename), reg)
            fpath_json = os.path.join(ANNDIR, filename.replace('.png', '.json'))
            with open(fpath_json, 'w') as fh:
                json.dump(json_, fh,)
    if nrois == 0:
        raise ValueError("nothing has been saved")
