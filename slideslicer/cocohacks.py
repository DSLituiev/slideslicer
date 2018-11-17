import numpy as np
import json
from pycocotools.mask import encode, decode
from warnings import warn
from .slideutils import convert_contour2mask


def remove_upper_channel(lo, hi):
    """
    take difference between two channels:
    # rule:
    lo )  0 0 1 1
    up )  0 1 0 1
     ->   0 0 1 0
    """
    lo = lo.astype(bool)
    hi = hi.astype(bool)
    return (lo ^ (lo & hi).astype(bool)).astype(bool)


def convert_cocorle2onehotmask(rois, tissuedict):
    """constructs a dense mask given a list of `rois`
    and a dictionary mapping roi names to channel
    numbers in tissuedict are expected to start at one
    as the default class is constructed
    and assigned to zeroth channel

    Inputs
        rois       : an MS-COCO formated list with RLE 'counts'
        tissuedict : a mapping from roi names to integers, 
            can come in 2 possible formats:
            + dictionary : {'tissue_1': 1, 'tissue_2': 2, ...}
            + list       : ['tissue_1', 'tissue_2', ... ]

    Calls `pycocotools.mask.decode`
    """
    if isinstance(tissuedict, list):
        tissuedict = {xx: ii+1 for ii, xx in enumerate(tissuedict)}

    nchannels = 1+max(tissuedict.values())
    maskarr = np.zeros(rois[-1]["size"] + [nchannels], dtype=bool)
    
    for roi_ in rois:
        mask = decode(roi_)
        name = roi_["name"]
        if name in tissuedict:
            channel = tissuedict[name]
            maskarr[..., channel] |= mask.astype(bool)
 
    for nn in range(maskarr.shape[-1]-2, 0, -1):
        maskarr[..., nn] =  remove_upper_channel(
                                                maskarr[..., nn],
                                                maskarr[...,nn+1:].any(-1)
                                                )
    maskarr[..., 0] = ~maskarr[...,1:].any(-1)

    if not  maskarr.sum(-1).max() == 1:
        print("maskarr.sum(-1).max()", maskarr.sum(-1).max())
        raise ValueError()

    return maskarr


def convert_cocorle2intmask(rois, tissuedict):
    """constructs an interger mask given a list of `rois`
    and a dictionary mapping roi names to channel
    numbers in tissuedict are expected to start at one
    as the default class is constructed
    and assigned to zeroth channel
    
    Inputs
        rois       : an MS-COCO formated list with RLE 'counts'
        tissuedict : a mapping from roi names to integers, 
            can come in 2 possible formats:
            + dictionary : {'tissue_1': 1, 'tissue_2': 2, ...}
            + list       : ['tissue_1', 'tissue_2', ... ]

    Calls `pycocotools.mask.decode`
    """
    if isinstance(tissuedict, list):
        tissuedict = {xx: ii+1 for ii, xx in enumerate(tissuedict)}

    nchannels = 1+max(tissuedict.values())
    maskarr = np.zeros(rois[-1]["size"], dtype=bool)
    
    for roi_ in rois:
        mask = decode(roi_)
        name = roi_["name"]
        if name in tissuedict:
            channel = tissuedict[name]
            maskarr = np.maximum(maskarr, channel*mask.astype(np.uint8))
    return maskarr


def convert_contour2cocorle(verts, w, h, format=None):
    mask = convert_contour2mask(verts, w,h, order='F')[...,np.newaxis]
    entry = encode(mask)[0]
    if format is str:
        entry['counts'] = entry['counts'].decode('ascii')
    return entry


def construct_sparse_mask(*args):
    warn('use convert_vertices2intmask instead of construct_sparse_mask',
        DeprecationWarning)
    return construct_sparse_mask(*args)

def construct_dense_mask(rois, tissuedict):
    warn('use convert_cocorle2onehotmask instead of construct_dense_mask',
        DeprecationWarning)
    return convert_cocorle2onehotmask(rois, tissuedict)

def dense_to_sparse(maskarr):
    return (np.arange(maskarr.shape[-1]).reshape([1,1,-1]) *
            maskarr).sum(-1)


def read_roi_to_sparse(jsonfile, roidict):
    with open(jsonfile) as fh:
        rois = json.load(fh)
    return construct_sparse_mask(rois, roidict)


def read_roi_to_dense(jsonfile, roidict):
    with open(jsonfile) as fh:
        rois = json.load(fh)
    return construct_dense_mask(rois, roidict)
