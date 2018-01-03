
import numpy as np
import json
from pycocotools.mask import encode, decode

def construct_dense_mask(rois, tissuedict):
    """constructs a dense mask given a list of `rois`
    and a dictionary mapping roi names to channel
    numbers in tissuedict are expected to start at one
    as the default class is constructed
    and assigned to zeroth channel

    Calls `pycocotools.mask.decode`
    """
    nchannels = 1+max(tissuedict.values())
    maskarr = np.zeros(rois[-1]["size"] + [nchannels], dtype=bool)
    
    for roi_ in rois:
        mask = decode(roi_)
        name = roi_["name"]
        if name in tissuedict:
            channel = tissuedict[name]
            maskarr[..., channel] |= mask.astype(bool)
    maskarr[..., 0] = ~maskarr.any(-1)
    assert maskarr.sum(-1).max() == 1
    return maskarr


def construct_sparse_mask(rois, tissuedict):
    """constructs a sparse mask given a list of `rois`
    and a dictionary mapping roi names to channel
    numbers in tissuedict are expected to start at one
    as the default class is constructed
    and assigned to zeroth channel

    Calls `pycocotools.mask.decode`
    """
    nchannels = 1+max(tissuedict.values())
    maskarr = np.zeros(rois[-1]["size"], dtype=bool)
    
    for roi_ in rois:
        mask = decode(roi_)
        name = roi_["name"]
        if name in tissuedict:
            channel = tissuedict[name]
            maskarr = np.maximum(maskarr, channel*mask.astype(np.uint8))
    return maskarr


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
