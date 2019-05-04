#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from PIL import Image
from pycocotools.mask import encode, decode

def get_outfile(infile, outdir):
    outfile = os.path.basename(infile)
    outsubdir = os.path.basename(os.path.dirname(infile))
    outfile = os.path.join(outdir, outsubdir, outfile)
    return outfile


def filegen(indir, ext='png'):
    for dd in os.scandir(indir):
        if not os.path.isdir(dd.path):
            continue
        for ff in os.scandir(dd.path):
            if not ff.name.endswith(ext):
                continue
            yield ff.path


def subsample_coco_mask(mask, factor):
    mask = Image.fromarray(mask)
    sz = np.asarray(mask.size)
    out_sz = (sz / factor).astype(int)
    mask.thumbnail(size, Image.ANTIALIAS)
    mask = np.asarray(mask, order='F')
    return mask


def subsample_verts(verts, factor):
    verts = (np.asarray(verts)//factor).tolist()

    vvprev = [None, None]

    newverts = []
    for vv in verts:
        if np.all(vv == vvprev):
           continue
        else:
            newverts.append(vv)
        vvprev = vv.copy()
    return newverts


def subsample_roi(fn, fnout, factor):
    with open(fn) as fh:
        rois = json.load(fh)

    if len(rois) == 0:
        print("NO ROIS IN\t%s" % fn)

    for roi in rois:
        mask = decode(roi)
        mask = subsample_coco_mask(mask, factor)
        cocomask = encode(mask)
        cocomask['counts'] = cocomask['counts'].decode()
        roi.update(cocomask)

        verts = roi["vertices"]
        roi.update({"vertices": subsample_verts(verts, factor)})
        if 'zoom' in roi:
            roi.pop('zoom')

    with open(fnout, 'w+') as fh:
        json.dump(rois, fh)

####################################################################
if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'indir', type=str, help = 'input directory, e.g. `fullsplit/all`')

    parser.add_argument(
        '--outdir',
        default='',
        type=str,
        help = 'output directory')

    parser.add_argument(
      '--original-side',
      type=int,
      default=1024,
      help='Original side (in pixels) of the image patches')


    parser.add_argument('--img', dest='img', action='store_true')
    parser.add_argument('--no-img', dest='img', action='store_false')
    parser.set_defaults(img=True)

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
      '--target-side',
      type=int,
      default=512,
      help='Requested side (in pixels) of the image patches')

    group.add_argument(
      '--factor',
      type=int,
      default=None,
      help='Requested subsampling rate')
    
    prms = parser.parse_args()

    if prms.factor is None:
        factor = prms.original_side // prms.target_side 
        side = prms.target_side
    else:
        factor = prms.factor
        side = prms.original_side // prms.factor
    size = side, side

    if len(prms.outdir)==0:
        basedir, set_ = os.path.split(prms.indir.rstrip('/'))
        basedir, split_ = os.path.split(basedir.rstrip('/'))
        #basedir = os.path.dirname(basedir)
        #basedir = os.path.dirname(basedir)
        outdir = "{}/data_{}_subsample_{:d}x/{}/all".format(basedir, side, factor, split_)
    else:
        outdir = prms.outdir

    print("SAVING TO:", outdir, sep='\t')
    os.makedirs(outdir, exist_ok = True)
    
    if prms.img:
        print("SUBSAMPLING IMAGES")
        for infile in filegen(prms.indir):
            outfile = get_outfile(infile, outdir=outdir)
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            if infile != outfile:
                try:
                    im = Image.open(infile)
                    im.thumbnail(size, Image.ANTIALIAS)
                    im.save(outfile, "png")
                except IOError as ee:
                    print( "cannot create thumbnail for '%s'" % infile)
                    print(ee)

    print("SUBSAMPLING MASKS")
    for infile in filegen(prms.indir, ext = '.json'):
        outfile = get_outfile(infile, outdir=outdir)
        if infile != outfile:
            try:
                subsample_roi(infile, outfile, factor)
            except IOError as ee:
                print( "cannot create thumbnail for '%s'" % infile)
                print(ee)
            except Exception as ee:
                print( "cannot create thumbnail for '%s'" % infile)
                raise ee
