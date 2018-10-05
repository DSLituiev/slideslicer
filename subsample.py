import os, sys
from PIL import Image

def get_outfile(infile):
    outfile = os.path.basename(infile)
    outsubdir = os.path.basename(os.path.dirname(infile))
    outfile = os.path.join(OUTDIR, outsubdir, outfile)
    return outfile

def filegen(indir, ext='png'):
    for dd in os.scandir(indir):
        if not os.path.isdir(dd.path):
            continue
        for ff in os.scandir(dd.path):
            if not ff.name.endswith(ext):
                continue
            yield ff.path



from pycocotools.mask import encode, decode
import json
from PIL import Image
import numpy as np



def subsample_coco_mask(mask):
    mask = Image.fromarray(mask)
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

def subsample_roi(fn, fnout):
    with open(fn) as fh:
        rois = json.load(fh)

    if len(rois) == 0:
        print("NO ROIS IN\t%s" % fn)

    for roi in rois:
        mask = decode(roi)
        mask = subsample_coco_mask(mask)
        cocomask = encode(mask)
        cocomask['counts'] = cocomask['counts'].decode()
        roi.update(cocomask)

        verts = roi["vertices"]
        roi.update({"vertices":subsample_verts(verts, factor)})
        if 'zoom' in roi:
            roi.pop('zoom')

    with open(fnout, 'w+') as fh:
        json.dump(rois, fh)

####################################################################

original_side= 1024
#side = 512 //2 
#factor = original_side // side

indir = sys.argv[1]
factor = int(sys.argv[2])


basedir = os.path.dirname(indir.rstrip('/'))
basedir = os.path.dirname(basedir)
basedir = os.path.dirname(basedir)

side = original_side // factor
OUTDIR = "{}/data_{}_subsample_{}x/fullsplit/all".format(basedir, side, factor)
print("SAVING TO:", OUTDIR, sep='\t')
size = side, side
os.makedirs(OUTDIR, exist_ok = True)

print("SUBSAMPLE IMAGES")
for infile in filegen(indir):
    outfile = get_outfile(infile)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "png")
        except IOError as ee:
            print( "cannot create thumbnail for '%s'" % infile)
            print(ee)


print("SUBSAMPLE MASKS")
for infile in filegen(indir, ext = '.json'):
    outfile = get_outfile(infile)
    if infile != outfile:
        try:
            subsample_roi(infile, outfile)
        except IOError as ee:
            print( "cannot create thumbnail for '%s'" % infile)
            print(ee)
        except Exception as ee:
            print( "cannot create thumbnail for '%s'" % infile)
            raise ee



