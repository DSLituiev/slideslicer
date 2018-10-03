import numpy as np
from skimage import color, exposure, transform, io

def preprocess_img(img):
    if img.shape[-1] > 3:
        img = img[:,:,:3]
    # Histogram normalization in v channel
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)

    # roll color axis to axis 0
    # img = np.rollaxis(img, -1)
    return img

if __name__ == '__main__':
    import os
    import sys

    def _get_output_name(img_path):
        out_path = img_path.split('.')
        out_path = '.'.join(out_path[:-1]) + '-histeq.' + out_path[-1]
       
    OUTDIR = "./data_1024_histeq"
    
    def _get_output_name(img_path):
        out_path = img_path.split(os.sep)
        out_path = os.path.join(OUTDIR, *out_path[-2:])
        return out_path

    img_path = sys.argv[1]
    out_path = _get_output_name(img_path)
    if os.path.exists(out_path):
        print("SKIPPING EXISTING:\t%s" % out_path)
        sys.exit(2)

    print("SAVING TO", out_path, sep='\t')

    os.makedirs(os.path.dirname(out_path), exist_ok = True)
    img = preprocess_img(io.imread(img_path))
    io.imsave(out_path, img)
