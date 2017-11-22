
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import numpy as np
from collections import Counter
import pandas as pd
import re
import json
# import pyaml


# In[2]:


# see https://github.com/bgilbert/anonymize-slide


# In[3]:


fnxml = "examples/6371/6371 1.xml"
fnjson = re.sub(".xml$", ".json", fnxml)
# fnyaml = re.sub(".xml$", ".yaml", fnxml)
with open(fnxml) as fh:
    soup = BeautifulSoup(fh, 'lxml')


# In[ ]:





# In[4]:


def get_ellipse_points(verticeslist, num=200):
    a,b = tuple(abs(verticeslist[1] - verticeslist[0])/2)
    xc, yc = tuple(abs(verticeslist[1] + verticeslist[0])/2)
    tt = np.arange(0,2*np.pi, 2*np.pi/num)
    x = xc + a*np.cos(tt)
    y = yc + b* np.sin(tt)
    return np.c_[x,y].tolist()
#     return ((x),list(y))


# In[5]:


def get_vertices(region):
    verticeslist = [cc for cc in region.vertices.children if cc!='\n']
    verticeslist = [(vv.get('x'), vv.get('y')) for vv in verticeslist]
    verticeslist = [(float(x), float(y)) for x,y in verticeslist]
#     verticeslist = np.asarray(verticeslist)
    if region["type"] == '2':
        verticeslist = get_ellipse_points(np.asarray(verticeslist), num=200)
    return verticeslist


# In[6]:


regions = soup.find_all("region")


# In[7]:


regionlist = []
for rr in regions:
    name = rr.get("text").lower().rstrip('.')
    regionlist.append(dict(name=name, vertices = get_vertices(rr)))


# for an ellipse, 
# 
#    area = $\pi \times r \times R$

# In[8]:


# ellipses = [{"vertices":get_vertices(rr), "length": rr["length"], "area":rr["area"]} for rr in regions if rr["type"]=='2']
# ell = ellipses[1]


# In[9]:


pd.Series([rr["name"] for rr in regionlist]).value_counts()


# In[10]:


with open(fnjson, 'w+') as fh:
    json.dump(regionlist, fh)


# In[11]:


# with open(fnyaml, 'w+') as fh:
#     pyaml.dump(regionlist, fh)


# In[12]:


from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')


# In[13]:


for rr in regionlist:
    plt.plot([x for x,_ in  rr["vertices"]],
             [y for _,y in  rr["vertices"]])


# In[14]:


# rr


# In[15]:


import openslide
from PIL import Image, ImageDraw
import cv2
import numpy as np


# In[16]:


def get_roi_mask(roi, width, height, fill=1, shape='polygon', radius=3):
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


# In[17]:


fnsvs = "examples/6371/6371 1.svs"


# In[18]:


slide = openslide.OpenSlide(fnsvs)
img = np.asarray(slide.associated_images["thumbnail"])


# In[19]:


slide.associated_images.keys()


# In[20]:


# slide.associated_images['label'] = None


# In[ ]:





# In[21]:


macro =  np.asarray(slide.associated_images["macro"])
macrohsv = cv2.cvtColor(macro,  cv2.COLOR_BGR2HSV)
# For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
lower = np.r_[59, 250, 242]
upper = np.r_[61, 255, 255]

mask = cv2.inRange(macrohsv, lower, upper)
_, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = [np.squeeze(x) for x in contours if x.shape[0]>10]
contour = contours[np.argmax([x.shape[0] for x in contours])]
del contours
len(contour)


# In[22]:


# contour


# In[23]:


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


# In[24]:


# plt.figure(figsize = (18,10))
# plt.imshow(macro[ymin:ymax, xmin:xmax])
# # plt.imshow(macro)
# # for co in [contour]:
# #     plt.plot(co[:,0], co[:,1])


# In[ ]:





# In[25]:


# Image.fromarray(macro).save("macro.png")
# cv2.imwrite("macro.png", macro)


# In[ ]:





# In[ ]:





# In[26]:


th = slide.associated_images["thumbnail"]
th.getbbox()


# In[27]:


for kk,vv in slide.associated_images.items():
    print(kk, vv.size)


# In[28]:


plt.imshow(img)


# In[ ]:





# In[74]:


def get_median_color(slide):
    return np.apply_over_axes(np.median, 
                              np.asarray(slide.associated_images["thumbnail"]),
                              [0,1]).ravel()

median_color = get_median_color(slide)


# In[30]:


def get_chunk_masks(img, color=False, filtersize=7,
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


def get_contours_from_mask(mask, minlen = 50):
    im2, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if minlen is not None:
        contours = [np.squeeze(x) for x in contours if x.shape[0]>minlen]
    return contours

def get_chunk_countours(img, color=False, filtersize=7, minlen = 100):
    mask = get_chunk_masks(img, color=color, filtersize=filtersize)
    contours = get_contours_from_mask(mask, minlen = minlen)
    return contours


# In[31]:


def get_thumbnail_magnification(slide):
    ratio = np.asarray(slide.dimensions) / np.asarray(slide.associated_images["thumbnail"].size)
    return ratio # np.sqrt(np.prod(ratio))

def roi_loc(roi):
    xmin, ymin = roi.min(0)
    xmax, ymax = roi.max(0)
    return np.r_[xmin, ymin], np.r_[xmax-xmin, ymax-ymin]


# In[ ]:





# In[ ]:





# In[94]:


mask = get_chunk_masks(img, color=False, filtersize=7)
contours = get_contours_from_mask(mask, minlen = 100)


# In[33]:


plt.imshow(mask)


# In[ ]:





# In[34]:


plt.figure(figsize = (18,10))
plt.imshow(img)
for co in contours:
    plt.plot(co[:,0], co[:,1])


# In[35]:


img.shape


# In[36]:


height, width, _ = img.shape
# objmask = get_roi_mask(co, width, height, fill=1, shape='polygon', radius=3)


# In[37]:


# objmask = np.zeros((height,width), np.uint8)
# objmask = cv2.fillPoly(objmask, [co], 1)


# In[125]:


# cv2.moments(co)
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


# In[126]:


# cropped, crop_contour = crop_rotate_contour(img, co)
co = contours[2]
# print(co[:2])
cr = CropRotateRoi(img, co, enlarge=1.0, borderValue=median_color)

crimg = cr(img, crop=False)
crroi = cr(co)


# In[127]:


plt.figure(figsize=(15,2))
plt.imshow(crimg)

plt.plot(crroi[:,0], crroi[:,1])


# In[54]:


def plot_roi(roi, **kwargs):
    return plt.plot(roi[:,0], roi[:,1], **kwargs)


# In[148]:


def get_highres_roi(slide, roi, color=True, filtersize=35, minlen=500,
                    median_color=None,
                    angle=None
                    ):
    if median_color is None:
        median_color = get_median_color(slide)
    ratio = get_thumbnail_magnification(slide)
    
    roi_start, roi_size = roi_loc(roi)
    region = slide.read_region((roi_start*ratio).astype(int), 0, 
                               (roi_size*ratio).astype(int))

    crmag = CropRotateRoi(region, roi*ratio,
                          angle=angle,
                          use_offset=True,
                          borderValue=median_color)
    region_, co_scaled_ = crmag(region, roi*ratio, crop=True)
    # re-estimate the contour
    mask_ = get_chunk_masks(region_, color=color, filtersize=filtersize)
    chunk_contour = get_contours_from_mask(mask_, minlen = minlen)
    print(len(chunk_contour))
    assert len(chunk_contour)>0
    # take the longest contour
    maxidx = np.argmax([len(x) for x in chunk_contour])
    chunk_contour = chunk_contour[maxidx]
    return region, chunk_contour


# ## Now we can slice the chunk horizonatally, and stack highres images from the slices

# In[43]:


np.ceil(crimg.shape[1]/int(np.ceil(crimg.shape[1]/100)))


# In[89]:


stepsize = 100
nsteps = int(np.ceil(crimg.shape[1]/stepsize))
print(nsteps)


# In[149]:


# rotation_matrix = cr.rotation_matrix.copy()
# rotation_matrix[:,2] *= ratio
# rotation_matrix

for step in range(nsteps):
    currentchunk = crimg[:,stepsize*step:stepsize*(step+1),:]
    chunk_contours = get_chunk_countours(currentchunk, 
                               minlen = 10 , color=False,
                               filtersize=7)
    assert len(chunk_contours) ==1
    chunk_contour = chunk_contours[0]
    chunk_slice_contour_origcoord = np.linalg.solve(CropRotateRoi._pad_affine_matrix_(cr.affine_matrix), 
                                                    CropRotateRoi._pad_vectors_(chunk_contour).T).T[:,:2]
    
    region, chunk_slice_contour_origcoord_mag = get_highres_roi(slide, chunk_slice_contour_origcoord,
                                                               angle=cr.angle
                                                               )
    break


# In[153]:


plt.imshow(region_)
plot_roi(chunk_slice_contour_origcoord_mag)


# In[578]:


# plt.imshow(mask)
# plot_roi(contours[0])


# In[562]:





# In[466]:


ratio = get_thumbnail_magnification(slide)
ratio


# In[467]:


roi_start, roi_size = roi_loc(co)
roi_start, roi_size


# In[468]:


roi_start + roi_size


# In[129]:


def scale_roi(co, start, ratio=1):
    return (co - start)*ratio


# In[470]:


rois_within_chunk = []
for rr in regionlist:
    vertices = np.asarray(rr["vertices"]/ratio)
    left = (vertices>=roi_start).all(0).all()
    right = (vertices<=(roi_start + roi_size)).all(0).all()
#     print(rr["name"], left, right)
    if left and right:
        print(rr["name"])
        rois_within_chunk.append(rr)


# In[471]:


(vertices<=(roi_start + roi_size)).all(0).all()


# In[473]:


# co


# In[150]:


((co-roi_start)*ratio).max(0), region.size


# In[212]:


# co*ratio - (co*ratio).min(0)


# In[304]:


# plot_roi(co_scaled)
# median_color


# In[399]:


# co = contours[0]
# print(co[:2])
# cr = CropRotateRoi(region, co_scaled, enlarge=1.07, offset=True)
cr = CropRotateRoi(region, co*ratio, enlarge=1.05, 
                   use_offset=True, borderValue=median_color)
region, co_scaled_ = cr(region, co*ratio, crop=True)
region.shape


# In[386]:


plt.imshow(region)


# In[402]:


mask_ = get_chunk_masks(region, color=True, filtersize=15)
chunk_contour = get_contours_from_mask(mask_, minlen = 100)
del mask


# In[403]:


# chunk_contour
plot_roi(chunk_contour[0])


# In[406]:


plt.figure(figsize=(20,8))
plt.imshow(np.asarray(region))

plot_roi(co_scaled_)
plot_roi(chunk_contour[0])
for rr in rois_within_chunk:
    roi = cr.apply_roi(np.asarray(rr["vertices"]), use_offset=True)
    if (roi<0).all(0).any():
        continue
    plt.plot(roi[:,0],
             roi[:,1])


# In[397]:




