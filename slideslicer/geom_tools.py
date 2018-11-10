from functools import reduce
import numpy as np
import shapely
from shapely.geometry import Polygon
import cv2
from warnings import warn


def clean_polygon(pp):
    coords = np.asarray(pp.boundary.coords)
    danglingpiece = np.asarray(pp.buffer(0).boundary.coords)
    negmask = reduce(lambda x,y: x|y, ((coords == row).all(1) for row in danglingpiece))
    pp = Polygon(coords[~negmask])
    return pp.buffer(0)


def resolve_selfintersection(pp, areathr=1e-3, fraction=5, depth = 0):
    pbuf = pp.buffer(0)#.interiors
    areadiff = abs(pp.area- pbuf.area)/pp.area
        
    if (areadiff > areathr):
        pp = clean_polygon(pp)
        pp = resolve_selfintersection(pp, areathr=1e-3, fraction=3, depth=depth+1)
    else:
        try:
            pp = Polygon(pbuf)
        except NotImplementedError as ee:
            #warn(str(ee))
            ind = np.argmax([x.area for x in pbuf])
            return pbuf[ind]
        
    assert pp.is_valid
    return pp

def _permute_vertices_(vertices, fraction=3, break_point=None):
    if break_point is None:
        if fraction>1:
            fraction = 1/fraction
        break_point = int(vertices.shape[0]*fraction)
    

    pp = Polygon(np.flipud(np.vstack([vertices[break_point:],
                                      vertices[:break_point]])))
    return pp

def _permute_polygon_(pp, fraction=3, break_point=None):
    vertices = np.asarray(pp.boundary)
    del pp
    if break_point is None:
        if fraction>1:
            fraction = 1/fraction
        break_point = int(vertices.shape[0]*fraction)
    
    break_point = break_point if break_point <len(vertices) else len(vertices)-2

    pp = Polygon(np.flipud(np.vstack([vertices[break_point:],
                                      vertices[:break_point]])))
    return pp


def get_contour_centre(vertices):
    if len(vertices)==1:
        return vertices[0]

    mmnts = cv2.moments(np.asarray(vertices, dtype='int32'))
    cX = int(mmnts["m10"] / mmnts["m00"])
    cY = int(mmnts["m01"] / mmnts["m00"])
    return (cX, cY)
