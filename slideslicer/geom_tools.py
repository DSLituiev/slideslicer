from functools import reduce
import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import MultiLineString
import cv2
from warnings import warn


def clean_polygon(pp):
    area = pp.area
    coords = np.asarray(pp.boundary.coords)
    buff = pp.buffer(0)
    if buff.area ==0:
        for _ in range(3):
            pp = _permute_polygon_(pp, fraction=3)
            buff = pp.buffer(0)
            if buff.area*3 > area:
                break
    boundary = buff.boundary
    try:
        if isinstance(boundary, MultiLineString):
            if len(boundary) == 0:
                return Polygon()
            ind = np.argmax([x.area for x in  boundary])
            warn('strange shapes: areas:\t{}'.format(str([x.area for x in  boundary])) )
            boundary = boundary[ind]
        danglingpiece = np.asarray(boundary.coords)
    except NotImplementedError as ee:
        warn("shape type:\t{}".format(type(pp)))
        warn("boundary type:\t{}".format(type(boundary)))
        raise ee
        
    negmask = reduce(lambda x,y: x|y, ((coords == row).all(1) for row in danglingpiece))
    if np.mean(negmask)>0.5:
        #warn('having to invert the mask')
        negmask = ~negmask
    pp = Polygon(coords[~negmask])
    return pp.buffer(0)


def resolve_selfintersection(pp, areathr=1e-3, areakeep=100,
                             fraction=5, depth=0):
    """ reslove self-intersection for a shapely.geometry.Polygon object
    param: areathr  -- min area difference 
    """
    pbuf = pp.buffer(0)#.interiors
    try:
        areadiff = abs(pp.area- pbuf.area)/pp.area
    except ZeroDivisionError:
        return MultiPolygon([pp])

    if (areadiff > areathr):
        pp = clean_polygon(pp)
        pps = resolve_selfintersection(pp, areathr=areathr,
                                      fraction=3, depth=depth+1)
        return pps
    else:
        try:
            pp = Polygon(pbuf)
            assert pp.is_valid
            return MultiPolygon([pp])
        except NotImplementedError as ee:
            #warn(str(ee))
            ind = np.argmax([x.area for x in pbuf])
            ind = [ii for ii, x in enumerate(pbuf) if x.area > areakeep]
            return MultiPolygon([pbuf[ii] for ii in ind])


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


def get_ellipse_verts_from_bbox(vertices, points=50):
    (x0,y0), (x1, y1) = vertices
    MajorAxisLength = int(1*abs(x0-x1))
    MinorAxisLength = int(1*abs(y0-y1))
    Centroid = np.r_[(x1+x0)/2, (y1+y0)/2]
    xx, yy = get_ellipse_verts((MinorAxisLength, MajorAxisLength) , Centroid)
    return list(zip(xx,yy))


def get_ellipse_verts(AxesLengths, Centroid, points=50):
    """from matlab blog:
    https://blogs.mathworks.com/steve/2015/08/17/ellipse-visualization-and-regionprops/"""
    MinorAxisLength, MajorAxisLength = AxesLengths
    t = np.linspace(0,2*np.pi,points);

    a = MajorAxisLength/2;
    b = MinorAxisLength/2;
    Xc = Centroid[0];
    Yc = Centroid[1];
    phi = 0# deg2rad(-s(k).Orientation);
    x = Xc + a*np.cos(t)*np.cos(phi) - b*np.sin(t)*np.sin(phi);
    y = Yc + a*np.cos(t)*np.sin(phi) + b*np.sin(t)*np.cos(phi);
    return x,y
