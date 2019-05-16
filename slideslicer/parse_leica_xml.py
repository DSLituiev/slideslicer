import numpy as np
import lxml
import lxml.etree


def _parse_vertices_(reg):
    # the query returns a list of strings
    vertices_x = reg.xpath('Vertices/Vertex/@X')
    vertices_y = reg.xpath('Vertices/Vertex/@Y')
    
    # convert strings to integers 
    # and pair them into [(x1, y2), (x2, y2), ... ] format
    vertices = (list(zip((float(x) for x in vertices_x),
                                   (float(x) for x in vertices_y))))
    return vertices


def _parse_xml_region_all_attrs_(reg, vertices_np=False, name=None):
    attrs = dict(reg.attrib)
    types_ = {'Id':int,
              'Type':int,
               'NegativeROA': int,
               'Analyze': int,
               'InputRegionId': int,
               'DisplayId': int,
               'ImageFocus': int,
               'Selected':int,
               'Zoom': float, 
               'Length': float,
               'Area': float,
               'LengthMicrons': float,
               'AreaMicrons': float,
               }
    for kk,vv in attrs.items():
        if kk in types_:
            try:
                attrs[kk] = types_[kk](vv)
            except Exception as ee:
                attrs[kk] = -1
                print(kk, ee)

#    for kk ['Id', 'Text', 'Area']:
#        vv = attrs[kk]
#        vv = int(vv) if len(vv)>0 else -1
#        attrs[kk] = vv
    if name is not None and (attrs['Text'] is None or attrs['Text']==''):
        attrs['Text'] = name
    attrs['Vertices'] = _parse_vertices_(reg)
    if vertices_np:
        attrs['Vertices'] = np.asarray(attrs['Vertices'])
    return dict(zip(map(lambda x:x.lower(),attrs.keys()),
                attrs.values()))

def _parse_xml_region_(reg):
    "parse a `region` element from Leica annotation XML"
    tag = reg.xpath('./@Text')
    tag = (tag[0]) if len(tag)>0 else -1
    
    id_ = reg.xpath('./@Id')
    id_ = int(id_[0]) if len(id_)>0 else -1

    area = reg.xpath('./@Area')
    area = float(area[0]) if len(area)>0 else -1.0
    
    return {"id": id_,
           "tag": tag,
           "vertices": vertices,
           "area": area,
          }

def parse_xml2annotations(fnxml):
    """Convert a leica XML annotation to a list of dictionaries
    Input:
        xml file name
        
    Output:
        a list of dictionaries with entries:
         - id       : "Id" field
         - tag      : "Textt" field
         - vertices : list of [(x1, y1), (x2, y2), ...] 
    """
    root = lxml.etree.parse(fnxml)
    rois = []

    annotations = root.xpath('//*/Annotation')
    for ann in annotations:
        try:
            attrbs = ann.xpath('./Attributes')[0]
            attrb = attrbs.xpath('''./Attribute[contains(@Name,'Description')]''')[0]
            name = attrb.xpath('./@Value')
            name = name[0]
        except:
            name = None
        # Get all "Region" elements in the tree
        for reg in ann.xpath('./*/Region'):
            rois.append(_parse_xml_region_all_attrs_(reg, name=name))
    return rois

# legacy
def xml_to_annotations(x):
    DeprecationWarning('`xml_to_annotations` deprecated; use `parse_xml2annotations` instead')
    return parse_xml2annotations(x)
