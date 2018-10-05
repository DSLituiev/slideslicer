import numpy as np
import lxml
import lxml.etree

def _parse_xml_region_(reg):
    "parse a `region` element from Leica annotation XML"
    tag = reg.xpath('./@Text')
    tag = (tag[0]) if len(tag)>0 else -1
    
    id_ = reg.xpath('./@Id')
    id_ = int(id_[0]) if len(id_)>0 else -1

    area = reg.xpath('./@Area')
    area = float(area[0]) if len(area)>0 else -1.0
    
    
    # the query returns a list of strings
    vertices_x = reg.xpath('Vertices/Vertex/@X')
    vertices_y = reg.xpath('Vertices/Vertex/@Y')
    
    # convert strings to integers 
    # and pair them into [(x1, y2), (x2, y2), ... ] format
    vertices = np.asarray(list(zip((float(x) for x in vertices_x),
                                   (float(x) for x in vertices_y))))
    
    return {"id": id_,
           "tag": tag,
           "vertices": vertices,
           "area": area,
          }

def xml_to_annotations(fnxml):
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
    # Get all "Region" elements in the tree
    regions = root.xpath('//*/Region')
#     print(len(regions), "regions")
    annotations = []
    for reg in regions:
        annotations.append(_parse_xml_region_(reg))
    return annotations

