
// a script for exporting ROIs as vertices and other metadata in JSON format
// to be replaced with proper serialization of qupath.lib.roi.AreaROI

//  In order to run it, you'll need to create a QuPath project, 
// load the image and annotation,
// go to the "Automate" drop down menu,
// and press first item "Show script editor".
// From the script editor, open and run the script attached --
// it must export coordinates into the project directory.

//--------------------------------------------------------

//groovy.json.methods.collect { println it.name };
//@Grab("org.codehaus.groovy:groovy-json")
//import groovy.json.*
//import flexjson.JSONDeserializer
 
//import com.fasterxml.jackson.core.JsonGenerationException;
//import com.fasterxml.jackson.databind.JsonMappingException;
//import com.fasterxml.jackson.databind.ObjectMapper;
//import com.fasterxml.jackson.databind.SerializationFeature;
//ObjectMapper mapper = new ObjectMapper();
// enable pretty printing
//mapper.enable(SerializationFeature.INDENT_OUTPUT);
//JSONSerializer serializer = new JSONSerializer();
//     serializer.serialize( person );


//public interface JsonSerializer<qupath.lib.roi.AreaROI>
// class AreaSerializer implements JsonSerializer<qupath.lib.roi.AreaROI>() {
//   public JsonElement serialize(Id id, Type typeOfId, JsonSerializationContext context) {
//     return new JsonPrimitive(id.getValue());
//   }
// }


//import java.io.IOException;
import org.json.*

def serializeArea(roi){
    JSONObject jroi = new JSONObject();
    // serialize the object
    for (prp in ['roiType', 'empty', 'area']){
        jroi.put(prp, roi[prp]);
    }
    
    for (prp in ["geometryType"]){
        jroi.put(prp, roi.geometry[prp]);
    }

    jroi.put("interiorPoint", [roi.geometry.interiorPoint.x, roi.geometry.interiorPoint.y])
    jroi.put("centroid", [roi.centroidX, roi.centroidY])
    jroi.put("boundsStart", [roi.boundsX, roi.boundsY])
    jroi.put("boundsSize", [roi.boundsWidth, roi.boundsHeight])
//    println roi.geometry.coordinates[0].x
//    jroi.put("vertices", roi.geometry.coordinates)
    def jvert = new ArrayList<>();;
    for (point in roi.getPolygonPoints()){
        jvert.add([point.x, point.y]);
    }
    jroi.put("vertices", jvert)

    return jroi;
}

// make a file name and directory
def entry = getProjectEntry()
def name = entry.getImageName() + '.txt'
def path = buildFilePath(PROJECT_BASE_DIR, 'annotation')
mkdirs(path)
path = buildFilePath(path, name)
// collect ROIs
JSONArray jstr = new JSONArray();
for (annotation in getAnnotationObjects()) {
    def pathClass = annotation.getPathClass()
    def roi = annotation.getROI()
    jstr.put(serializeArea(roi));
}

File file = new File(path)
file.write jstr.toString();
println jstr;
