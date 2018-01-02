import yaml

# load dictionary (str -> int)

fn =  "tissuedict.yaml"
with open(fn) as fh:
    tissuedict = yaml.load(fh)
tissuedict

# for each json file
#    load the json file
#    for each roi in json file:
#       convert roi entry to binary mask
#       multiply entry by correspondent int from tissuedict

