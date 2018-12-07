
from object_detection.utils import label_map_util
import json
import argparse

def read_classes_names(classes_path):
    '''loads the classes'''
    with open(classes_path, 'r') as f:
        class_names = f.readlines()
    dict={}
    for i,c in enumerate(class_names):
        dict[i+1]={"id":i+1,"name":c.strip() }
    return dict
            
def read_label_map(classes_path):
        return label_map_util.create_category_index_from_labelmap(classes_path, use_display_name=True)

def read_json(classes_path):

    with open(classes_path, 'r') as f:
        dict=json.load(f)
    #print(dict)
    return dict

def write_json(data,out_path):
    with open(out_path, 'w') as f:
        json.dump(data, f)

def write_classes_names(data,out_path):
    keys = data.keys() 
    sorted(keys)
    txt=""
    for key in keys:
        txt='%s%s\n'%(txt,data[key]["name"])
    with open(out_path, 'w') as f:
        f.write(txt)
    
def write_label_map(data,out_path):
    keys = data.keys() 
    sorted(keys)
    txt=""
    for key in keys:
        out = ''
        out += 'item {\n'
        out += '  name: "' + data[key]["name"]+'"\n'
        out += '  id: ' + str(data[key]["id"])+'\n'
        out += '}\n'
        txt='%s%s'%(txt,out)
    with open(out_path, 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classes formet converter between "classes_lines","label_map","json".')
    parser.add_argument(
    '--in_formet', type=str,required=True,
    default="label_map",
    help='input format: "classes_lines","label_map","json" ')
    parser.add_argument(
    '--out_formet',type=str,required=True,
    default="classes_lines",
    help='output format: "classes_lines","label_map","json"')
    parser.add_argument(
    '--input',  
    default="model_data/coco_label_map.pbtxt",
    #default="model_data/coco_classes.txt",
    #default="object_detection/data/mscoco_label_map.pbtxt",
    type=str,required=True,
    help='input file')
    parser.add_argument(
    '--output',  
    default="model_data/coco_classes80.txt",
    #default="model_data/coco_label_map.pbtxt",
    #default="model_data/coco_classes80.json",
    #default="model_data/coco_classes90.json",
    type=str,required=True,
    help='output file')
    FLAGS = parser.parse_args()
    read = {
        "classes_lines" : read_classes_names,
        "label_map" : read_label_map,
        "json" : read_json,
    }
    data=read[FLAGS.in_formet](FLAGS.input)
    write = {
        "classes_lines" : write_classes_names,
        "label_map" : write_label_map,
        "json" : write_json,
    }
    write[FLAGS.out_formet](data,FLAGS.output)


   