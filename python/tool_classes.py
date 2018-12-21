
from object_detection.utils import label_map_util
import json
import argparse
from enum import Enum
class KeyType(Enum):
    INDEX = 'index'
    NAME = 'name'
    ID = 'id'
   

def read_classes_names(classes_path):
    '''loads the classes'''
    with open(classes_path, 'r') as f:
        class_names = f.readlines()
    list=[]
    for i,c in enumerate(class_names):
        list.append({"id":i+1,"name":c.strip() })

    return list
def get_class_item(datas,key,key_type):
    if key_type==KeyType.INDEX.value:
        key=int(key)
        return key,datas[key]
    if key_type==KeyType.NAME.value:
        key=str(key)
        for i,data in enumerate(list(datas)):
            if data["name"]==key:
                return i,datas[i]
    if key_type==KeyType.ID.value:
        key=int(key)
        for i,data in enumerate(list(datas)):
            if data["id"]==key:
                return i,datas[i]
    return None 
def get_id_list(datas):
    ids=[]
    for data in datas:
        ids.append(data["id"])
    return ids
def read_label_map(classes_path):
        return label_map_util.create_categories_from_labelmap(classes_path, use_display_name=True)

def read_json(classes_path):

    with open(classes_path, 'r') as f:
        list=json.load(f)
    #print(dict)
    return list

def write_json(datas,out_path):
    with open(out_path, 'w') as f:
        json.dump(datas, f)

def write_classes_names(datas,out_path):
    txt=""
    for data in datas:
        txt='%s%s\n'%(txt,data["name"])
    with open(out_path, 'w') as f:
        f.write(txt)
    
def write_label_map(datas,out_path):
    txt=""
    for data in datas:
        out = ''
        out += 'item {\n'
        out += '  name: "' + data["name"]+'"\n'
        out += '  id: ' + str(data["id"])+'\n'
        out += '}\n'
        txt='%s%s'%(txt,out)
    with open(out_path, 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classes formet converter between "classes_lines","label_map","json".')
    parser.add_argument(
    '--in_formet', type=str,required=True,
    default="classes_lines",
    help='input format: "classes_lines","label_map","json" ')
    parser.add_argument(
    '--out_formet',type=str,required=True,
    default="label_map",
    help='output format: "classes_lines","label_map","json"')
    parser.add_argument(
    '--input',  
    default="model_data/coco_classes80.txt",
    type=str,required=True,
    help='input file')
    parser.add_argument(
    '--output',  
    default="model_data/coco_label_map.pbtxt",
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


   