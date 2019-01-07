# dataset format of yolo (https://pjreddie.com/darknet/yolo/)
import numpy as np
import os
import glob
import xml.etree.ElementTree as ET
from tool_classes import read_label_map,get_class_item

ANNOTATIONS_folder='dataset/raccoon/annotations'
IMAGES_folder='dataset/raccoon/images'
CLASSES_labels='model_data/raccoon_labels_map.pbtxt'
OUTPUT_label_folder='dataset/raccoon/labels'
OUTPUT_list_folder='dataset/raccoon'
VAILD_ratio=0.1
LABEL_offset=0 # index from 0
RANDOM=True

def get_class(classes_path_raw):
    # return label_map_util.create_category_index_from_labelmap(classes_path_raw, use_display_name=True)
    return read_label_map(classes_path_raw)
def get_class_index(datas,name):
    index,_=get_class_item(datas,name,'name')
    return index
def tfloat(d):
    return float(d.text)
def convert(size, box):
    xmin=tfloat(box.find('xmin'))
    ymin=tfloat(box.find('ymin'))
    xmax=tfloat(box.find('xmax'))
    ymax=tfloat(box.find('ymax'))

    dw = 1./tfloat(size.find('width'))
    dh = 1./tfloat(size.find('height'))
    x = (xmax + xmin)/2.0 - 1
    y = (ymax + ymin)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def process_list(xml_list,output_list_file,output_label_folder,class_list):
    out_list_txt=""
    images_folder="\\".join(IMAGES_folder.split("/"))
    for xml_file in xml_list:
        filename=os.path.split(xml_file)[1].split(".")[0]
        image_path= os.path.join( os.getcwd(),images_folder,filename+'.jpg')
        label_path= os.path.join( os.getcwd(),output_label_folder,filename+'.txt')
        out_list_txt='%s%s\n'%(out_list_txt,image_path)
        
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size=root.find('size')
        
        out_box_txt = ""
        for member in root.findall('object'):
           
            x,y,w,h=convert(size, member.find('bndbox'))
            
            out_box_txt=out_box_txt+'%s %s %s %s %s\n'%(
                    get_class_index(class_list,member.find('name').text)+LABEL_offset,
                    x,
                    y,
                    w,
                    h,
            )
        with open(label_path, 'w') as g:
            g.write(out_box_txt)


    with open(output_list_file, 'w') as f:
        f.write(out_list_txt)
    

def main():
    
    txt=""
    class_list=get_class(CLASSES_labels)
    xml_list=glob.glob(ANNOTATIONS_folder + '/*.xml')
    if RANDOM:
        np.random.seed(10101)
        np.random.shuffle(xml_list)
        np.random.seed(None)
    len_all=len(xml_list)
    len_vaild=int(len_all*VAILD_ratio)
    len_train=len_all-len_vaild
    train_list=xml_list[:len_train]
    vaild_list=xml_list[len_train:]
    if not os.path.isdir(OUTPUT_label_folder): os.makedirs(OUTPUT_label_folder)
    process_list(train_list,os.path.join(OUTPUT_list_folder,'yolo_train.txt'),OUTPUT_label_folder,class_list)
    if len_vaild>0:
        process_list(vaild_list,os.path.join(OUTPUT_list_folder,'yolo_vaild.txt'),OUTPUT_label_folder,class_list)
    
    print('Successfully')
    


if __name__ == '__main__':

    
    main()
