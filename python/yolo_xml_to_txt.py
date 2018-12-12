import os
import glob
import xml.etree.ElementTree as ET
from convert_classes import read_label_map,get_class_item
def get_class(classes_path_raw):
    # return label_map_util.create_category_index_from_labelmap(classes_path_raw, use_display_name=True)
    return read_label_map(classes_path_raw)
def get_class_index(datas,name):
    index,_=get_class_item(datas,name,'name')
    return index
def main(annotations_folder,images_folder,output_file,classes_path,label_offset):
    xml_list = []
    txt=""
    class_list=get_class(classes_path)
    for xml_file in glob.glob(annotations_folder + '/*.xml'):
        filename=os.path.split(xml_file)[1].split(".")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root_txt='%s/%s.jpg'%(images_folder,filename)
        box_list = ""
        for member in root.findall('object'):
            box_list=box_list+' %s,%s,%s,%s,%s'%(
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text),
                    get_class_index(class_list,member[0].text)+label_offset)
        txt='%s%s%s\n'%(txt,root_txt,box_list)
    list_file = open(output_file, 'w')
    list_file.write(txt)
    list_file.close()
    print('Successfully')
    


if __name__ == '__main__':
    ANNOTATIONS_folder='dataset/raccoon/annotations'
    IMAGES_folder='dataset/raccoon/images'
    CLASSES_labels='model_data/raccoon_labels_map.pbtxt'
    OUTPUT_file='model_data/raccoon_annotations.txt'
    LABEL_offset=0 # index from 0
    main(ANNOTATIONS_folder,IMAGES_folder,OUTPUT_file,CLASSES_labels,LABEL_offset)
