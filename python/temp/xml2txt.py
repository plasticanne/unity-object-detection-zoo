import os
import glob
import xml.etree.ElementTree as ET



def xml_to_csv(path):
    xml_list = []
    txt=""
    for xml_file in glob.glob(path + '/*.xml'):
        filename=os.path.split(xml_file)[1].split(".")[0]
        tree = ET.parse(xml_file)
        root = tree.getroot()
        root_txt='./data/images/%s.jpg'%(filename)
        box_list = ""
        for member in root.findall('object'):
            box_list=box_list+' %s,%s,%s,%s,%s'%(
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text),
                    0)
        txt='%s%s%s\n'%(txt,root_txt,box_list)
    list_file = open('raccoon_labels.txt', 'w')
    list_file.write(txt)
    list_file.close()
    


def main():
    
    image_path = os.path.join(os.getcwd(), 'annotations')
    xml_to_csv(image_path)
    print('Successfully')


main()
