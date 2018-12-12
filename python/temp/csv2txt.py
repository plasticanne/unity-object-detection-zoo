import csv
with open('./data/test_labels.csv', newline='') as csvfile:
    rows = csv.reader(csvfile)
    list_file = open('test_labels.txt', 'w')
    for i, v in enumerate(rows):
        if i is not 0:
            list_file.write('./data/%s %s,%s,%s,%s,%s'%(v[0],v[4],v[5],v[6],v[7],v[3]))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
            list_file.close()