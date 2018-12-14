import numpy as np


class YOLO_Kmeans:

    def __init__(self, cluster_number):
        self.cluster_number = cluster_number
       

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters

    def result2txt(self, data,output):
        f = open(output, 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self,filename):
        f = open(filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def tfrecord2boxes(self,filename):
        import tensorflow as tf
        
        def parse_exmp(serial_exmp):
            feats = tf.parse_single_example(serial_exmp, features={
                #'image/filename':tf.FixedLenFeature([], tf.string),
                #'image/encoded':tf.FixedLenFeature([1], tf.string),
                'image/height':tf.FixedLenFeature([1], tf.int64),
                'image/width':tf.FixedLenFeature([1], tf.int64),
                #'image/object/class/label':tf.VarLenFeature( tf.int64),
                #'image/object/class/index':tf.VarLenFeature( tf.int64),
                'image/object/bbox/xmin':tf.VarLenFeature( tf.float32),
                'image/object/bbox/xmax':tf.VarLenFeature( tf.float32),
                'image/object/bbox/ymin':tf.VarLenFeature( tf.float32),
                'image/object/bbox/ymax':tf.VarLenFeature( tf.float32)
            })
            return feats
        dataset = tf.data.TFRecordDataset(RECORDS)
        dataset = dataset.map(parse_exmp).batch(1)
        iter_dataset   = dataset.make_one_shot_iterator().get_next()
        with tf.Session() as sess:  
            result=np.empty(shape=[0, 2])
            while True:
                try:
                    parsed_features=sess.run(iter_dataset)
                    w=np.around(parsed_features["image/object/bbox/xmax"].values*parsed_features['image/width'])
                    -np.around(parsed_features["image/object/bbox/xmin"].values*parsed_features['image/width'])
                    h=np.around(parsed_features["image/object/bbox/ymax"].values*parsed_features['image/height'])
                    -np.around(parsed_features["image/object/bbox/ymin"].values*parsed_features['image/height'])
                    boxs=np.vstack((w,h)).astype(np.int32).T
                    result=np.vstack((result,boxs))
                except tf.errors.OutOfRangeError:
                    return result     

    def data2clusters(self,output,all_boxes):
       
        result = self.kmeans(all_boxes, k=self.cluster_number)
        result = result[np.lexsort(result.T[0, None])]
        self.result2txt(result,output)
        print("K anchors:\n {}".format(result))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes, result) * 100))


if __name__ == "__main__":
    # data from
    data_from=1
    # 0: tf-records
    RECORDS = ['dataset/raccoon/a.record']
    # 1: orginal txt file
    ANNOTATIONS = "model_data/raccoon_annotations.txt"

    # args
    LEN_anchors = 9
    OUTPUT_anchors='model_data/raccoon_anchors2.txt'


    kmeans = YOLO_Kmeans(LEN_anchors)
    if data_from==1:
        all_boxes = kmeans.txt2boxes(ANNOTATIONS)
    else:
        all_boxes = kmeans.tfrecord2boxes(RECORDS)
    kmeans.data2clusters(OUTPUT_anchors,all_boxes)
