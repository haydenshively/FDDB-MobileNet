import numpy as np
import cv2
import tensorflow as tf

from ByGoogle import label_map_util




path_to_ckpt = 'Models/fullybaked/' + 'frozen_inference_graph.pb'
path_to_labels = 'Data/Generated/' + 'FDDB_label_map.pbtxt'

num_classes = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(path_to_labels)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (width, height) = image.size
    return np.array(image.getdata()).reshape((height, width, 3)).astype(np.uint8)

cam=cv2.VideoCapture(0)
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        #image = Image.open(path_to_image)
        while cam.isOpened():
            image_np = cam.read()[1]#load_image_into_numpy_array(image)#
            image_np = cv2.pyrDown(image_np)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            image_np_expanded = np.expand_dims(image_np, axis=0)

            (boxes, scores, classes, num) = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_np_expanded})

            # print(scores[0,0])# max
            # print(boxes[0,0])# max
            good_boxes = boxes[scores > .3].tolist()
            verified_boxes = []
            for good_box in good_boxes:
                h = good_box[2] - good_box[0]
                # w = good_box[3] - good_box[1]
                if h > 0.1:
                    verified_boxes.append(good_box)

            verified_boxes = np.asarray(verified_boxes)
            if len(verified_boxes.shape) is 2:
                height, width = image_np.shape[:2]
                mean = verified_boxes.mean(axis = 0)
                mean[0::2] *= height
                mean[1::2] *= width
                mean = mean.astype('uint16')

                # cv2.rectangle(image_np,(mean[1],mean[0]),(mean[3],mean[2]),(255,255,255),3)
                face = image_np[mean[0]:mean[2], mean[1]:mean[3]]

                eyes = eye_cascade.detectMultiScale(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY))
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

                cv2.imshow('face', cv2.pyrUp(face))



            cv2.imshow('test', cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break
