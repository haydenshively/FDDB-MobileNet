from PIL import Image
import cv2
import numpy

import os
import io
import math

import tensorflow as tf
from ByGoogle import dataset_util

class Directory(object):
    def __init__(self, pathString):
        self.pathString = pathString
        self.directory = os.fsencode(pathString)
    def ls(self):
        return os.listdir(self.directory)
    def pathTo(self, file):
        return self.pathString + os.fsdecode(file)

def create_TFRecord(jpg_directory, jpg_name, xmins, xmaxs, ymins, ymaxs, class_strings, class_IDs):
    image_data = cv2.imread(jpg_directory + jpg_name)
    # cv2.imwrite(jpg_directory + jpg_name[:-4] + 'gray' + '.jpg', cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY))
    # cv2.imshow('image preview', image_data)
    # cv2.waitKey(1)
    image_data2 = Image.open(jpg_directory + jpg_name)#[:-4] + 'gray' + '.jpg')
    imgBytes = io.BytesIO()
    image_data2.save(imgBytes, format='JPEG')
    imgBytes = imgBytes.getvalue()
    image_format = b'jpg'#jpeg
    height, width = image_data.shape[:2]

    xmins = [i/width for i in xmins] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [i/width for i in xmaxs] # List of normalized right x coordinates in bounding box (1 per box)
    ymins = [i/height for i in ymins] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [i/height for i in ymaxs] # List of normalized bottom y coordinates in bounding box (1 per box)

    class_strings = [str.encode(i) for i in class_strings]

    tfRecord = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature(str.encode(jpg_name)),
    'image/source_id': dataset_util.bytes_feature(str.encode(jpg_name)),
    'image/encoded': dataset_util.bytes_feature(imgBytes),
    'image/format': dataset_util.bytes_feature(image_format),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(class_strings),
    'image/object/class/label': dataset_util.int64_list_feature(class_IDs),
    }))
    return tfRecord


def generate_arr_from(face_data_string):
    array = face_data_string.split(' ')
    return [float(i) for i in array if i is not '']


def main(_):
    writer = tf.python_io.TFRecordWriter('Generated/FDDB.record')
    database = 'Downloaded/'
    annotations = database + 'Annotations/'
    images_dir = database + 'Images/'

    current_image_path = None
    remaining_faces = None
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []

    directory = Directory(annotations)
    for fold in directory.ls():
        with open(directory.pathTo(fold)) as fold_text:
            for line in fold_text:
                if '/' in line:
                    current_image_path = line[:-1]# ignore new line character
                elif ' ' in line:
                    # major axis radius, minor axis radius, angle, center x, center y
                    current_face_data = generate_arr_from(line[:-4])# ignore new line character and ' 1'

                    minor_axis = current_face_data[1]*math.cos(math.radians(current_face_data[2]))
                    xmins.append(current_face_data[3] - minor_axis)
                    xmaxs.append(current_face_data[3] + minor_axis)
                    major_axis = current_face_data[0]*math.cos(math.radians(current_face_data[0]))
                    ymins.append(current_face_data[4] - major_axis)
                    ymaxs.append(current_face_data[4] + major_axis)

                    remaining_faces -= 1
                elif int(line[0]):
                    remaining_faces = int(line)
                if remaining_faces is 0:
                    classStrings = ['face']*len(xmins)
                    classIDs = [1]*len(xmins)
                    tfRecord = create_TFRecord(images_dir, current_image_path + '.jpg', xmins, xmaxs, ymins, ymaxs, classStrings, classIDs)
                    writer.write(tfRecord.SerializeToString())

                    current_image_path = None
                    remaining_faces = None
                    xmins = []
                    xmaxs = []
                    ymins = []
                    ymaxs = []


    writer.close()

if __name__ == '__main__':
    tf.app.run()
