import sys
sys.path.append('./models/mask_classify_vinai')
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import resnet50, inception_resnet_v2
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.compat.v1.keras.backend import set_session
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from matplotlib import pyplot as plt
import io
from PIL import Image as pil_image
import argparse





parser = argparse.ArgumentParser(description='MaskClassify')
parser.add_argument('--trained_model', default='weights/resnet50.h5',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--img_root',type=str,help='Path to test img')
args = parser.parse_args()





config = tf.ConfigProto()
# dynamically grow the memory used on the GPU
config.gpu_options.allow_growth = True
# to log device placement (on which device the operation ran)
config.log_device_placement = True
sess = tf.Session(config=config)
# set this TensorFlow session as the default session for Keras
set_session(sess)


model = load_model('models/mask_classify_vinai/weights/resnet50.h5')
model_type = 'resnet50'
csize = 224

def classify(img_path='', img_arr=None, thresh=0.5):
    if img_arr is None:
        img_arr = img_to_array(
            load_img(img_path, target_size=(csize, csize))
        )
    else:
        img_arr = cv2.resize(img_arr, (csize, csize), interpolation=cv2.INTER_NEAREST)

    img_arr = np.expand_dims(img_arr, axis=0)

    if model_type == 'resnet50':
        img_arr = resnet50.preprocess_input(img_arr)
    else:
        img_arr = inception_resnet_v2.preprocess_input(img_arr)

    match = model.predict(img_arr)
    return (1 if match[0][0] > thresh else 0, match[0][0])



if __name__ == '__main__':
    rs = classify(img_path=args.img_root)
    if rs[0]:
        print('No mask')
    else:
        print('Have mask')


