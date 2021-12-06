import sys
sys.path.append('./models/face_detect_dsfd')
import cv2
import numpy as np
import pandas as pd
from models.face_detect_dsfd.data import *
from models.face_detect_dsfd.face_ssd import build_ssd
from models.face_detect_dsfd.demo_torch140 import test_oneimage_modify
import torch
from models.mask_classify_vinai.demo import classify
from tensorflow.keras.models import load_model
from models.face_detect_dsfd.demo_torch140 import infer,infer_flip,infer_multi_scale_sfd
from models.face_detect_dsfd.widerface_val_torch140 import bbox_vote


df = pd.read_csv('../data_zalo/train/train_meta.csv')
df = df[['fname','mask']]
df_mask = df.dropna()
label = df_mask['mask'].values
all_name = df_mask['fname'].values
root_data = '../data_zalo/train/images'


trained_model = 'models/face_detect_dsfd/weights/WIDERFace_DSFD_RES152.pth'

cfg = widerface_640
num_classes = len(WIDERFace_CLASSES) + 1 # +1 background
net = build_ssd('test', cfg['min_dim'], num_classes) # initialize SSD

net.load_state_dict(torch.load(trained_model, map_location='cpu'))
net.eval()
print('Finished loading face detect model!')


transform = TestBaseTransform((104, 117, 123))
thresh = cfg['conf_thresh']
cuda = False







y_pred = []
for idx,name in enumerate(all_name[:3]):
    lb_init = 1
    path_img = os.path.join(root_data,name)
    print(f'path_img : {path_img}')
    img = cv2.imread(path_img, cv2.IMREAD_COLOR)
    with torch.no_grad():
        #im,dets = test_oneimage_modify(net,path_img)
        max_im_shrink = ((2000.0 * 2000.0) / (img.shape[0] * img.shape[1])) ** 0.5
        shrink = max_im_shrink if max_im_shrink < 1 else 1

        det0 = infer(net, img, transform, thresh, cuda, shrink)
        det1 = infer_flip(net, img, transform, thresh, cuda, shrink)
        # shrink detecting and shrink only detect big face
        st = 0.5 if max_im_shrink >= 0.75 else 0.5 * max_im_shrink
        det_s = infer(net, img, transform, thresh, cuda, st)
        index = np.where(np.maximum(det_s[:, 2] - det_s[:, 0] + 1, det_s[:, 3] - det_s[:, 1] + 1) > 30)[0]
        det_s = det_s[index, :]
        # enlarge one times
        factor = 2
        bt = min(factor, max_im_shrink) if max_im_shrink > 1 else (st + max_im_shrink) / 2
        det_b = infer(net, img, transform, thresh, cuda, bt)
        # enlarge small iamge x times for small face
        if max_im_shrink > factor:
            bt *= factor
            while bt < max_im_shrink:
                det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, bt)))
                bt *= factor
            det_b = np.row_stack((det_b, infer(net, img, transform, thresh, cuda, max_im_shrink)))
        # enlarge only detect small face
        if bt > 1:
            index = np.where(np.minimum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) < 100)[0]
            det_b = det_b[index, :]
        else:
            index = np.where(np.maximum(det_b[:, 2] - det_b[:, 0] + 1, det_b[:, 3] - det_b[:, 1] + 1) > 30)[0]
            det_b = det_b[index, :]
        det = np.row_stack((det0, det1, det_s, det_b))
        dets = bbox_vote(det)
        inds = np.where(dets[:, -1] >= 0.5)[0]
        if len(inds) == 0:
            y_pred.append(0.)
        for i in inds:
            bbox = [int(itm) for itm in dets[i, :4]]
            score = dets[i, -1]
            # img_arr = im[bbox[0]]
            # cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255))
            if all(vl >= 0 for vl in bbox):
                img_arr = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
                img_arr = img_arr[:, :, ::-1]
                pred_lb = classify(img_arr=img_arr)[0]
                if pred_lb == 1:
                    y_pred.append(0.)
                    lb_init = 0
                    break
    if lb_init:
        y_pred.append(1.)
print(f'y_pred: {y_pred}')












