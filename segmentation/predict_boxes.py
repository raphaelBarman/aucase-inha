from segmentation_utils import *
from post_processing_utils import *
from predict_utils import *
import tensorflow as tf
import pandas as pd
import os
import json
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
output_dir = '/scratch/raphael/boxes_faster_rcnn/predictions'
force_refresh = False
model_dir_faster_r_cnn = '/scratch/raphael/object_detection/experiment/models/faster_rcnn_inception_resnet_v2_atrous_coco/frozen_model/frozen_inference_graph.pb'
model_dir_dh_segment = '/home/rbarman/models/segmentation_objects_description/model2/export/'
use_faster_r_cnn = True
image_dir = '/scratch/raphael/data/drouot'
prediction_csv = '/scratch/raphael/data/classified.csv'

ocr_dir_google = '/scratch/raphael/data/ocr/drouot_ocr_google'
ocr_dir_inha = '/scratch/raphael/data/ocr/ocr_inha'
use_inha_ocr = True
idx2cote_path = '../data/idx2cote.json'
tmp_preds = os.path.join(output_dir, 'preds_tmp')
os.makedirs(tmp_preds, exist_ok=True)

df_images = pd.read_csv(prediction_csv)
images_list = df_images[df_images['class'] == 'objects_description']['filename'].apply(lambda filename: os.path.join(image_dir, filename)).values

with open(idx2cote_path, 'r') as infile:
    idx2cote = {int(k) : v for k,v in json.load(infile).items()}

if use_faster_r_cnn:
    model_dir = model_dir_faster_r_cnn
else:
    model_dir = model_dir_dh_segment

if use_inha_ocr:
    ocr_dir = ocr_dir_inha
else:
    ocr_dir = ocr_dir_google

bboxes_data = {}
if use_faster_r_cnn:
    print("### Predicting images ###")
    prediction_data = predict_faster_r_cnn(images_list, model_dir, save=True, save_dir=tmp_preds)
    print("### Creating bboxes ###")
    for file_name in tqdm(images_list):
        basename = os.path.basename(file_name).split('.')[0].replace('.jpg', '')
        save_path = os.path.join(output_dir, basename + '_faster_r_cnn_boxes.npy')
        if not force_refresh and os.path.exists(save_path):
            bboxes_data[basename] = np.load(save_path)
            continue

        image = cv2.imread(file_name)
        gray_img = cvt2gray(image)
        bboxes_sel = get_bboxes_sel_fastrcnn(basename,
            gray_img,
            prediction_data,
            proba_threshold_section = 0.9532822389856309,
            line_padding_section = 0,
            complete_with_lines_section=0,
            complete_with_bboxes_section=1,
            proba_threshold_sale = 0.7437949137209756,
            line_padding_sale = 60,
            complete_with_lines_sale = 0,
            complete_with_bboxes_sale = 1,
            ocr_dir = ocr_dir,
            use_inha_ocr = use_inha_ocr,
            idx2cote = idx2cote)
        if bboxes_sel is not None:
            bboxes_sel = get_not_intersecting_with_class(bboxes_sel)
        else:
            print("emtpy boxes", file_name)
            bboxes_sel = np.array([])
        np.save(save_path, bboxes_sel)
        bboxes_data[basename] = bboxes_sel
else:
    print("### Predicting images ###")
    prediction_data = predict_dhSegment(images_list, model_dir, save=True, save_dir=tmp_preds)
    print("### Creating bboxes ###")
    for file_name in tqdm(images_list):
        basename = os.path.basename(file_name).split('.')[0].replace('.jpg', '')
        save_path = os.path.join(output_dir, basename + '_faster_r_cnn_boxes.npy')
        if not force_refresh and os.path.exists(save_path):
            bboxes_data[basename] = np.load(save_path)
            continue
        image = cv2.imread(file_name)
        gray_img = cvt2gray(image)
        bboxes_sel = get_bboxes_sel(basename,
            gray_img,
            prediction_data,
            bin_thresh_content = 0.7523492382365334, 
            bin_thresh_section = 0.5512835270136308, 
            ksize_w = 44, 
            ksize_h = 0, 
            line_padding = 10, 
            crop_padding = 50, 
            indent_thresh = 0.3221336755234053, 
            contour_area_thresh = 1000, 
            area_thresh = 934,
            ocr_dir = ocr_dir,
            use_inha_ocr = use_inha_ocr,
            idx2cote = idx2cote)
        if bboxes_sel is None:
            print("emtpy boxes", file_name)
            bboxes_sel = np.array([])
        np.save(save_path, bboxes_sel)
        bboxes_data[basename] = bboxes_sel

