import segmentation_utils as seg_utils
import post_processing_utils as pp_utils
import predict_utils as pred_utils
import pandas as pd
import os
import json
import cv2
from tqdm import tqdm
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def predict_boxes(config):
    output_dir = config['segmentation']['output_dir']
    force_refresh = config['segmentation']['force_refresh']
    model_dir = config['segmentation']['model_dir']
    use_faster_rcnn = config['segmentation']['use_faster_rcnn']
    image_dir = config['images_folder']
    prediction_csv = config['classification']['predicted_classes_csv']
    ocr_dir_google = config['segmentation']['ocr_dir_google']
    ocr_dir_inha = config['segmentation']['ocr_dir_inha']
    use_inha_ocr = config['segmentation']['use_inha_ocr']
    idx2cote_path = os.path.join(config['data']['output_dir'], 'idx2cote.json')

    tmp_preds = os.path.join(output_dir, 'preds_tmp')

    os.makedirs(tmp_preds, exist_ok=True)
    df_images = pd.read_csv(prediction_csv)
    images_list = df_images[
        df_images['class'] == 'objects_description']['filename'].apply(
            lambda filename: os.path.join(image_dir, filename)).values

    with open(idx2cote_path, 'r') as infile:
        idx2cote = {int(k): v for k, v in json.load(infile).items()}

    if use_inha_ocr:
        ocr_dir = ocr_dir_inha
    else:
        ocr_dir = ocr_dir_google

    bboxes_data = {}
    if use_faster_rcnn:
        print("### Predicting images ###")
        prediction_data = pred_utils.predict_faster_r_cnn(
            images_list, model_dir, save=True, save_dir=tmp_preds)
        print("### Creating bboxes ###")
        for file_name in tqdm(images_list):
            basename = os.path.basename(file_name).split('.')[0].replace(
                '.jpg', '')
            save_path = os.path.join(output_dir,
                                     basename + '_faster_r_cnn_boxes.npy')
            if not force_refresh and os.path.exists(save_path):
                bboxes_data[basename] = np.load(save_path)
                continue

            image = cv2.imread(file_name)
            gray_img = seg_utils.cvt2gray(image)
            bboxes_sel = pp_utils.get_bboxes_sel_fastrcnn(
                basename,
                gray_img,
                prediction_data,
                proba_threshold_section=0.9532822389856309,
                line_padding_section=0,
                complete_with_lines_section=0,
                complete_with_bboxes_section=1,
                proba_threshold_sale=0.7437949137209756,
                line_padding_sale=60,
                complete_with_lines_sale=0,
                complete_with_bboxes_sale=1,
                ocr_dir=ocr_dir,
                use_inha_ocr=use_inha_ocr,
                idx2cote=idx2cote)
            if bboxes_sel is not None:
                bboxes_sel = pp_utils.get_not_intersecting_with_class(
                    bboxes_sel)
            else:
                print("emtpy boxes", file_name)
                bboxes_sel = np.array([])
            np.save(save_path, bboxes_sel)
            bboxes_data[basename] = bboxes_sel
    else:
        print("### Predicting images ###")
        prediction_data = pred_utils.predict_dhSegment(
            images_list, model_dir, save=True, save_dir=tmp_preds)
        print("### Creating bboxes ###")
        for file_name in tqdm(images_list):
            basename = os.path.basename(file_name).split('.')[0].replace(
                '.jpg', '')
            save_path = os.path.join(output_dir,
                                     basename + '_faster_r_cnn_boxes.npy')
            if not force_refresh and os.path.exists(save_path):
                bboxes_data[basename] = np.load(save_path)
                continue
            image = cv2.imread(file_name)
            gray_img = seg_utils.cvt2gray(image)
            bboxes_sel = pp_utils.get_bboxes_sel(
                basename,
                gray_img,
                prediction_data,
                bin_thresh_content=0.7523492382365334,
                bin_thresh_section=0.5512835270136308,
                ksize_w=44,
                ksize_h=0,
                line_padding=10,
                crop_padding=50,
                indent_thresh=0.3221336755234053,
                contour_area_thresh=1000,
                area_thresh=934,
                ocr_dir=ocr_dir,
                use_inha_ocr=use_inha_ocr,
                idx2cote=idx2cote)
            if bboxes_sel is None:
                print("emtpy boxes", file_name)
                bboxes_sel = np.array([])
            np.save(save_path, bboxes_sel)
            bboxes_data[basename] = bboxes_sel
