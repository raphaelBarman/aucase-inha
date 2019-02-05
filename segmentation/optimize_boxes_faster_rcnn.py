import segmentation_utils as seg_utils
import post_processing_utils as pp_utils
from predict_utils import predict_faster_r_cnn
import metrics
import json
import os
from os import path
import pandas as pd
import numpy as np
from tqdm import tqdm
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def optimize_boxes_faster_rcnn(config):
    annotation_directory = config['segmentation']['annotation_directory']
    csv_test_path = config['segmentation']['csv_test_path']
    model_dir = config['segmentation']['model_dir']
    ocr_dir_google = config['segmentation']['ocr_dir_google']
    ocr_dir_inha = config['segmentation']['ocr_dir_inha']
    use_inha_ocr = config['segmentation']['use_inha_ocr']
    idx2cote_path = os.path.join(config['data']['output_dir'], 'idx2cote.json')
    output_dir = config['segmentation']['optimize_output_dir']
    restore_df = config['segmentation']['restore_df']
    compute_prediction_data = config['segmentation']['compute_prediction_data']

    prediction_data_path = 'segmentation_objects_description_pred_data.npy'

    os.makedirs(output_dir, exist_ok=True)

    with open(idx2cote_path, 'r') as infile:
        idx2cote = {int(k): v for k, v in json.load(infile).items()}

    if use_inha_ocr:
        ocr_dir = ocr_dir_inha
    else:
        ocr_dir = ocr_dir_google

    df_path = path.join(output_dir, 'df.pickle')

    if restore_df and path.exists(df_path):
        df = pd.read_pickle(df_path)
    else:
        df = seg_utils.get_df(annotation_directory)
        df.to_pickle(df_path)

    df_test = pd.read_csv(csv_test_path)
    test_images = df_test['filename'].values
    basename_test = df_test['filename'].apply(lambda x: path.basename(x))

    if compute_prediction_data:
        prediction_data = predict_faster_r_cnn(test_images, model_dir)
        np.save(
            path.join(output_dir, prediction_data_path), prediction_data)
    else:
        prediction_data = np.load(
            path.join(output_dir, prediction_data_path)).item(0)
        prediction_data = {
            path.basename(k).replace('.jpg', ''): v
            for k, v in prediction_data.items()}

    def get_score(proba_threshold_section=1.0,
                  line_padding_section=60,
                  complete_with_lines_section=0,
                  complete_with_bboxes_section=0,
                  proba_threshold_sale=0.93,
                  line_padding_sale=60,
                  complete_with_lines_sale=0,
                  complete_with_bboxes_sale=1):
        results = []
        for file_name in tqdm(basename_test, leave=False):
            gray_img = df.loc[file_name]['imageGray']
            bboxes_sel = pp_utils.get_bboxes_sel_fastrcnn(
                file_name,
                gray_img,
                prediction_data,
                proba_threshold_section=proba_threshold_section,
                line_padding_section=line_padding_section,
                complete_with_lines_section=complete_with_lines_section,
                complete_with_bboxes_section=complete_with_bboxes_section,
                proba_threshold_sale=proba_threshold_sale,
                line_padding_sale=line_padding_sale,
                complete_with_lines_sale=complete_with_lines_sale,
                complete_with_bboxes_sale=complete_with_bboxes_sale,
                ocr_dir=ocr_dir,
                use_inha_ocr=use_inha_ocr,
                idx2cote=idx2cote)
            if bboxes_sel is None:
                print("No boxes with", file_name)
                continue
            else:
                bboxes_sel = pp_utils.get_not_intersecting_with_class(
                    bboxes_sel)
            if len(bboxes_sel) > 0:
                results.append(
                    np.mean(
                        metrics.per_class_mAP(
                            bboxes_sel,
                            df.loc[file_name]['bboxes_sel'])))
            else:
                results.append(0)
        return np.mean(results)

    print(get_score(
        *[0.9532822389856309, 0, 0, 1, 0.7437949137209756, 60, 0, 1]))

    space = [
        Real(0.001, 1.0, name='proba_threshold_section'),
        Integer(0, 60, name='line_padding_section'),
        Integer(0, 1, name='complete_with_lines_section'),
        Integer(0, 1, name='complete_with_bboxes_section'),
        Real(0.001, 1.0, name='proba_threshold_sale'),
        Integer(0, 60, name='line_padding_sale'),
        Integer(0, 1, name='complete_with_lines_sale'),
        Integer(0, 1, name='complete_with_bboxes_sale')
    ]

    @use_named_args(space)
    def objective(**params):
        return -get_score(**params)
    reg_gp = gp_minimize(objective, space, n_calls=100, verbose=True)
    print(reg_gp.x)

    print(get_score(*(reg_gp.x)))
