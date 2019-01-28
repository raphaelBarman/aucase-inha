from segmentation_utils import *
from post_processing_utils import *
from predict_utils import predict_faster_r_cnn
from metrics import *
import json
import os
import pandas as pd
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

annotations_dir = '/scratch/raphael/data/annotations/objects_description/'
csv_test_path = './test.csv'
model_dir = '/scratch/raphael/object_detection/experiment/models/faster_rcnn_inception_resnet_v2_atrous_coco.bak/frozen_model/frozen_inference_graph.pb'
ocr_dir_google = '/scratch/raphael/data/ocr/drouot_ocr_google'
ocr_dir_inha = '/scratch/raphael/data/ocr/ocr_inha'
use_inha_ocr = True
idx2cote_path = '../data/idx2cote.json'
output_dir = '/scratch/raphael/boxes_dhsegment'
os.makedirs(output_dir, exist_ok=True)
restore_df = True
compute_prediction_data = False

with open(idx2cote_path, 'r') as infile:
    idx2cote = {int(k) : v for k,v in json.load(infile).items()}

if use_inha_ocr:
    ocr_dir = ocr_dir_inha
else:
    ocr_dir = ocr_dir_google

df_path = os.path.join(output_dir, 'df.pickle')

if restore_df and os.path.exists(df_path):
    df = pd.read_pickle(df_path)
else:
    df = get_df(annotations_dir)
    df.to_pickle(df_path)

df_test = pd.read_csv(csv_test_path)
test_images =  df_test['filename'].values
basename_test = df_test['filename'].apply(lambda x: os.path.basename(x))

if compute_prediction_data:
    prediction_data = predict_faster_r_cnn(test_images)
    np.save(os.path.join(output_dir, 'segmentation_objects_description_pred_data_fastrcnn.npy'), prediction_data)
else:
    prediction_data = np.load(os.path.join(output_dir, 'segmentation_objects_description_pred_data_fastrcnn.npy')).item(0)
    prediction_data = {os.path.basename(k).replace('.jpg', ''): v for k,v in prediction_data.items()}

def get_score(proba_threshold_section = 1.0,
                   line_padding_section = 60,
                   complete_with_lines_section=0,
                   complete_with_bboxes_section=0,
                   proba_threshold_sale = 0.93,
                   line_padding_sale = 60,
                   complete_with_lines_sale=0,
                   complete_with_bboxes_sale=1):
    results = []
    for file_name in tqdm(basename_test, leave=False):
        gray_img = df.loc[file_name]['imageGray']
        bboxes_sel = get_bboxes_sel_fastrcnn(file_name,
                       gray_img,
                       prediction_data,
                       proba_threshold_section = proba_threshold_section,
                       line_padding_section = line_padding_section,
                       complete_with_lines_section=complete_with_lines_section,
                       complete_with_bboxes_section=complete_with_bboxes_section,
                       proba_threshold_sale = proba_threshold_sale,
                       line_padding_sale = line_padding_sale,
                       complete_with_lines_sale=complete_with_lines_sale,
                       complete_with_bboxes_sale=complete_with_bboxes_sale,
                       ocr_dir = ocr_dir,
                       use_inha_ocr = use_inha_ocr,
                       idx2cote = idx2cote)
        if bboxes_sel is None:
            print("No boxes with", file_name)
            continue
        else:
            bboxes_sel = get_not_intersecting_with_class(bboxes_sel)
        if len(bboxes_sel) > 0:
            results.append(np.mean(per_class_mAP(bboxes_sel, df.loc[file_name]['bboxes_sel'])))
        else:
            results.append(0)
    return np.mean(results)

print(get_score(*[0.9532822389856309, 0, 0, 1, 0.7437949137209756, 60, 0, 1]))

from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize

space = [
   Real(0.001, 1.0, name='proba_threshold_section'),
   Integer(0, 60, name='line_padding_section'),
   Integer(0,1, name='complete_with_lines_section'),
   Integer(0,1, name='complete_with_bboxes_section'),
   Real(0.001, 1.0, name='proba_threshold_sale'),
   Integer(0, 60, name='line_padding_sale'),
   Integer(0,1, name='complete_with_lines_sale'),
   Integer(0,1, name='complete_with_bboxes_sale')
]

@use_named_args(space)
def objective(**params):
   return -get_score(**params)
reg_gp = gp_minimize(objective, space, n_calls=100 , verbose=True)
print(reg_gp.x)

print(get_score(*(reg_gp.x)))