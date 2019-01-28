from segmentation_utils import *
from post_processing_utils import *
from metrics import *
from predict_utils import predict_dhSegment
import json
import os
import pandas as pd
from dh_segment.loader import LoadedModel

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

annotations_dir = '/scratch/raphael/data/annotations/objects_description/'
csv_test_path = './test.csv'
model_dir = '/home/rbarman/models/segmentation_objects_description/model2/export/'
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
    prediction_data = predict_dhSegment(test_images)
    np.save(os.path.join(output_dir, 'segmentation_objects_description_pred_data.npy', prediction_data))
else:
    prediction_data = np.load(os.path.join(output_dir, 'segmentation_objects_description_pred_data.npy')).item(0)

def get_score(bin_thresh_content = 0.8,
              bin_thresh_section = 0.2,
              ksize_w = 0,
              ksize_h = 0,
              line_padding = 18,
              crop_padding = 14,
              indent_thresh = 0.1,
              contour_area_thresh = 200,
              area_thresh = 500):
    results = []
    for file_name in tqdm(basename_test, leave=False):
        gray_img = df.loc[file_name]['imageGray']
        bboxes_sel = get_bboxes_sel(file_name,
                       gray_img,
                       prediction_data,
                       bin_thresh_content = bin_thresh_content,
                       bin_thresh_section = bin_thresh_section,
                       ksize_w = ksize_w,
                       ksize_h = ksize_h,
                       line_padding = line_padding,
                       crop_padding = crop_padding,
                       indent_thresh = indent_thresh,
                       contour_area_thresh = contour_area_thresh,
                       area_thresh = area_thresh,
                       ocr_dir = ocr_dir,
                       use_inha_ocr = use_inha_ocr,
                       idx2cote = idx2cote)
        if bboxes_sel is None:
            print("Empty pred", file_name)
            continue
        if len(bboxes_sel) > 0:
            results.append(np.mean(per_class_mAP(bboxes_sel, df.loc[file_name]['bboxes_sel'])))
        else:
            results.append(0)
    return np.mean(results)

print(get_score(*[0.7523492382365334, 0.5512835270136308, 44, 0, 10, 50, 0.3221336755234053, 1000, 934]))

from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
from skopt import gp_minimize
space = [
   Real(0.5, 1.0, name='bin_thresh_content'),
   Real(0.1, 1.0, name='bin_thresh_section'),
   Integer(20, 60, name='ksize_w'),
   Integer(0, 5, name='ksize_h'),
   Integer(0, 30, name='line_padding'),
   Integer(20, 100, name='crop_padding'),
   Real(0.05, 1.0, name='indent_thresh'),
   Integer(0, 10000, name='contour_area_thresh'),
   Integer(0, 15000, name='area_thresh'),
]

@use_named_args(space)
def objective(**params):
   return -get_score(**params)
reg_gp = gp_minimize(objective, space, n_calls=100 , verbose=True,  random_state=42)
print(reg_gp.x)
np.save("best_dh.npy", np.array(reg_gp.x))
print(get_score(*(reg_gp.x)))