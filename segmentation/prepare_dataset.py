from segmentation_utils import *
import os
import json
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()


annotation_directory = '/scratch/raphael/data/annotations/objects_description'
image_directory = '/scratch/raphael/data/drouot'
csv_train_path = './train.csv'
csv_test_path = './test.csv'
test_size = 0.2

image_directory = os.path.join(image_directory, '')

df = get_df(annotation_directory).reset_index()

data_bboxes = []
for img_path, bboxes_sel, img_size in df[['imagePath', 'bboxes_sel', 'img_size']].values:
    height, width = img_size
    full_path = image_directory + img_path
    for class_, geom in bboxes_sel:
        xmin, ymin, xmax, ymax = geom.bounds
        data_bboxes.append((full_path, width, height, class_, int(xmin), int(ymin), int(xmax), int(ymax)))

df_bboxes = pd.DataFrame(data_bboxes, columns=['filename','width','height','class','xmin','ymin','xmax','ymax'])

train, test = train_test_split(df_bboxes['filename'].unique(), test_size=test_size, random_state=42)

df_bboxes.set_index('filename').loc[train].reset_index().to_csv(csv_train_path, index=False)
df_bboxes.set_index('filename').loc[test].reset_index().to_csv(csv_test_path, index=False)