import segmentation.segmentation_utils as utils
import pandas as pd
import numpy as np
import cv2
import os
from tqdm import tqdm
tqdm.pandas()


def create_masks(config):
    csv_train_path = config['segmentation']['csv_train_path']
    csv_test_path = config['segmentation']['csv_test_path']
    dhSegment_dir = config['segmentation']['dhSegment_dir']

    os.makedirs(dhSegment_dir, exist_ok=True)

    dhSegment_dir = os.path.abspath(dhSegment_dir)

    dhSegment_dir_train = os.path.join(dhSegment_dir, 'train')
    os.makedirs(dhSegment_dir_train, exist_ok=True)

    dhSegment_dir_test = os.path.join(dhSegment_dir, 'test')
    os.makedirs(dhSegment_dir_test, exist_ok=True)

    df_train = pd.read_csv(csv_train_path)
    df_test = pd.read_csv(csv_test_path)

    class2color = {
        'section_author': (255, 0, 0),
        'sale_description': (0, 128, 128)
    }

    def process_images(group, save_directory):
        filename, width, height = group.iloc[0][[
            'filename', 'width', 'height'
        ]].values
        mask = np.zeros((height, width, 3)).astype(np.uint8)
        for class_, xmin, ymin, xmax, ymax in group[[
                'class', 'xmin', 'ymin', 'xmax', 'ymax'
        ]].values:
            color = class2color[class_]
            mask[xmin:xmax, ymin:ymax, :] = color
        cv2.imwrite(
            os.path.join(
                save_directory,
                os.path.basename(filename).replace('.jpg', '') + '.png'),
            mask.transpose(1, 0, 2)[:, :, ::-1])

    print("### Processing training set ###")
    df_train.groupby('filename').progress_apply(
        lambda group: process_images(group, dhSegment_dir_train))
    print("### Processing testing set ###")
    df_test.groupby('filename').progress_apply(
        lambda group: process_images(group, dhSegment_dir_test))

    def normalize_filename(filename, dir):
        png_name = os.path.basename(filename).replace('.jpg', '') + '.png'
        return os.path.join(dir, png_name)

    X_train = df_train['filename'].unique()
    y_train = df_train['filename'].apply(
        lambda x: normalize_filename(x, dhSegment_dir_train)).unique()

    X_test = df_test['filename'].unique()
    y_test = df_test['filename'].apply(
        lambda x: normalize_filename(x, dhSegment_dir_test)).unique()

    utils.zip2csv(
        zip(X_train, y_train), os.path.join(dhSegment_dir, 'train.csv'))
    utils.zip2csv(
        zip(X_test, y_test), os.path.join(dhSegment_dir, 'test.csv'))

    with open(os.path.join(dhSegment_dir, 'classes.txt'), 'w') as outfile:
        outfile.write("%d %d %d\n" % class2color['section_author'])
        outfile.write("%d %d %d\n" % class2color['sale_description'])
        outfile.write("0 0 0\n")