import segmentation.segmentation_utils as utils
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()


def prepare_dataset(config):
    images_folder = config['images_folder']

    annotation_directory = config['segmentation']['annotation_directory']
    csv_train_path = config['segmentation']['csv_train_path']
    csv_test_path = config['segmentation']['csv_test_path']
    test_size = config['segmentation']['test_size']

    os.makedirs(
        os.path.dirname(os.path.abspath(csv_train_path)), exist_ok=True)

    df = utils.get_df(annotation_directory).reset_index()

    data_bboxes = []
    for img_path, bboxes_sel, img_size in df[[
            'imagePath', 'bboxes_sel', 'img_size'
    ]].values:
        height, width = img_size
        full_path = os.path.join(images_folder, img_path)
        for class_, geom in bboxes_sel:
            xmin, ymin, xmax, ymax = geom.bounds
            data_bboxes.append((full_path, width, height, class_, int(xmin),
                                int(ymin), int(xmax), int(ymax)))

    df_bboxes = pd.DataFrame(
        data_bboxes,
        columns=[
            'filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax',
            'ymax'
        ])

    train, test = train_test_split(
        df_bboxes['filename'].unique(), test_size=test_size, random_state=42)

    df_bboxes.set_index('filename').loc[train].reset_index().to_csv(
        csv_train_path, index=False)
    df_bboxes.set_index('filename').loc[test].reset_index().to_csv(
        csv_test_path, index=False)
