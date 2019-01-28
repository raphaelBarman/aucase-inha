import numpy as np
import cv2
import base64
from glob import glob
import json
from PIL import Image
import shapely.geometry as geometry
import io
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

def load_image(img_base64):
    nparr = np.frombuffer(base64.b64decode(img_base64), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def cvt2gray(image):
    return 255-cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def get_bbox(image, shape):
    label = shape['label']
    mask_content = np.zeros(image.shape, dtype=image.dtype)
    cv2.fillConvexPoly(mask_content, np.array(shape['points']), 255)

    img = cv2.bitwise_and(mask_content, image)

    if label == 'annotation':
        ret,thresh = cv2.threshold(img,200,255,cv2.THRESH_BINARY)
    else:
        ret,thresh = cv2.threshold(img,135,255,cv2.THRESH_BINARY)
    cnt, _, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return (label, cv2.boundingRect(cnt))

def get_bboxes(row):
    image = row['imageGray'].copy()
    shapes = row['shapes']
    bboxes = []
    for shape in shapes:
        bboxes.append(get_bbox(image, shape))
    return bboxes

def zip2csv(zip_csv, filepath):
    with open(filepath, 'w') as outfile:
        [outfile.write("%s,%s\n"%x) for x in zip_csv]

def get_df(json_path):
    jsons = glob(os.path.join(json_path, '*.json'))
    
    print("### Loading jsons ###")
    jsons_data = []
    for json_file in tqdm(jsons):
        with open(json_file, 'r') as infile:
            jsons_data.append(json.load(infile))

    df = pd.DataFrame(jsons_data)
    df = df[['imagePath', 'imageData', 'shapes']]
    df['imagePath'][~df['imagePath'].str.startswith('0')] = df['imagePath'][~df['imagePath'].str.startswith('0')].str.split('\\').apply(lambda x: x[-1]).str.split('/').apply(lambda x: x[-1])
    df['imagePath'][~df['imagePath'].str.endswith('.jpg')] = df['imagePath'][~df['imagePath'].str.endswith('.jpg')].apply(lambda x: x+'.jpg')
    print("### Loading images ###")
    df['imageData'] = df['imageData'].progress_apply(lambda imgdata : Image.open(io.BytesIO(base64.b64decode(imgdata))))
    print("### converting to grayscale ###")
    df['imageGray'] = df['imageData'].progress_apply(lambda imgdata: 255-cv2.cvtColor(np.array(imgdata), cv2.COLOR_BGR2GRAY))
    df['img_size'] = df['imageData'].apply(lambda img: img.size)
    print("### Creating bounding boxes ###")
    df['bounding_boxes'] = df.progress_apply(get_bboxes, axis=1)
    print("### Creating ground truth ###")
    df['bboxes_sel'] = df.progress_apply(get_bboxes_sel, axis=1)
    df.set_index('imagePath', inplace=True)
    return df

def get_bboxes_sel(row):
    img_path = row['imagePath']
    bounding_boxes = row['bounding_boxes']
    img_size = row['img_size']
    shapes = row['shapes']
    bbox_sel = []

    sale_descriptions_bboxes = []
    object_bboxes = []
    for bbox in bounding_boxes:
        class_, (x,y,w,h) = bbox
        box = geometry.box(x,y,x+w,y+h)
        if class_ == 'sale_description':
            sale_descriptions_bboxes.append(box)
        elif class_.startswith('section'):
            bbox_sel.append(('section_author', box))
        elif class_.startswith('object'):
            object_bboxes.append(box)

    for bbox in sale_descriptions_bboxes:
        xs = []
        ys = []
        x,y = bbox.exterior.coords.xy
        xs.extend(list(x))
        ys.extend(list(y))
        for bbox_other in object_bboxes:
            if bbox.intersects(bbox_other):
                x,y = bbox_other.exterior.coords.xy
                xs.extend(list(x))
                ys.extend(list(y))
        minx = int(min(xs))
        miny = int(min(ys))
        maxx = int(max(xs))
        maxy = int(max(ys))
        w = maxx-minx
        h = maxy-miny
        bbox_sel.append(('sale_description', geometry.box(int(minx), int(miny), int(minx)+int(w), int(miny)+int(h))))
    return np.array(bbox_sel)

