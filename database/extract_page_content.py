import extraction_utils as extr_utils
import numbers_utils as num_utils
import section_utils as sec_utils
import shapely.affinity as affinity
import cv2
import pandas as pd
from unidecode import unidecode
from glob import glob
import os
import numpy as np
import json
from tqdm import tqdm
tqdm.pandas()


def extract_page_content(config):
    image_dir = config['images_folder']
    output_dir = config['database']['output_dir']
    force_refresh = config['database']['force_refresh']
    boxes_prediction_path = config['segmentation']['output_dir']
    ocr_dir = config['database']['ocr_dir']
    section_classified_csv_path = config['database'][
        'section_classified_csv_path']
    metadata_path = config['data']['output_dir']

    idx2cote_path = os.path.join(metadata_path, 'idx2cote.json')
    idx2inha_path = os.path.join(metadata_path, 'idx2inhaIdx.json')

    page_content_tmp = os.path.join(output_dir, 'tmp_content')
    os.makedirs(page_content_tmp, exist_ok=True)

    with open(idx2cote_path, 'r') as infile:
        idx2cote = {int(k): v for k, v in json.load(infile).items()}

    with open(idx2inha_path, 'r') as infile:
        idx2inha = {int(k): int(v) for k, v in json.load(infile).items()}

    boxes_files = glob(os.path.join(boxes_prediction_path, '*.npy'))

    print("### Extracting page content ###")
    page_contents = []
    for boxes_file in tqdm(boxes_files):
        basename = "_".join(os.path.basename(boxes_file).split('_')[:2])
        save_path = os.path.join(page_content_tmp,
                                 basename + '_page_content.npy')
        if not force_refresh and os.path.exists(save_path):
            page_content, scale_x, scale_y = np.load(save_path)
            page_content = np.array(page_content)
        else:
            boxes = np.load(boxes_file)
            boxes = np.array(
                [(x[0], affinity.scale(x[1], xfact=2.0)) for x in boxes])
            image = cv2.imread(os.path.join(image_dir, basename + '.jpg'))
            height, width, _ = image.shape
            ocr_bboxes, scale_x, scale_y = extr_utils.get_google_ocr_bboxes(
                basename, width, height, ocr_dir, idx2cote)
            page_content = extr_utils.page_content_from_boxes(
                boxes, ocr_bboxes, filtered=True, image=image)
            np.save(save_path, (page_content, scale_x, scale_y))
        doc_id = int(basename.split('_')[0])
        page_id = int(basename.split('_')[1].rstrip('lr'))
        if 'r' in basename:
            page_id += 0.75
        if 'l' in basename:
            page_id += 0.25
        for entity in page_content:
            (entity_type, entity_geom), entity_words = entity
            xmin, ymin, xmax, ymax = extr_utils.get_words_bbox(
                entity_words, scale_x, scale_y)
            words_ref = extr_utils.words2words_ref(entity_words)
            text = "".join(extr_utils.words2words_ref(entity_words)[:, 1])
            page_contents.append(
                (doc_id, page_id, entity_type, text, basename, xmin, ymin,
                 xmax, ymax, words_ref, scale_x, scale_y))
    df_page_content = pd.DataFrame(
        page_contents,
        columns=[
            'doc', 'page', 'entity_type', 'text', 'basename', 'xmin', 'ymin',
            'xmax', 'ymax', 'words_ref', 'scale_x', 'scale_y'
        ])
    df_page_content = df_page_content.set_index(
        ['doc', 'page']).sort_values(by=['doc', 'page', 'ymin'])
    df_page_content.set_index(
        df_page_content.groupby(level=[0, 1]).cumcount(),
        append=True,
        inplace=True)
    df_page_content.index.set_names('entity', level=2, inplace=True)
    df_page_content['inha_ref'] = df_page_content['basename'].apply(
        lambda basename: extr_utils.internal2inha(basename, idx2cote))
    df_page_content['image_url'] = df_page_content['basename'].apply(
        lambda basename: extr_utils.internal2url(basename, idx2inha))

    print("### Correcting sale references ###")
    df_sale = df_page_content[df_page_content['entity_type'] ==
                              'sale_description'].copy()
    df_sale['num'] = df_sale['text'].apply(num_utils.parse_num)
    df_sale = df_sale.groupby(level=0).progress_apply(
        num_utils.complete_document)

    print("### Creating section hiearchy ###")
    df_sections = df_page_content[df_page_content['entity_type'] ==
                                  'section_author'].copy()
    df_sections = df_sections.groupby(['doc', 'page']).progress_apply(
        lambda group: sec_utils.get_height_infos(group, image_dir))
    df_sections_classified = pd.read_csv(section_classified_csv_path)
    df_sections = df_sections.join(
        df_sections['text'].str.strip().str.lower().apply(
            unidecode).reset_index().set_index('text').join(
                df_sections_classified.set_index('text')).reset_index(
                    drop=True).set_index(['doc', 'page', 'entity']))

    print("### Creating section hierachy tree ###")
    section_hierarchy = sec_utils.get_section_hierarchy(df_sections)

    df_page_content = pd.concat([df_sale, df_sections], sort=True).sort_index()
    df_page_content['entity_type'] = df_page_content['entity_type'].apply({
        'sale_description':
        'object',
        'section_author':
        'section'
    }.get)
    df_page_content.to_pickle(os.path.join(output_dir, 'df_page_content.pkl'))

    np.save(
        os.path.join(output_dir, 'section_hierarchy.npy'), section_hierarchy)
