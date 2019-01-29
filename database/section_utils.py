import numpy as np
import cv2
import pandas as pd
import os
from anytree import AnyNode

def get_height_infos(page_df, image_dir):
    global height_infos
    basename, scale_x, scale_y = page_df.iloc[0][['basename', 'scale_x', 'scale_y']]
    image = cv2.imread(os.path.join(image_dir, basename + '.jpg'))
    infos = []  
    for doc, page, entity, words_ref in page_df['words_ref'].reset_index().values:
        coords = (np.stack((words_ref[:,0])).reshape(-1,2) * [scale_x, scale_y]).astype(int)
        minx, miny = coords.min(axis=0)
        maxx, maxy = coords.max(axis=0)
        gray = cv2.cvtColor(image[miny:maxy, minx:maxx],cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        cnt, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        infos.append((doc, page, entity, [(cv2.contourArea(contour),
                              cv2.boundingRect(contour)[-1])
                             for contour in sorted(contours, key=cv2.contourArea)[::-1]]))
    infos = pd.DataFrame(infos, columns=['doc', 'page', 'entity', 'height_infos']).set_index(['doc', 'page', 'entity'])
    return page_df.join(infos)

def get_section_hierarchy(df_sections):
    df_sections = df_sections.copy()
    df_sections['mean_height'] = df_sections['height_infos'].apply(lambda x: np.mean(np.array(x)[:,1][:5]))
    df_sections = df_sections.join(df_sections[df_sections['class'] == 'author'].groupby('doc')['mean_height'].mean().to_frame('mean_author_height'))
    root = AnyNode(id='root')
    def add_entity(root, doc, page, entity, text, class_, mean_true_height):
        return AnyNode(id= page*1000000+entity, parent=root, doc=doc, page=page, entity=entity, text=text, class_=class_, mean_true_height=mean_true_height)

    def process_doc(doc):
        document = AnyNode(id=doc.reset_index()['doc'].iloc[0], parent=root)
        prev_root = document
        prev_entity = document
        for doc, page, entity, text, class_, mean_true_height, mean_author_height in doc[['text', 'class', 'mean_height', 'mean_author_height']].reset_index().values:
            mean_author_diff = np.abs(mean_true_height-mean_author_height)
            if class_ == 'category' and mean_author_diff >= 2:
                prev_root = add_entity(document, doc, page, entity, text, class_, mean_true_height)
            elif class_ in ['author', 'category', 'ecole']:
                prev_entity = add_entity(prev_root, doc, page, entity, text, class_, mean_true_height)
    df_sections.groupby('doc').progress_apply(process_doc)
    return root