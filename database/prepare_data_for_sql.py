from sql_utils import Sql_utils
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
tqdm.pandas()
import os
from anytree.search import findall, findall_by_attr

section_hierarchy_path = '/scratch/raphael/page_content/section_hierarchy.npy'
df_page_content_path = '/scratch/raphael/page_content/df_page_content.pkl'
actor_excel_path = './actors.xlsx'
image_dir = '/scratch/raphael/data/drouot'
idx2cote_path = '/home/rbarman/sacase-inha/data/idx2cote.json'
idx2inha_path = '/home/rbarman/sacase-inha/data/idx2inhaIdx.json'
inha_bib_num_metadata_path = '/home/rbarman/sacase-inha/data/inha_bib_num_metadata.json'
iiif_manifests_path = '/home/rbarman/sacase-inha/data/iiif_manifests.json'

print("### Loading data utils ###")
sql_utils = Sql_utils(idx2cote_path, idx2inha_path, inha_bib_num_metadata_path, iiif_manifests_path)

print("### Actor table ###")
# Actor Table
actors = pd.read_excel(actor_excel_path).set_index('raw_name')
actors = actors[actors['last_name'] != 'Drouot']
actors.drop(columns=['name'], inplace=True)
actors.fillna("", inplace=True)

no_dup_actors = actors.drop_duplicates(subset=['last_name', 'first_name', 'role'])
no_dup_actors = no_dup_actors[no_dup_actors['last_name'] != 'Drouot']
no_dup_actors = no_dup_actors.reset_index().set_index(['last_name', 'first_name', 'role']).sort_index()
no_dup_actors['actor_id'] = np.arange(len(no_dup_actors))
no_dup_actors.sort_index(inplace=True)

actors_with_id = (actors.reset_index().set_index(['last_name', 'first_name', 'role']).sort_index().join(no_dup_actors[['actor_id']])).reset_index()

actor_raw2id = lambda t: actors_with_id.replace("", np.nan).set_index('raw_name')['actor_id'].to_dict().get(t) if t != 'Drouot' else np.nan

actor_table = no_dup_actors.reset_index()[['actor_id', 'last_name', 'first_name', 'role']]

print("### Sale table ###")

# Sale table
images = glob(os.path.join(image_dir, '*.jpg'))
sale_table = pd.Series(list(set([int(os.path.basename(x).split('_')[0]) for x in images]))).to_frame('sale_id')
sale_table['date'] = sale_table['sale_id'].apply(sql_utils.idx2date_formatted)
sale_table['cote_inha'] = sale_table['sale_id'].apply(sql_utils.idx2cote.get)
sale_table['url_inha'] = sale_table['sale_id'].apply(sql_utils.idx2base_url)

print("### Actor Sale table ###")

# Actor Sale table

table_actor_sale = []
for doc_id in tqdm(sale_table['sale_id'].values):
    for actor in sql_utils.idx2actors(doc_id):
        if actor != 'Drouot':
            table_actor_sale.append((actor_raw2id(actor), doc_id))
table_actor_sale = pd.DataFrame(table_actor_sale, columns=['actor_id', 'doc_id'])

# Loading 
section_hierarchy = np.load(section_hierarchy_path).item(0)
df_page_content = pd.read_pickle(df_page_content_path)

print("### Section table ###")

# Section table
df_sections = df_page_content[df_page_content['entity_type'] == 'section'].copy()

hierarchy_documents = list(findall(section_hierarchy, filter_=lambda n: n.depth == 1))
sections_table = []
for document in tqdm(hierarchy_documents):
    for supersection in document.children:
        sections_table.append(sql_utils.get_section_infos(supersection, df_sections))
        for section in supersection.children:
            sections_table.append(sql_utils.get_section_infos(section, df_sections, parent=supersection))

section_table = pd.DataFrame(sections_table, columns = ['sale_id', 'page', 'num_entity',  'parent_section_sale', 'parent_section_page', 'parent_section_entity', 'class', 'text', 'bbox', 'inha_url', 'iiif_url'])

print("### Object table table ###")

# Sale table

df_sale = df_page_content[df_page_content['entity_type'] == 'object'].copy()

doc2node = {node.id:node.descendants for node in section_hierarchy.children}
def get_max_smallest(row):
    doc, page, entity = row.values
    if doc not in doc2node:
        return None
    nodes = list(doc2node[doc])
    id_ = page*1000000 + entity
    best_node = None
    biggest = -1
    for node in nodes:
        if node.id < id_:
            if node.id > biggest:
                biggest = node.id
                best_node = node
        else:
            break
    return best_node
df_sale['parent_section'] = df_sale.reset_index()[['doc', 'page', 'entity']].progress_apply(get_max_smallest, axis=1).values
df_sale['bbox'] = df_sale[['xmin', 'ymin', 'xmax', 'ymax']].progress_apply(lambda row: tuple(row.values.astype(int).tolist()), axis=1)
df_sale['parent_section_sale'] = df_sale['parent_section'].apply(lambda section: int(section.doc) if section is not None else None)
df_sale['parent_section_page'] = df_sale['parent_section'].apply(lambda section: section.page if section is not None else None)
df_sale['parent_section_entity'] = df_sale['parent_section'].apply(lambda section: int(section.entity) if section is not None else None)
df_sale['iif_url'] = df_sale['basename'].apply(sql_utils.basename2iiifbase)