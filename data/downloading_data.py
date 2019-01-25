from download_utils import *
import pandas as pd
from multiprocessing import Pool
from operator import itemgetter

download_folder = './drouot_39-45'

# downloading or loading the metadata from the OAI-PMH
inha_bib_num_metadata = download_oai_dc('inha_bib_num_metadata.json')

# Create a dataframe with the inha ids
df_metadata = pd.DataFrame.from_dict(inha_bib_num_metadata)
df_metadata['id'] = pd.to_numeric(df_metadata.identifier.apply(itemgetter(0)).str.split('/').apply(itemgetter(-1)))

# Creating dictionaries from internal ids to inha ids
idx2inha = {x: y for x,y in df_metadata.reset_index()[['index', 'id']].values}
inha2idx = {v: k for k,v in idx2inha.items()}
with open('idx2inhaIdx.json', 'w', encoding='utf-8') as outfile:
    json.dump({str(k): str(v) for k,v in idx2inha.items()}, outfile)

# Create the urls of the iiif manifest
df_metadata['iiif_manifest_urls'] = df_metadata['id'].apply(lambda id: "http://bibliotheque-numerique.inha.fr/iiif/%s/manifest"%id)

# Download the iiif manifests
manifests_json, failed_manifests = download_iiif_manifests(df_metadata['iiif_manifest_urls'].values, "iiif_manifests.json")
print('There were %d manifest that failed to download'%len(failed_manifests))
# Add to the dataframe the raw manifests and the list of urls of the images
# !!! Dependent on the format of the iiif manifest of the bib num, may break...
manifest_dict = {}
for manifest in manifests_json.values():
    id_ = int(manifest['@id'].replace('/manifest', '').split('/')[-1])
    manifest_dict[inha2idx[id_]] = manifest
images_dict = {}
for idx, record in manifest_dict.items():
    for image in record['sequences'][0]['canvases']:
        if not idx in images_dict:
            images_dict[idx] = []
        images_dict[idx].append(image['images'][0]['resource']['@id'])
images = [x for y in images_dict.values() for x in y]
df_metadata = df_metadata.join(pd.DataFrame.from_dict({k: [v] for k,v in manifest_dict.items()}, orient='index', columns=['manifest_raw']))
df_metadata = df_metadata.join(pd.DataFrame.from_dict({k: [v] for k,v in images_dict.items()}, orient='index', columns=['images']))

# Finding only the indexes of catalogs made by Drouot between 1939 and 1945
drouot_idxs = df_metadata[df_metadata.creator.notnull()][df_metadata[df_metadata.creator.notnull()].creator.apply(lambda x: 'Drouot' in x)].index
dates = pd.to_datetime(df_metadata[df_metadata.date.apply(len) == 3][df_metadata[df_metadata.date.apply(len) == 3].date.apply(itemgetter(1)).str.startswith("19")].date.apply(itemgetter(1)))
period_idxs = dates[(dates > '1939-01-01') & (dates < '1945-12-31')].index
not_null_idxs = df_metadata[df_metadata.images.notnull()].index

# Creating a download dictionary containing the filename as a key and the url as value
if not os.path.exists(download_folder):
    os.makedirs(download_folder, exist_ok = True)
download_dict = {}
for catalogue_idx, images in df_metadata.loc[drouot_idxs.intersection(period_idxs).intersection(not_null_idxs)].reset_index()[['index', 'images']].values:
    prefix = "%s/%06d_"%(download_folder, catalogue_idx)
    for index, image in enumerate(images):
        filename = prefix + "%06d.jpg"%index
        download_dict[filename] = image

# Downloading all the images
pbar = tqdm(total=len(download_dict))
def update(i):
    pbar.update()
if __name__ == '__main__':
    pool = Pool(8)
    for item in download_dict.items():
        pool.apply_async(download_image, args=(item,), callback=update)
    pool.close()
    pool.join()
    pbar.close()