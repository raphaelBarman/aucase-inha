import data_downloader.download_utils as du
import pandas as pd
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
from operator import itemgetter


def downloading_data(config):
    """Downloads the necessary metadata and images"""
    metadata_folder = config['data']['output_dir']
    images_folder = config['images_folder']

    os.makedirs(metadata_folder, exist_ok=True)
    # downloading or loading the metadata from the OAI-PMH
    print("### Downloading OAI PMH metadata ###")
    inha_bib_num_metadata = du.download_oai_dc(
        os.path.join(metadata_folder, 'inha_bib_num_metadata.json.gz'))

    # Create a dataframe with the inha ids
    df_metadata = pd.DataFrame.from_dict(inha_bib_num_metadata)
    df_metadata['id'] = pd.to_numeric(
        df_metadata.identifier.apply(itemgetter(0)).str.split('/').apply(
            itemgetter(-1)))

    # Creating dictionaries from internal ids to inha ids
    idx2inha = {
        x: y
        for x, y in df_metadata.reset_index()[['index', 'id']].values
    }
    inha2idx = {v: k for k, v in idx2inha.items()}
    with open(
        os.path.join(metadata_folder, 'idx2inhaIdx.json'), 'w', encoding='utf-8') as outfile:
        json.dump({str(k): str(v) for k, v in idx2inha.items()}, outfile)

    iiif_base_url = "http://bibliotheque-numerique.inha.fr/iiif/%s/manifest"
    # Create the urls of the iiif manifest
    df_metadata['iiif_manifest_urls'] = df_metadata['id'].apply(
        lambda id: iiif_base_url % id)

    # Download the iiif manifests
    print("### Downloading iiif manifests ###")
    manifests_json, failed_manifests = du.download_iiif_manifests(
        df_metadata['iiif_manifest_urls'].values,
        os.path.join(metadata_folder, "iiif_manifests.json.gz"))
    print('There were %d manifest that failed to download' %
          len(failed_manifests))

    idx2cote = {}
    for inha, manifest in manifests_json.items():
        if 'metadata' in manifest:
            for metadata in manifest['metadata']:
                if metadata['label'] == 'reference':
                    if inha in inha2idx:
                        idx2cote[inha2idx[inha]] = "_".join(
                            metadata['value'].split('_')[2:4]).replace(
                                '.pdf', '')

    with open(os.path.join(metadata_folder, 'idx2cote.json'), 'w') as outfile:
        json.dump(idx2cote, outfile)

    # Add to the dataframe the raw manifests and the list of urls of the images
    # Dependent on the format of the iiif manifest of the bib num, may break
    manifest_dict = {}
    for manifest in manifests_json.values():
        id_ = int(manifest['@id'].replace('/manifest', '').split('/')[-1])
        manifest_dict[inha2idx[id_]] = manifest
    images_dict = {}
    for idx, record in manifest_dict.items():
        for image in record['sequences'][0]['canvases']:
            if idx not in images_dict:
                images_dict[idx] = []
            images_dict[idx].append(image['images'][0]['resource']['@id'])
    images = [x for y in images_dict.values() for x in y]
    df_metadata = df_metadata.join(
        pd.DataFrame.from_dict({k: [v]
                                for k, v in manifest_dict.items()},
                               orient='index',
                               columns=['manifest_raw']))
    df_metadata = df_metadata.join(
        pd.DataFrame.from_dict({k: [v]
                                for k, v in images_dict.items()},
                               orient='index',
                               columns=['images']))

    # Finding only the indexes of catalogs made by Drouot between 1939 and 1945
    drouot_idxs = df_metadata[df_metadata.creator.notnull()][df_metadata[
        df_metadata.creator.notnull()].creator.apply(
            lambda x: 'Drouot' in x)].index
    dates = pd.to_datetime(df_metadata[df_metadata.date.apply(len) == 3][
        df_metadata[df_metadata.date.apply(len) == 3].date.apply(
            itemgetter(1)).str.startswith("19")].date.apply(itemgetter(1)))
    period_idxs = dates[(dates > '1939-01-01') & (dates < '1945-12-31')].index
    not_null_idxs = df_metadata[df_metadata.images.notnull()].index

    period_idxs = not_null_idxs
    drouot_idxs = not_null_idxs

    # Creating a download dictionary containing
    # the filename as a key and the url as value
    if not os.path.exists(images_folder):
        os.makedirs(images_folder, exist_ok=True)
    download_dict = {}
    for catalogue_idx, images in df_metadata.loc[drouot_idxs.intersection(
            period_idxs).intersection(not_null_idxs)].reset_index()[[
                'index', 'images'
            ]].values:
        prefix = os.path.join(images_folder, '%06d_'%catalogue_idx)
        for index, image in enumerate(images):
            filename = prefix + "%06d.jpg" % index
            download_dict[filename] = image

    # Downloading all the images
    print("### Downloading images ###")
    pbar = tqdm(total=len(download_dict))

    def update(i):
        pbar.update()

    pool = Pool(8)
    for item in download_dict.items():
        pool.apply_async(du.download_image, args=(item, ), callback=update)
    pool.close()
    pool.join()
    pbar.close()
