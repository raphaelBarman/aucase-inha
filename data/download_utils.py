from sickle import Sickle
import json
from tqdm import tqdm
import os
import requests
import shutil

def download_oai_dc(outfile=None, base_url='http://bibliotheque-numerique.inha.fr/oai', sets=16800, force_refresh=False):
    if os.path.exists(outfile) and not force_refresh:
        with open(outfile, "r") as file:
            return json.load(file)

    sickle = Sickle(base_url)
    records = sickle.ListRecords(**{'metadataPrefix': 'oai_dc',
                                    'set': "oai:sets:%d"%sets})

    records_fetched = list()
    for record in tqdm(records):
        records_fetched.append(record.metadata)
    if outfile:
        with open(outfile, "w") as file:
            json.dump(records_fetched, file)

    return records_fetched

def download_iiif_manifests(iiif_urls, outfile=None, force_redownload=False):
    if os.path.exists(outfile) and not force_redownload:
        with open(outfile, 'r', encoding='utf-8') as infile:
            return json.load(infile), []
    
    manifests_json = {}
    failed_manifests = []

    def get_iiif_manifest(iiif_url):
        r = requests.get(iiif_url)
        if r.status_code == 200:
            return r.json()
        else:
            print("Error with", iiif_url)
            
    for idx, iiif_url in tqdm(enumerate(iiif_urls)):
        manifest = get_iiif_manifest(iiif_url)
        if manifest:
            manifests_json[idx] = manifest
        else:
            failed_manifests.append(idx)
    if outfile:
        with open(outfile, "w") as file:
            json.dump(manifests_json, file)
    return manifests_json, failed_manifests

def download_image(item):
    filename, url = item
    if os.path.isfile(filename):
        return
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)