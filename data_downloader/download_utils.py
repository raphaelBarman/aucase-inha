from sickle import Sickle
import json
from tqdm import tqdm
import os
import gzip
import requests
import shutil


def write_json_gzip(filename, obj):
    json_string = json.dumps(obj) + "\n"
    json_bytes = json_string.encode('utf-8')
    with gzip.GzipFile(filename, 'w') as outfile:
        outfile.write(json_bytes)


def read_json_gzip(filename):
    with gzip.GzipFile(filename, 'r') as infile:
        json_bytes = infile.read()
    json_string = json_bytes.decode('utf-8')
    return json.loads(json_string)


def download_oai_dc(outfile=None,
                    base_url='http://bibliotheque-numerique.inha.fr/oai',
                    sets=16800,
                    force_refresh=False):
    """Downloads a set from a oai-pmh repository and returns it
    if given an outfile, save the resuls to it,
    will also use it as a cache if needed
    """
    if os.path.exists(outfile) and not force_refresh:
        return read_json_gzip(outfile)

    sickle = Sickle(base_url)
    records = sickle.ListRecords(**{
        'metadataPrefix': 'oai_dc',
        'set': "oai:sets:%d" % sets
    })

    records_fetched = list()
    i = 0
    for record in tqdm(records):
        if i == 100:
            break
        records_fetched.append(record.metadata)
        i += 1
    records_fetched = records_fetched
    if outfile:
        write_json_gzip(outfile, records_fetched)

    return records_fetched


def download_iiif_manifests(iiif_urls, outfile=None, force_redownload=False):
    """Download all the iiifs manifest from a list of urls
    if given an outfile, save the resuls to it,
    will also use it as a cache if needed
    """
    if os.path.exists(outfile) and not force_redownload:
        return read_json_gzip(outfile), []

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
        write_json_gzip(outfile, manifests_json)
    return manifests_json, failed_manifests


def download_image(item):
    """Download an image from a tuple of filename urls"""
    filename, url = item
    if os.path.isfile(filename):
        return
    response = requests.get(url, stream=True)
    with open(filename, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
