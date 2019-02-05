from PIL import Image
import os
from tqdm import tqdm


def split_image(filepath):
    """Split an image in two,
    saves two images with l and r prefixes added"""
    img = Image.open(filepath)
    left = img.crop((0, 0, img.width // 2, img.height))
    right = img.crop((img.width // 2, 0, img.width, img.height))
    left.save(
        os.path.join(
            os.path.dirname(filepath),
            os.path.basename(filepath).replace('.', 'l.')),
        format='JPEG',
        subsampling=0,
        quality=100)
    right.save(
        os.path.join(
            os.path.dirname(filepath),
            os.path.basename(filepath).replace('.', 'r.')),
        format='JPEG',
        subsampling=0,
        quality=100)


def cutting_pages(config):
    """Cut the pages from a given list in two
    and removes the originals (!!!)"""
    config = config['data']
    if 'two_pages_path' in config:
        two_pages_path = config['two_pages_path']
    else:
        two_pages_path = 'two_pages_files.txt'

    with open(two_pages_path, 'r') as infile:
        two_pages_files = infile.read().splitlines()

    for file in tqdm(two_pages_files):
        split_image(file)
        os.remove(file)
