from PIL import Image
import os
from tqdm import tqdm

with open('two_pages_files.txt', 'r') as infile:
    two_pages_files = infile.read().splitlines()

def split_image(filepath):
    img = Image.open(filepath)
    l = img.crop((0,0,img.width//2, img.height))
    r = img.crop((img.width//2,0,img.width, img.height))
    l.save(os.path.join(os.path.dirname(filepath),  os.path.basename(filepath).replace('.', 'l.')), format='JPEG', subsampling=0, quality=100)
    r.save(os.path.join(os.path.dirname(filepath),  os.path.basename(filepath).replace('.', 'r.')), format='JPEG', subsampling=0, quality=100)

for file in tqdm(two_pages_files):
    split_image(file)
    os.remove(file)