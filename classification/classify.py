from classification_utils import gen_flow_from_two_flows
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import pandas as pd
import os
import numpy as np
from glob import glob

# Configuration options
IMAGE_SIZE = (224, 224)
use_page_num = True
images_dir = '/scratch/raphael/data/drouot'
model_path = './vgg16_page_num.h5'
labels_path = './labels.txt'
classification_csv = './classified_images.csv'
predicted_classes_csv = './predicted_classes.csv'
df_classification = pd.read_csv(classification_csv, header=None, names=['filename', 'class'])

# If no image dir is given
if images_dir is not None and len(images_dir) > 0 and not os.path.exists(df_classification['filename'].iloc[0]):
    images_dir = os.path.join(images_dir, '')
    df_classification['filename'] = images_dir + df_classification['filename']
    if not os.path.exists(df_classification['filename'].iloc[0]):
        print('Invalid directory or images path')
        raise

# Load the labels
labels_dict = {}
with open(labels_path, 'r') as infile:
    labels_dict = {int(line.split()[1]) : line.split()[0] for line in infile.read().splitlines()}

# Create an image generator
image_datagen = ImageDataGenerator(preprocessing_function=lambda x: preprocess_input(x.astype(float), mode='tf'))

# Load the model
model = load_model(model_path)

# Load the images
filenames = [os.path.basename(x) for x in glob(images_dir + '*.jpg')]
page_nums = [int(x.split('_')[-1].replace('.jpg', '').rstrip('lr')) for x in filenames]
df_filenames = pd.DataFrame(np.array([filenames, [0]*len(filenames)]).T, columns=['filename', 'class'])

# Create the flows for prediciting
batch_size = 1
data = image_datagen.flow_from_dataframe(df_filenames, directory=images_dir, class_mode='input', shuffle=False, batch_size=batch_size, target_size=IMAGE_SIZE)
flow = data
if use_page_num:
    nums = image_datagen.flow(np.zeros((len(filenames),1,1,3)), page_nums, shuffle=False, batch_size=batch_size)
    flow = gen_flow_from_two_flows(data, nums)

# Predict the images classes from the model
preds = model.predict_generator(flow, steps=len(filenames)//batch_size, verbose=1)

# Save the predictions probabilities and take the maximum probability
np.save('./preds.npy', preds)
preds_argmax = np.argmax(preds, axis=1)
preds_labels = np.array(list(map(labels_dict.get,preds_argmax)))

# Create a dataframe to correct the prediction using the manual labels
df_pred = pd.DataFrame(list(zip(data.filenames, preds_labels)), columns=['filename', 'class'])
df_corrected = df_pred.set_index('filename').join(df_classification.set_index('filename'), rsuffix='_true')
df_corrected['class'][df_corrected['class_true'].notnull()] = df_corrected['class_true'][df_corrected['class_true'].notnull()]

# Save the corrected predicitions
df_corrected.reset_index()[['filename', 'class']].to_csv(predicted_classes_csv, header=None, index=False)