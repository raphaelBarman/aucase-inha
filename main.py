from config import load_config
from data_downloader.downloading_data import downloading_data
from classification.train import train
from classification.classify import classify
from segmentation import prepare_dataset, create_masks, create_tf_dataset
from segmentation.optimize_boxes_dhsegment import optimize_boxes_dhsegment
from segmentation.predict_boxes import predict_boxes
from database.extract_page_content import extract_page_content
from database.prepare_data_for_sql import prepare_data_for_sql

# Load configuration
config = load_config('./config.toml')
## Download the data
#downloading_data(config)
#
## Classification
### Train the classifier
#train(config)
### Classify the images
#classify(config)
#
## Segmentation
### Prepare the dataset
#prepare_dataset.prepare_dataset(config)
### Prepare the dhSegment masks
#create_masks.create_masks(config)
### Prepare the tf api data
#create_tf_dataset.create_tf_dataset(config)
### Optimize the parameters
#optimize_boxes_dhsegment(config)
## Predict the boxes
#predict_boxes(config)
# Page content
## Extract the page content
extract_page_content(config)
## Prepare csv for sql
prepare_data_for_sql(config)
