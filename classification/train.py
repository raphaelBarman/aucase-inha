import pandas as pd
import os
import cv2
from classification_utils import *
from tensorboard_utils import TrainValTensorBoard
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras import layers
from keras.optimizers import RMSprop
from keras.layers import GlobalMaxPooling2D
from tqdm import tqdm
tqdm.pandas()

# Configuration options
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCH = 20
STEPS_PER_EPOCH = 50
learning_rate = 1e-4
test_size = 0.2
over_sampling = False
use_page_num = True
mixup = False
resume_training = False
model_checkpoint_path = './vgg16_page_num.h5'
use_tensorboard = False
log_directory = './log'
image_dir = '/scratch/raphael/data/drouot'
classification_csv = './classified_images.csv'

# Loading the annotations
df_classification = pd.read_csv(classification_csv, header=None, names=['filename', 'class'])

# If no image dir is given
if image_dir is not None and len(image_dir) > 0 and not os.path.exists(df_classification['filename'].iloc[0]):
    image_dir = os.path.join(image_dir, '')
    df_classification['filename'] = image_dir + df_classification['filename']
    if not os.path.exists(df_classification['filename'].iloc[0]):
        print('Invalid directory or images path')
        raise

print("loading the images")
df_classification['image_data'] = df_classification['filename'].progress_apply(lambda x: cv2.resize(cv2.imread(x)[..., ::-1], IMAGE_SIZE, interpolation=cv2.INTER_CUBIC))
df_classification['page_num'] = df_classification['filename'].apply(lambda x: int(os.path.basename(x).split('_')[1].replace('.jpg', '').rstrip('lr')))

# Splitting the dataset in training and testing
X_train, X_test, y_train, y_test = train_test_split(df_classification, pd.get_dummies(df_classification['class']), test_size=test_size, random_state=42)

# Get the labels names
labels = y_train.columns.values.tolist()
with open('./labels.txt', 'w') as outfile:
    for idx, label in enumerate(labels):
        outfile.write("%s %d\n"%(label, idx))
NUM_CLASSES = len(labels)

# Create the flow for the training and the test data for the evalutation
flow, test_data, class_weights = prepare_data(X_train, X_test, y_train, y_test, over_sampling, mixup, use_page_num, BATCH_SIZE)

# Only recreate the model if asked
if not (resume_training and os.path.exists(model_checkpoint_path)):
    # Use a pretrained vgg16 on imagnet
    pre_trained_model = VGG16(include_top=False, weights='imagenet', input_tensor=None,
                                input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # Adding a second input layer if using the page numbers
    if use_page_num:
        input_2 = layers.Input(shape=[1], name="num")
        num_layer = layers.Dense(1, )(input_2)

    # Adding the top layers for the prediction with our number of classes
    x = pre_trained_model.output
    x = GlobalMaxPooling2D()(x)
    # If using the page numbers, adding a branch to input them just before the first fully connected layer
    if use_page_num:
        x = layers.concatenate([x, num_layer])
    x = layers.Dense(1024, activation='relu', name='fc2')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    # If using the page numbers, set the input accordingly
    if use_page_num:
        model = Model([pre_trained_model.input, input_2], x)
    else:
        model = Model(pre_trained_model.input, x)

    optimizer = RMSprop(lr=learning_rate)

    model.compile(loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['acc'])
else:
    model = load_model(model_checkpoint_path)
    print("restored model")

# Create a tensorboard callback if needed
callbacks = []
if use_tensorboard:
    callbacks = [TrainValTensorBoard(labels,
                    log_dir=log_directory,
                    write_graph=False)]

# Train the model
history = model.fit_generator(flow,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                epochs=NUM_EPOCH,
                                callbacks=callbacks,
                                validation_data=test_data,
                                class_weight=class_weights)

# Save the model
model.save(model_checkpoint_path)