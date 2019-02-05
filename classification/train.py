import pandas as pd
from os import path
import cv2
import classification.classification_utils as utils
from classification.tensorboard_utils import TrainValTensorBoard
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
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


def train(config):
    images_folder = config['images_folder']
    classification_csv = config['classification']['classification_csv']
    IMAGE_SIZE = config['classification']['IMAGE_SIZE']
    BATCH_SIZE = config['classification']['BATCH_SIZE']
    NUM_EPOCH = config['classification']['NUM_EPOCH']
    STEPS_PER_EPOCH = config['classification']['STEPS_PER_EPOCH']
    learning_rate = config['classification']['learning_rate']
    test_size = config['classification']['test_size']
    over_sampling = config['classification']['over_sampling']
    use_page_num = config['classification']['use_page_num']
    mixup = config['classification']['mixup']
    resume_training = config['classification']['resume_training']
    model_checkpoint_path = config['classification']['model_checkpoint_path']
    labels_txt_path = config['classification']['labels_txt_path']
    use_tensorboard = config['classification']['use_tensorboard']
    log_directory = config['classification']['log_directory']

    # Loading the annotations
    df = pd.read_csv(
        classification_csv, header=None, names=['filename', 'class'])

    # If no image dir is given
    if images_folder is not None and len(
            images_folder) > 0 and not path.exists(df['filename'].iloc[0]):
        image_dir = path.join(images_folder, '')
        df['filename'] = image_dir + df['filename']
        if not path.exists(df['filename'].iloc[0]):
            print('Invalid directory or images path')
            raise

    print("loading the images")
    df['image_data'] = df['filename'].progress_apply(
        lambda x: cv2.resize(
            cv2.imread(x)[..., ::-1],
            IMAGE_SIZE,
            interpolation=cv2.INTER_CUBIC))
    df['page_num'] = df['filename'].apply(
        lambda x: int(
            path.basename(x).split('_')[1].replace('.jpg', '').rstrip('lr')))

    # Splitting the dataset in training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        df, pd.get_dummies(df['class']), test_size=test_size, random_state=42)

    # Get the labels names
    labels = y_train.columns.values.tolist()
    with open(labels_txt_path, 'w') as outfile:
        for idx, label in enumerate(labels):
            outfile.write("%s %d\n" % (label, idx))
    NUM_CLASSES = len(labels)

    # Create the flow for the training and the test data for the evalutation
    flow, test_data, class_weights = utils.prepare_data(
        X_train, X_test, y_train, y_test, over_sampling, mixup, use_page_num,
        BATCH_SIZE)

    # Only recreate the model if asked
    if not (resume_training and path.exists(model_checkpoint_path)):
        # Use a pretrained vgg16 on imagnet
        pre_trained_model = VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

        # Adding a second input layer if using the page numbers
        if use_page_num:
            input_2 = layers.Input(shape=[1], name="num")
            num_layer = layers.Dense(1, )(input_2)

        # Adding the top layers for the prediction with our number of classes
        x = pre_trained_model.output
        x = GlobalMaxPooling2D()(x)
        # If using the page numbers,
        # adding a branch to input them
        # just before the first fully connected layer
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

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['acc'])
    else:
        model = load_model(model_checkpoint_path)
        print("restored model")

    # Create a tensorboard callback if needed
    callbacks = []
    if use_tensorboard:
        callbacks = [
            TrainValTensorBoard(
                labels, log_dir=log_directory, write_graph=False)
        ]

    # Train the model
    model.fit_generator(
        flow,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=NUM_EPOCH,
        callbacks=callbacks,
        validation_data=test_data,
        class_weight=class_weights)

    # Save the model
    model.save(model_checkpoint_path)