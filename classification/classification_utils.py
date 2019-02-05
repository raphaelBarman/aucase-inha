import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
from sklearn.utils import class_weight


def gen_flow_for_two_inputs(X1, X2, y, datagen, batch_size):
    """Create a generator from two inputs"""
    genX1 = datagen.flow(X1, y, batch_size=batch_size, seed=55, shuffle=True)
    genX2 = datagen.flow(X1, X2, batch_size=batch_size, seed=55, shuffle=True)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]


def gen_flow_for_two_inputs_mixup(X1, X2, y, datagen, batch_size):
    """Create a mixup generator from two inputs"""
    genX1 = MixupGenerator(
        X1, y, batch_size=batch_size, datagen=datagen, seed=55)()
    genX2 = MixupGenerator(
        X1, X2, batch_size=batch_size, datagen=datagen, seed=55)()
    while True:
        X1i = next(genX1)
        X2i = next(genX2)
        yield [X1i[0], X2i[1][:, 0]], X1i[1]


def gen_flow_from_two_flows(genX1, genX2):
    """Merge two flows"""
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]


def preprocess_input_tf(x):
    return preprocess_input(x.astype(float), mode='tf')


def prepare_data(X_train_orig,
                 X_test_orig,
                 y_train_orig,
                 y_test_orig,
                 over_sampling=False,
                 mixup=False,
                 use_page_num=False,
                 BATCH_SIZE=32):
    """ Generates a flow for the training data and test data
    according to the given parameters
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input_tf,
        rotation_range=2,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode='nearest')

    X_train_page_num = X_train_orig['page_num'].values
    X_test_page_num = X_test_orig['page_num'].values

    X_train = np.array(X_train_orig['image_data'].values.tolist())
    X_test = np.array(X_test_orig['image_data'].values.tolist())
    y_train = y_train_orig.values
    y_test = y_test_orig.values

    if over_sampling:
        categories = X_train_orig['class'].unique()
        majority_class, num_majority = X_train_orig['class'].value_counts(
        ).idxmax(), X_train_orig['class'].value_counts().max()
        X_train_over = X_train_orig[X_train_orig['class'] == majority_class]
        for category in categories:
            if majority_class == category:
                continue
            X_category = X_train_orig[X_train_orig['class'] == category]
            X_train_over = pd.concat(
                [X_train_over,
                 X_category.sample(num_majority, replace=True)])
        y_train_over = pd.get_dummies(X_train_over['class'])
        X_train_page_num = X_train_over['page_num'].values

        y_train = y_train_over.values
        X_train = np.array(X_train_over['image_data'].values.tolist())

    X_test_prep = []
    for x in X_test:
        X_test_prep.append(preprocess_input_tf(x))
    X_test_prep = np.array(X_test_prep)

    test_data = (X_test_prep, y_test)

    flow = train_datagen.flow(
        X_train, y_train, batch_size=BATCH_SIZE, seed=55, shuffle=True)
    if mixup:
        flow = MixupGenerator(
            X_train, y_train, batch_size=BATCH_SIZE, datagen=train_datagen)()

    if use_page_num:
        flow = gen_flow_for_two_inputs(X_train, X_train_page_num, y_train,
                                       train_datagen, BATCH_SIZE)
        if mixup:
            X_train_page_num = np.repeat(
                X_train_page_num.reshape(-1, 1), 4, axis=1)
            flow = gen_flow_for_two_inputs_mixup(
                X_train, X_train_page_num, y_train, train_datagen, BATCH_SIZE)
        test_data = ([X_test_prep, X_test_page_num], y_test)
    class_weights = class_weight.compute_class_weight(
        'balanced', np.unique(np.argmax(y_train, axis=1)),
        np.array(np.argmax(y_train, axis=1)))
    return flow, test_data, class_weights


class MixupGenerator():
    """Creates a mixup generator"""

    def __init__(self,
                 X_train,
                 y_train,
                 batch_size=32,
                 alpha=0.2,
                 shuffle=True,
                 datagen=None,
                 seed=55):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.alpha = alpha
        self.shuffle = shuffle
        self.sample_num = len(X_train)
        self.datagen = datagen
        np.random.seed(seed)

    def __call__(self):
        while True:
            indexes = self.__get_exploration_order()
            itr_num = int(len(indexes) // (self.batch_size * 2))

            for i in range(itr_num):
                batch_ids = indexes[i * self.batch_size * 2:(i + 1) *
                                    self.batch_size * 2]
                X, y = self.__data_generation(batch_ids)

                yield X, y

    def __get_exploration_order(self):
        indexes = np.arange(self.sample_num)

        if self.shuffle:
            np.random.shuffle(indexes)

        return indexes

    def __data_generation(self, batch_ids):
        _, h, w, c = self.X_train.shape
        random_l = np.random.beta(self.alpha, self.alpha, self.batch_size)
        X_l = random_l.reshape(self.batch_size, 1, 1, 1)
        y_l = random_l.reshape(self.batch_size, 1)

        X1 = self.X_train[batch_ids[:self.batch_size]]
        X2 = self.X_train[batch_ids[self.batch_size:]]
        X = X1 * X_l + X2 * (1 - X_l)

        if self.datagen:
            for i in range(self.batch_size):
                X[i] = self.datagen.random_transform(X[i])
                X[i] = self.datagen.standardize(X[i])

        if isinstance(self.y_train, list):
            y = []

            for y_train_ in self.y_train:
                y1 = y_train_[batch_ids[:self.batch_size]]
                y2 = y_train_[batch_ids[self.batch_size:]]
                y.append(y1 * y_l + y2 * (1 - y_l))
        else:
            y1 = self.y_train[batch_ids[:self.batch_size]]
            y2 = self.y_train[batch_ids[self.batch_size:]]
            y = y1 * y_l + y2 * (1 - y_l)

        return X, y
