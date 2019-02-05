from keras.callbacks import TensorBoard
import os
import numpy as np
import tensorflow as tf
import sklearn.metrics as sklm
import re
import itertools
import tfplot
import matplotlib
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(correct_labels,
                          predict_labels,
                          labels,
                          title='Confusion matrix',
                          tensor_name='dev/cm',
                          normalize=False):
    '''
    Plot a tensorflow confusion matrix
    '''
    cm = confusion_matrix(correct_labels, predict_labels)
    if normalize:
        cm = cm.astype('float') * 10 / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm, copy=True)
        cm = cm.astype('int')

    np.set_printoptions(precision=2)

    fig = matplotlib.figure.Figure(
        figsize=(7, 7), facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)

    classes = [
        re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x)
        for x in labels
    ]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=12, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=14)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=12, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], 'd') if cm[i, j] != 0 else '.',
            horizontalalignment="center",
            fontsize=14,
            verticalalignment='center',
            color="black")
    fig.set_tight_layout(True)
    summary = tfplot.figure.to_summary(fig, tag=tensor_name)
    return summary


class TrainValTensorBoard(TensorBoard):
    def __init__(self, labels, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')
        # Labels Name
        self.labels = labels

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {
            k.replace('val_', ''): v
            for k, v in logs.items() if k.startswith('val_')
        }
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()
        if self.validation_data is not None:
            num_dim = len(self.validation_data) - 3
            predict = np.argmax(
                np.round(
                    np.asarray(
                        self.model.predict(self.validation_data[:num_dim]))),
                axis=1)
            targ = np.argmax(self.validation_data[num_dim], axis=1)
        else:
            predict = []
            targ = []
            for _ in range(self.validation_steps):
                X, y = next(self.validation_generator)
                b_predict = np.argmax(
                    np.round(np.asarray(self.model.predict(X))), axis=1)
                predict.extend(b_predict.tolist())
                targ.extend(np.argmax(y, axis=1).tolist())
            predict = np.array(predict)
            targ = np.array(targ)

        summary = tf.summary.text(
            'confusion_matrix',
            tf.convert_to_tensor(str(sklm.confusion_matrix(targ, predict))))

        summary = plot_confusion_matrix(
            targ, predict, self.labels, tensor_name='confusion_matrix')
        writer = tf.summary.FileWriter(self.val_log_dir)
        writer.add_summary(summary, epoch)
        writer.close()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
