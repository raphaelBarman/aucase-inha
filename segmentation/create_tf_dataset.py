import os
import io
from tqdm import tqdm
import pandas as pd
import tensorflow as tf

csv_train_path = './train.csv'
csv_test_path = './test.csv'
tf_record_path = './'

class2int = {
    'section_author' : 1,
    'sale_description': 2
}

def create_tf_example(group_info):
    _, group = group_info
    filename, width, height = group.iloc[0][['filename', 'width', 'height']].values
    with tf.gfile.GFile(filename, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)

    filename = filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for class_, xmin, xmax, ymin, ymax in group[['class', 'xmin', 'xmax', 'ymin', 'ymax']].values:
        xmins.append(xmin / width)
        xmaxs.append(xmax / width)
        ymins.append(ymin / height)
        ymaxs.append(ymax / height)
        classes_text.append(class_.encode('utf8'))
        classes.append(class2int[class_])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_format])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))
    return tf_example

def main(_):
    writer = tf.python_io.TFRecordWriter(os.path.join(tf_record_path, 'train.record'))
    df_train = pd.read_csv(csv_train_path)
    for group in tqdm(df_train.groupby('filename')):
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    writer.close()

    writer = tf.python_io.TFRecordWriter(os.path.join(tf_record_path, 'test.record'))
    df_test = pd.read_csv(csv_test_path)
    for group in tqdm(df_test.groupby('filename')):
        tf_example = create_tf_example(group)
        writer.write(tf_example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    tf.app.run()