import toml
import collections
from os.path import join

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None

    source: https://gist.github.com/angstwad/bf22d1822c38a92ec0a9
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


default_config = {
    'data': {
        'two_pages_path': './two_pages_files.txt'
    },
    'classification': {
        'classification_csv': './classified_images.csv',
        'BATCH_SIZE': 33,
        'IMAGE_SIZE': (224, 224),
        'NUM_EPOCH': 20,
        'STEPS_PER_EPOCH': 50,
        'learning_rate': 0.0001,
        'log_directory': './log',
        'mixup': False,
        'over_sampling': False,
        'resume_training': False,
        'test_size': 0.3,
        'use_page_num': True,
        'use_tensorboard': False
    },
    'segmentation': {
        'annotation_directory':
        '/scratch/raphael/data/annotations/objects_description',
        'compute_prediction_data':
        False,
        'force_refresh':
        False,
        'model_dir':
        '/home/rbarman/models/segmentation_objects_description/model2/export/',
        'ocr_dir_google':
        '/scratch/raphael/data/ocr/drouot_ocr_google',
        'ocr_dir_inha':
        '/scratch/raphael/data/ocr/ocr_inha',
        'restore_df':
        True,
        'test_size':
        0.2,
        'tf_record_path':
        './data',
        'use_faster_rcnn':
        True,
        'use_inha_ocr':
        True
    },
    'database': {
        'actor_excel_path': './actors.xlsx',
        'force_refresh': False,
        'mysql_host': 'localhost',
        'mysql_pass': 'pass',
        'mysql_schema': 'sacase',
        'mysql_user': 'user',
        'ocr_dir': '/scratch/raphael/data/ocr/drouot_ocr_google',
        'section_classified_csv_path': './section_classified.csv',
    },
}
def load_config(config_path):
    with open(config_path, 'r') as infile:
        config = toml.load(infile)
    base_folder = config.get('base_folder') or './data' 
    default_config['base_folder'] = base_folder
    default_config['images_folder'] = join(base_folder, 'drouot_39-45')
    default_config['data']['output_dir'] = join(base_folder, 'metadata')
    default_config['classification']['labels_path'] = join(base_folder, 'labels.txt')
    default_config['classification']['model_checkpoint_path'] = join(base_folder, 'vgg16_page_num.h5')
    default_config['classification']['labels_txt_path'] = join(base_folder, 'labels.txt')
    default_config['classification']['predicted_classes_csv'] = join(base_folder, 'predicted_classes.csv')
    default_config['segmentation']['csv_train_path'] = join(base_folder, 'segmentation_train.csv')
    default_config['segmentation']['csv_test_path'] = join(base_folder, 'segmentation_test.csv')
    default_config['segmentation']['dhSegment_dir'] = join(base_folder, 'dhSegment')
    default_config['segmentation']['tf_record_path'] = join(base_folder, 'tf_api')
    default_config['segmentation']['optimize_output_dir'] = join(base_folder, 'optimize_segmentation')
    default_config['segmentation']['output_dir'] = join(base_folder, 'boxes_prediction')
    default_config['database']['output_dir'] = join(base_folder, 'page_content')
    default_config['database']['sql_output_dir'] = join(base_folder, 'sql_csv')
    dict_merge(default_config, config)
    return default_config


#data_folder = './data'
#images_folder = './data/drouot_39-45'
#
#[data]
#output_dir = './data/medatadata'
#two_pages_path = './data/metadata/two_pages_files.txt'
#
#[classification]
#IMAGE_SIZE = [224, 224]
#BATCH_SIZE = 32
#NUM_EPOCH = 20
#STEPS_PER_EPOCH = 50
#learning_rate = 1e-4
#test_size = 0.2
#over_sampling = False
#use_page_num = True
#mixup = False
#resume_training = False
#model_checkpoint_path = './data/vgg16_page_num.h5'
#use_tensorboard = False
#log_directory = './log'
#classification_csv = './data/classified_images.csv'
#labels_path = './labels.txt'
#predicted_classes_csv = './data/predicted_classes.csv'
#
#[segmentation]
#output_dir = './data/boxes_prediction'
#force_refresh = False
#use_faster_r_cnn = True
#annotation_directory = '/scratch/raphael/data/annotations/objects_description'
#csv_train_path = './train.csv'
#csv_test_path = './test.csv'
#test_size = 0.2
#mask_output_dir = './masks'
#csv_output_dir = './masks'
#tf_record_path = './data'
#model_dir = '/home/rbarman/models/segmentation_objects_description/model2/export/'
#ocr_dir_google = '/scratch/raphael/data/ocr/drouot_ocr_google'
#ocr_dir_inha = '/scratch/raphael/data/ocr/ocr_inha'
#use_inha_ocr = True
#optimize_output_dir = '/scratch/raphael/boxes_dhsegment'
#restore_df = True
#compute_prediction_data = False'
#
#[database]
#output_dir = '/scratch/raphael/page_content'
#force_refresh = False
#ocr_dir = '/scratch/raphael/data/ocr/drouot_ocr_google'
#section_classified_csv_path = './section_classified.csv'
#actor_excel_path = './actors.xlsx'
#sql_output_dir = '/scratch/raphael/page_content/sql_csv'
#mysql_host = 'localhost'
#mysql_schema = 'sacase'
#mysql_user = 'user'
#mysql_pass = 'pass'