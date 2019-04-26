import sys
sys.path.append('../models/research/slim')
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
from nets import inception, resnet_v1
import vgg_preprocessing, inception_preprocessing
from tensorflow.contrib import slim
import h5py

df = pd.read_csv('../dogBreed/sample_submission.csv')
synset = list(df.columns[1:])


inception_size = inception.inception_v3.default_image_size
resnet_size = resnet_v1.resnet_v1_152.default_image_size

model_dict = {'InceptionV3':{'model': inception.inception_v3,
                           'size':inception_size,
                           'scope':inception.inception_v3_arg_scope(),
                           'output':'AvgPool_1a',
                           'numclasses':1001,
                           'preprocessing':inception_preprocessing,
                           'ckpt_path':'inception_v3.ckpt'},
              'resnet_v1_152':{'model': resnet_v1.resnet_v1_152,
                           'size':resnet_size,
                           'scope':resnet_v1.resnet_arg_scope(),
                           'output':'global_pool',
                           'numclasses':1000,
                           'preprocessing':vgg_preprocessing,
                           'ckpt_path':'resnet_v1_152.ckpt'}
              }

train_filenames = glob('Images/*/*.jpg')
test_filenames = glob('test/*.jpg')
train_num = len(train_filenames)
test_num = len(test_filenames)
labels = [synset.index(filename.split('/')[1][10:].lower()) for filename in train_filenames]


def get_features(model_name, model):

    size = model['size']
    feature_model = model['model']
    preprocessing = model['preprocessing']

    filename = tf.placeholder(tf.string)
    file_contents = tf.read_file(filename)
    #image = tf.image.decode_jpeg(file_contents, channels=3)
    image_tmp= tf.image.decode_jpeg(file_contents, channels=3)
    image = tf.image.resize_images(image_tmp, (size, size))
    processed_image = preprocessing.preprocess_image(image, size, size, is_training=False)
    processed_inputs = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(model['scope']):
        _, end_points = feature_model(processed_inputs, num_classes=model['numclasses'], is_training=False)
    feature = tf.squeeze(end_points[model['output']])
    init_fn = slim.assign_from_checkpoint_fn(model['ckpt_path'],
            slim.get_model_variables(model_name))

    train_features = np.zeros([train_num, 2048], dtype=np.float32)
    test_features = np.zeros([test_num, 2048], dtype=np.float32)
    with tf.Session() as sess:
        init_fn(sess)
        for i, train_filename in tqdm(enumerate(train_filenames), total=train_num):
            train_features[i] = sess.run(feature, feed_dict={filename:train_filename})
        for i, test_filename in tqdm(enumerate(test_filenames), total=test_num):
            test_features[i] = sess.run(feature, feed_dict={filename:test_filename})

    return train_features, test_features

train_features = []
test_features = []
for model_name, model in model_dict.items():
    features = get_features(model_name, model)
    train_features.append(features[0])
    test_features.append(features[1])

train_features = np.concatenate(train_features, axis=1)
test_features = np.concatenate(test_features, axis=1)

with h5py.File('features.h5', 'w') as h:
    h.create_dataset('train', data=train_features)
    h.create_dataset('test', data=test_features)
    h.create_dataset('labels', data=labels)
    h.create_dataset('synset', data=synset)
