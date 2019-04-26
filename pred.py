import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from glob import glob
import numpy as np
import h5py
from sklearn.utils import shuffle
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import Dropout, Dense, Input, Activation
from keras.layers.normalization import BatchNormalization


with h5py.File('features.h5', 'r') as h:
    train = np.array(h['train'])
    test = np.array(h['test'])
    labels = np.array(h['labels'])
    synset = list(h['synset'])

get_filename = lambda x:os.path.splitext(os.path.split(filename)[-1])[0]
test_filenames = [get_filename(filename) for filename in glob('test/*.jpg')]

train, labels = shuffle(train, labels)
labels = to_categorical(labels, num_classes=120)

input_tensor = Input(train.shape[1:])
x = BatchNormalization()(input_tensor)
x = Dense(1024)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
x = Dense(120, activation='softmax')(x)

model = Model(input_tensor, x)

model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, labels, batch_size=128, epochs=100)
print model.evaluate(train, labels, batch_size=128, verbose=0)
preds = model.predict(test, verbose=1)

with open('submission.csv', 'w') as f:
    f.write('id,{}\n'.format(','.join(synset)))
    for filename, pred in zip(test_filenames, preds):
        f.write('{},{}\n'.format(filename, ','.join(str(p) for p in pred)))
