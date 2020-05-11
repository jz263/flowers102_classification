import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Dropout, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import math

def fc_block(X,units,dropout,stage):
    fc_name = 'fc' + str(stage)
    X = Dense(units,activation ='elu',name = fc_name)(X)
    X = Dropout(dropout)(X)
    return X

def ResNet50_transfer():

    #call base_model
    base_model = ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor= Input(shape=img_size + (3,))
    )

    # freeze resnet layers' params
    for layer in base_model.layers:
        layer.trainable = False

    # top architecture
    X = base_model.output
    X = Flatten()(X)
    X = Dropout(0.4)(X)
    X = fc_block(X,fc_layer_units[0],dropout = 0.4,stage = 1)
    X = fc_block(X,fc_layer_units[1],dropout = 0.4,stage = 2)

    # output layer
    X = Dense(len(classes),activation='softmax',name = 'fc3_output')(X)

    # create model
    model = Model(inputs = base_model.input,outputs = X, name = 'ResNet50_transfer')

    return model

def generate_data(train_path,valid_path):
    # generate & augment training data
    train_datagen = ImageDataGenerator(rotation_range=30., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_datagen.mean = np.array([123.675, 116.28 , 103.53], dtype=np.float32).reshape((3, 1, 1))
    train_data = train_datagen.flow_from_directory(train_path, target_size=img_size, classes=None)
    # generate training data
    valid_datagen = ImageDataGenerator()
    valid_datagen.mean = np.array([123.675, 116.28 , 103.53], dtype=np.float32).reshape((3, 1, 1))
    valid_data = train_datagen.flow_from_directory(valid_path, target_size=img_size, classes=None)
    return train_data, valid_data

def call_back():
    early_stopping = EarlyStopping(verbose=1, patience=10, monitor='val_loss')
    model_checkpoint = ModelCheckpoint(filepath='102flowersmodel.h5', verbose=1, save_best_only=True, monitor='val_loss')
    callbacks = [early_stopping, model_checkpoint]
    return callbacks

train_path = 'dataset/flower_data_10/train'
valid_path = 'dataset/flower_data_10/valid'

nb_epoch = 20
batch_size = 32
img_size = (224,224)
classes = list(map(str,[1,2,3,4,5,6,7,8,9,10]))
rgb_mean = [123.68, 116.779, 103.939]
fc_layer_units = [512,64]

model = ResNet50_transfer()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
train_data, valid_data = generate_data(train_path,valid_path)
callbacks = call_back()
model.fit_generator(train_data, steps_per_epoch= math.ceil(train_data.samples / batch_size), epochs=nb_epoch,
                    validation_data=valid_data, validation_steps=math.ceil(valid_data.samples / batch_size),
                    callbacks=callbacks)
