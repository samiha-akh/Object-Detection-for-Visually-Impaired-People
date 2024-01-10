#Important Packages 
import os
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import load_model
from pathlib import Path
import numpy as np

#dataset pre-processing
BATCH_SIZE = 32
img_size = (224,224)
#training data generator
traingen = ImageDataGenerator(       rotation_range=90, 
                                     brightness_range=[0.3, 0.6],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

#test dataset generator
testgen = ImageDataGenerator(preprocessing_function=preprocess_input)

#dataset path for training and testing data
path_to_training_data = 'new_dataset/train'
path_to_testing_data = 'new_dataset/test'

#class labels
labels = sorted(os.listdir(path_to_training_data))
#training data
training_data = traingen.flow_from_directory(  path_to_training_data,
                                               target_size=img_size,
                                               class_mode='categorical',
                                               classes=labels,
                                               subset='training',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=90)
#validation data
validation_data = traingen.flow_from_directory(path_to_training_data,
                                               target_size=img_size,
                                               class_mode='categorical',
                                               classes=labels,
                                               subset='validation',
                                               batch_size=BATCH_SIZE, 
                                               shuffle=True,
                                               seed=90)

#testing data
testing_data = testgen.flow_from_directory(    path_to_testing_data,
                                               target_size=img_size,
                                               class_mode=None,
                                               classes=labels,
                                               batch_size=1, 
                                               shuffle=False,
                                               seed=90)

#base model attributes
input_shape = (224,224,3)

base_model = MobileNetV2(include_top=False,
            weights='imagenet', 
            input_shape=input_shape)

base_model.summary()
base_model.trainable = True
#set the last 3 layer trainable
for layer in base_model.layers[:-3]:
    layer.trainable=False

base_model.summary()

#top of the model
inputs = keras.Input(shape=input_shape)
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  
outputs = Dense(4, activation='softmax')(x)
model = keras.Model(inputs, outputs)

model.summary()

#path where the model will be saved
model_path = 'ambience_mode/new_metrics/softmax'

#we will always save the model at a number of epoch where it shows the best validation accuracy
top_weights_path = os.path.join(os.path.abspath(model_path), 'model_5_softmax_4_class.h5')
callbacks_list = [
        ModelCheckpoint(top_weights_path, monitor='val_accuracy', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_accuracy', patience=7, verbose=0)
    ]

#Required variables for model training
optmzr = Adam(learning_rate=0.0003)
model.compile(optimizer=optmzr, loss='categorical_crossentropy',
                  metrics=['accuracy'])
history = model.fit(training_data, epochs=30, steps_per_epoch=BATCH_SIZE, validation_data=validation_data,callbacks=callbacks_list)


#train-validation loss and train-validation accuracy graph
import matplotlib.pyplot as plt

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,16)
plt.plot(epochs, loss_train, 'r', label='training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1,16)
plt.plot(epochs, train_acc, 'r', label='training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()