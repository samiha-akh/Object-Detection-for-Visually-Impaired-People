#Important Packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import glob
import cv2
import numpy as np

#Dataset pre-processing
BATCH_SIZE = 32
img_size = (224,224)
#Training data generator
traingen = ImageDataGenerator(       rotation_range=180, 
                                     brightness_range=[0.6, 0.9],
                                     width_shift_range=0.5, 
                                     height_shift_range=0.5,
                                     horizontal_flip=True, 
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)



#Dataset path for training and testing data
path_to_training_data = 'new_dataset/rep_dataset/'
#Class labels
labels = sorted(os.listdir(path_to_training_data))

#Defining representative dataset
def representative_data_gen():
  #image size
  SIZE = 224
  #training data
  train_images = []

  for directory_path in sorted(glob.glob('new_dataset/rep_dataset/*')):
      for img_path in glob.glob(os.path.join(directory_path, "*")):
          print(img_path)       
          img = cv2.imread(img_path, cv2.IMREAD_COLOR)       
          img = cv2.resize(img, (SIZE, SIZE))
          img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
          img = (img/255.0)*2-1
          img = img.astype('float32')
          img = img[np.newaxis,:,:,:]
          train_images.append(img)
  for data in train_images:
    yield[data]


#Load the previously trained model
model = load_model('ambience_mode/new_metrics/softmax/model_5_softmax_4_class.h5')

#Model conversion parameters
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

#Convert and save the model
tflite_model_quant = converter.convert()

file = open( 'new_tflite_models/ptq_models/tflite_model_5_quant_v2_int8_new.tflite' , 'wb' ) 
file.write(tflite_model_quant)

#Check if the model converted properly
interpreter = tf.lite.Interpreter(model_path="new_tflite_models/ptq_models/tflite_model_5_quant_v2_int8.tflite")
interpreter.allocate_tensors()

print(interpreter.get_input_details()[0]['shape'])  
print(interpreter.get_input_details()[0]['dtype']) 

print(interpreter.get_output_details()[0]['shape'])  
print(interpreter.get_output_details()[0]['dtype'])