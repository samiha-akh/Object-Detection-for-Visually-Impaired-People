import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2

model = load_model('ambience_mode/new_metrics/softmax/model_5_softmax_4_class.h5')

labels = sorted(os.listdir('new_dataset/train/'))
print(labels)

img_size = (224,224)
testgen = ImageDataGenerator(preprocessing_function=preprocess_input)
path_to_testing_data = 'new_dataset/test'

#testing data
testing_data = testgen.flow_from_directory(    path_to_testing_data,
                                               target_size=img_size,
                                               class_mode=None,
                                               classes=labels,
                                               batch_size=1, 
                                               shuffle=False,
                                               seed=42)


prediction = model.predict(testing_data)

i = 0 # index of prediction score
for directory_path in sorted(glob.glob('new_dataset/test/*')):
  for img_path in glob.glob(os.path.join(directory_path,'*')):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img,img_size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred_score = np.around(prediction[i],decimals = 3)
    pred_score = 'Prediction Score : '+str(pred_score)
    pred_class =(np.argmax(prediction[i],axis=0))
    pred_class = '  Prediction Class : '+str(labels[pred_class])
    title = pred_score+pred_class
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    i+=1
    if(i==30):
      break