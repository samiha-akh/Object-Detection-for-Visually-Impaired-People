# Object-Detection-for-Visually-Impaired-People

# Description
TensorFlow object detection API and SSDLite MobileNetV2 have been used to create the proposed object detection model. The pre-trained SSDLite MobileNetV2 model is trained on the COCO dataset, with almost 328,000 images of 90 different objects. The gradient particle swarm optimization (PSO) technique has been used in this work to optimize the final layers and their corresponding hyperparameters of the MobileNetV2 model. Next, we used the Google text-to-speech module, PyAudio, playsound, and speech recognition to generate the audio feedback of the detected objects. A Raspberry Pi camera captures real-time video where real-time object detection is done frame by frame with Raspberry Pi 4B microcontroller. The proposed device is integrated into a head cap, which will help visually impaired people to detect obstacles in their path, as it is more efficient than a traditional white cane. Apart from this detection model, we trained a secondary computer vision model and named it the “ambiance mode.” In this mode, the last three convolutional layers of SSDLite MobileNetV2 are trained through transfer learning on a weather dataset. The dataset comprises around 500 images from four classes: cloudy, rainy, foggy, and sunrise. In this mode, the proposed system will narrate the surrounding scene elaborately, almost like a human describing a landscape or a beautiful sunset to a visually impaired person. The performance of the object detection and ambiance description modes are tested and evaluated in a desktop computer and Raspberry Pi embedded system. Detection accuracy and mean average precision, frame rate, confusion matrix, and ROC curve measure the model's accuracy on both setups. This low-cost proposed system is believed to help visually impaired people in their day-to-day life.

# COCO Dataset

![gr001](https://github.com/samiha-akh/Object-Detection-for-Visually-Impaired-People/assets/156142386/9f86a857-90fc-466e-a388-47f561bea2b7)

# Weather Dataset

![gr004](https://github.com/samiha-akh/Object-Detection-for-Visually-Impaired-People/assets/156142386/b51aca87-1a76-4987-91b3-3fd979e1ffa9)

# SSD Network Architecture
![gr002](https://github.com/samiha-akh/Object-Detection-for-Visually-Impaired-People/assets/156142386/a5078253-dd82-4b06-90e0-240ad39fe298)

# Object Detection

![gr018](https://github.com/samiha-akh/Object-Detection-for-Visually-Impaired-People/assets/156142386/7be623e8-8060-4820-a3e8-5d119e1fa894)





