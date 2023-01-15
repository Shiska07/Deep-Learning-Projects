This program uses an external dataset of cats and dogs images of different sizes provided on kaggle. A Convolutional Neural Network(CNN) is created to classify a test image as a cat(0) or a dog(1) image.

The data requires reprocessing before being used to train the model. This is done using ImageDataGenerator imported from tensorflow.keras.preprocessing.image. To make the model invariant to size, rotation, shift, etc., to resize all images to be of the same size and to provide class label to each image data, ImageDatagenerator can be extremely useful. 

After the model is trained, any new input image that requires classification must undergo the following steps: 

1. Size must be changed to target size
2. The image must be converted into an array
3. The shape/dimension must be changed to conver the single or multiple test images into a single batch.
