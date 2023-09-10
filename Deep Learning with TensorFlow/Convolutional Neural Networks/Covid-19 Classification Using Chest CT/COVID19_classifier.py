import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

'''
This program uses a keras Convolutional Neural network model trained using chest CT scan images of normal parients and patients with covid-19
'''
# variables
input_shape = (900,900,3)

# load saved model
model = load_model('COVID_Classifier.h5')


def adjust_image(new_scan):
    new_image = image.load_img(new_scan, target_size = input_shape[:2])
    new_image = image.img_to_array(new_image)
    new_image = np.expand_dims(new_image, axis = 0)
    new_image = new_image/255
    return new_image


def predict(new_image):
    prediction = model.predict_classes(new_image)
    certainity = (model.predict(new_image)[0][0])*100;
    if prediction[0][0] == 0:
        print("\nCOVID POSITIVE patient with {:.2f}% certainity.".format(certainity))
    else:
        print("\nCOVID NEGATIVE patient with {:.2f}% certainity.".format(certainity))


def main():
    im = input("Enter filename/path of your chest scan image: ")
    im = adjust_image(im)
    predict(im)


main()
