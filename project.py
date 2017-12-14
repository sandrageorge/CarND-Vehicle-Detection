import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from vehicle_detection_func import *
import os
import pickle

# Decide the features or combination of features to be used
# Traentry in a classifier - linear SVM
# Sliding window technique and minimize number of search
# Tracking to follow detected vehicles

# pickle_file = 'classifier_rbf.p'
# with open(pickle_file, 'rb') as f:
#     pickle_data = pickle.load(f)
#     # X_train = pickle_data['train_dataset']
#     # y_train = pickle_data['train_labels']
#     # X_test = pickle_data['test_dataset']
#     # y_test = pickle_data['test_labels']
#     X_scaler = pickle_data['X_scaler']
#     parameters = pickle_data['parameters']
#     svc = pickle_data['svc']
#
#     del pickle_data  # Free up memory
#
# print('Data and modules loaded.')

for k in parameters:
    print(k, ":", parameters[k])

test_img = mpimg.imread('./test_images/test1.jpg')


# from moviepy.editor import VideoFileClip
# # output = 'test_cars.mp4'
# # clip1 = VideoFileClip("test_video.mp4")
# output = 'project_video_cars.mp4'
# clip1 = VideoFileClip("project_video.mp4")
# # output = 'test_video_cars.mp4'
# # clip1 = VideoFileClip("test_video.mp4")
#
#
# # left_lane = Line()
# # right_lane = Line()
# clip = clip1.fl_image(find_cars) #NOTE: this function expects color images!!
# # clip.show()
# clip.write_videofile(output, audio=False)

rectangles = find_cars(image=test_img, parameters=parameters, X_scaler=X_scaler, svc=svc)
print(len(rectangles), 'rectangles found in image')