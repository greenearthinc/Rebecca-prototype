import cv2
import numpy as np
import os
import os.path
import math
from PCA import *

import roslib
import rospy
import message_filters
# ROS Image message
from sensor_msgs.msg import Image
from std_msgs.msg import Int16MultiArray
#from rospy_tutorials.msg import Floats
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError


# Instantiate CvBridge
bridge = CvBridge()

image_tmp = None


def validateRects(rectlist, targetpoint):
    TempRectList = np.zeros((rectlist.shape[0], rectlist.shape[1]+5))
    TempRectList[:,:4] = np.copy(rectlist)
    for rect in TempRectList:
        ax = rect[0] + (rect[2]/2.0)
        ay = rect[1] + (rect[3]/2.0)
        rect[4] = math.sqrt((targetpoint[0]-ax)**2 + (targetpoint[1]-ay)**2)
        if rect[3] > rect[2]*1.5:
            rect[5] = -1
    TempRectList = TempRectList[TempRectList[:,4].argsort()[::-1]]
    return TempRectList[-1]

def callback(image):
    global image_tmp
    image_tmp = bridge.imgmsg_to_cv2(image, "bgr8")


pca_model = PCA(PCA_MODEL_DIR)

rospy.init_node('Plant_Detector')

rospy.Subscriber("camera/color/image_rect_color", Image,  callback = callback, queue_size=1)
image_pub = rospy.Publisher("camera/color/result",Image,queue_size=10)

print 'waiting...'
while image_tmp is None:
    pass
print 'start...'


while image_tmp is not  None:
    im = np.copy(image_tmp)
    im_display = np.copy(im)
    im_original = np.copy(im)
    #im = cv2.resize(im,(200, 150), interpolation = cv2.INTER_LINEAR)
    im = cv2.GaussianBlur(im, (21, 21), 0)
    

    colors = []
    with open('./colors.txt') as color_file:
        for line in color_file:
            thiline = line.strip().split(' ')
            if thiline[0] != '#':
                colors.append([np.int(x) for x in thiline])
    mask = 0
    for color in colors:
        mask += cv2.inRange(im, np.array(color)-5, np.array(color)+5) #150,150,150
        '''
        cv2.imshow("mask", cv2.medianBlur(mask,5))
        key = cv2.waitKey(5)
        while key == -1:
            key = cv2.waitKey(5)
        '''
    mask = cv2.medianBlur(mask,5)
    mask = cv2.boxFilter(mask,ddepth=-1, ksize=(65,65))
    mask[mask!=0] = 255
    
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    #print(hierarchy)
    LENGTH = len(contours)
    if LENGTH != 0:
        rectList = []
        for i,cnt in enumerate(contours):
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(mask,(x,y),(x+w,y+h),255,cv2.FILLED)
                
    _, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
    LENGTH = len(contours)
    if LENGTH != 0:
        rectList = []
        for i,cnt in enumerate(contours):
                x,y,w,h = cv2.boundingRect(cnt)
                rectList.append([x,y,w,h])
    
    rect = validateRects(np.asarray(rectList), [mask.shape[0]/2.0, mask.shape[1]/2.0])
    if rect[5] !=  -1:
        rect = rect.astype(np.int)
        cv2.rectangle(im_display,(rect[0], rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
        
        plantImg = im_original[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
        plantImgs = np.expand_dims(plantImg, axis=0)
        
        prediction = pca_model.prediction(plantImgs)
        
        #print(prediction)
        cv2.putText(im_display,str(ACT[int(prediction[0])]),(rect[0], rect[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
        
        try:
            image_pub.publish(CvBridge().cv2_to_imgmsg(im_display, "bgr8"))
        except CvBridgeError, e:
            print e
        
    #cv2.imshow('Image',im_display)
    #key = cv2.waitKey(30)
    #while key == -1:
     #   key = cv2.waitKey(30)
                    
                
