import cv2
import numpy as np
import os
import os.path
import math
from PCA import *

FOLDER_NAME = './data/VGA_new/left'


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()   
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

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


pca_model = PCA() # train a new model use, PCA()
#pca_model = PCA(PCA_MODEL_DIR) # train a new model use, PCA()

for root, _, fnames in sorted(os.walk(FOLDER_NAME)):
    for fname in sorted(fnames):
        path = os.path.join(root, fname)
        if is_image_file(path):
            print(path)
            im = cv2.imread(path)
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
                rect = rect.astype(np.int)  ## this is the bounding box, only the first 4 elements of rect
                cv2.rectangle(im_display,(rect[0], rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
                
                plantImg = im_original[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
                plantImgs = np.expand_dims(plantImg, axis=0) # e.g., MAKE IT 1 BY 2 BY 2 from 2 by 2
                
                prediction = pca_model.prediction(plantImgs) # this stores all the images in a 3d array like n images by 640 by 480..
                
                #print(prediction)
                cv2.putText(im_display,str(ACT[int(prediction[0])]),(rect[0], rect[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
                
            cv2.imshow('Image',im_display)
            key = cv2.waitKey(30)
            while key == -1:
                key = cv2.waitKey(30)
                    
                
