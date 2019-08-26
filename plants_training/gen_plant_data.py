import cv2
import numpy as np
import os
import os.path
import math


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


def on_mouse(event, x, y, flags, im):
    if event == cv2.EVENT_LBUTTONDOWN:
        f = open('./colors.txt', 'a')
        f.write(str(im[y, x, 0]) + ' ' + str(im[y, x, 1]) + ' ' + str(im[y, x, 2]) + '\n')
        f.close()


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

count = 0

folder_name = 'VGA_new'
#directory = './data/'+folder_name+'/left'
directory = './data/'+folder_name+'/left'

for root, _, fnames in sorted(os.walk(directory)):
    for fname in sorted(fnames):		# filenames 
        path = os.path.join(root, fname)   # absolute path
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
                rect = rect.astype(np.int)
                cv2.rectangle(im_display,(rect[0], rect[1]),(rect[0]+rect[2],rect[1]+rect[3]),(0,255,0),2)
                
                cv2.imshow("mask", mask)
                key = cv2.waitKey(5)
                cv2.imshow('Image',im_display)
                cv2.setMouseCallback('Image', on_mouse, im)
                key = cv2.waitKey(30)
                while key == -1:
                    key = cv2.waitKey(30)
                if (key & 0xff)-48 in [0,1,2,3,4]:
                    print((key & 0xff)-48)
                    plant_id = (key & 0xff)-48
                    saveImg = im_original[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :]
                    cv2.imwrite('./RGB_plants/PLANT'+str(plant_id)+'_'+folder_name+'_{0:08d}.jpg'.format(count), saveImg)
                    count+=1
            


