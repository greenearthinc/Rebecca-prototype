from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import matplotlib.image as mpimg
import random
import time
import os
import re
import cv2

#Required Files and Folders
UNCROPPED = "RGB_plants"
CROPPED = "training_data"
PCA_MODEL_DIR = 'pca_model'

# Default Global Variables
TRAIN_SIZE = 160
VALD_SIZE = 20
TEST_SIZE = 20

#one dimention of face image (N*N)
FACE_SIZE = 32
# Default Image Shape
IMG_SHAPE = (FACE_SIZE, FACE_SIZE)
IMG_DIM = IMG_SHAPE[0]*IMG_SHAPE[1]

# Put in actors that you want to predict
ACT = ['PLANT0',  'PLANT1',   'PLANT2',  'PLANT3']

# The number of eigenfaces that do you want to compare
K = [2, 5, 10, 20]

class PCA:
    def __init__(self, directory=None):
        if directory != None:
            self.eigenvectors = np.load('./'+PCA_MODEL_DIR+'/eigenvectors.npy')
            self.mean_plant = np.load('./'+PCA_MODEL_DIR+'/mean_plant.npy')
            self.train_alphas = np.load('./'+PCA_MODEL_DIR+'/train_alphas.npy')
            self.train_labels = np.load('./'+PCA_MODEL_DIR+'/train_labels.npy')
            self.best_k = np.load('./'+PCA_MODEL_DIR+'/best_k.npy')
        else:
            self.cut_images()
            #build PCA
            print ("Constructing PCA...")
            self.eigenvectors, self.mean_plant, self.train_alphas, train_set, self.train_labels, vald_set, vald_labels, test_set, test_labels = self.get_pca_all_acts(ACT)
            self.best_k = 0
            max_accuracy = 0
            print ("Running validation set...")
            vald_alphas = self.get_alpha_basis(self.eigenvectors, self.mean_plant, vald_set)
            for k in K:
                prediction = self.pca_prediction(k, self.train_alphas, vald_alphas, self.train_labels)
                accuracy = self.calculate_accuracy(prediction, vald_labels)
                if (max_accuracy < accuracy):
                    max_accuracy, self.best_k = accuracy, k
                print ("The accuracy for k = %d is %.3f" % (k, accuracy))
                
            print ("Running test set...")
            test_alphas = self.get_alpha_basis(self.eigenvectors, self.mean_plant, test_set)
            prediction, best_match, best_match_err = self.pca_prediction(self.best_k, self.train_alphas, test_alphas, self.train_labels, True)
            accuracy = self.calculate_accuracy(prediction, test_labels)
            print ("Test: The final accuracy for k = %d on test set is %.3f" % (self.best_k, accuracy))
            self.display_save_image(self.mean_plant, IMG_SHAPE)
            self.display_save_25_comps(self.eigenvectors, IMG_SHAPE)
            self.show_prediction_err(prediction, test_labels, best_match, best_match_err, train_set, test_set, IMG_SHAPE, self.train_labels)
            np.save('./'+PCA_MODEL_DIR+'/eigenvectors', self.eigenvectors)
            np.save('./'+PCA_MODEL_DIR+'/mean_plant', self.mean_plant)
            np.save('./'+PCA_MODEL_DIR+'/train_alphas', self.train_alphas)
            np.save('./'+PCA_MODEL_DIR+'/train_labels', self.train_labels)
            np.save('./'+PCA_MODEL_DIR+'/best_k', self.best_k)
        
    def cut_images(self):
        im_files = sorted([ UNCROPPED+"/" + filename for filename in os.listdir( UNCROPPED+"/")])
        for filename in im_files:
            imRGB = cv2.imread(filename)
            canvas = np.copy(imRGB)
            while (canvas.shape[0] > FACE_SIZE*2):
                canvas = cv2.resize(canvas,(int(canvas.shape[0]/2), int(canvas.shape[1]/2)), interpolation = cv2.INTER_LINEAR)
            canvas = cv2.resize(canvas,(FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_LINEAR)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            print (filename.replace(UNCROPPED, CROPPED))
            cv2.imwrite(filename.replace(UNCROPPED, CROPPED), canvas)
    
    def display_save_image(self, image, im_shape):
        '''Display mean image'''
        figure()
        plt.subplot(1, 1, 1)
        plt.axis('off')
        gray()
        imshow(image.reshape(im_shape))
        savefig('part2_display_save_mean_plant.jpg')  
        show()
        

    def display_save_25_comps(self, V, im_shape):
        '''Display 25 components in Vectors'''
        figure()
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.axis('off')
            gray()
            imshow(V[i,:].reshape(im_shape))
        savefig('part2_display_save_25_comps.jpg')  
        show()
    
    def get_random_imgset(self, prefix, label, img_dir):
        '''Give you six matrices, training set, training labels, validation set, validation labels, test set and test labels.'''
        img_list = os.listdir(img_dir)
        random.shuffle(img_list)

        train_set = zeros((TRAIN_SIZE, IMG_DIM))
        vald_set = zeros((VALD_SIZE, IMG_DIM))
        test_set = zeros((TEST_SIZE, IMG_DIM))
        train_labels = zeros(TRAIN_SIZE)
        vald_labels = zeros(VALD_SIZE)
        test_labels = zeros(TEST_SIZE)
        i = 0
        for filename in img_list:

            if re.split(r'_', filename)[0] == prefix:
                # print prefix, filename
                if i < TRAIN_SIZE:
                    train_set[i, :], train_labels[i] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten(), label

                elif i < TRAIN_SIZE + VALD_SIZE:
                    vald_set[i - TRAIN_SIZE, :], vald_labels[i - TRAIN_SIZE] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten(), label  # if the code does not work , remove the /255.. basically do not normalise, if you see weird mean plant image

                elif i < TRAIN_SIZE + VALD_SIZE + TEST_SIZE:
                    test_set[i - TRAIN_SIZE - VALD_SIZE, :], test_labels[i - TRAIN_SIZE - VALD_SIZE] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten(), label
                '''
		if i < TRAIN_SIZE:
                    train_set[i, :], train_labels[i] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten()/255, label

                elif i < TRAIN_SIZE + VALD_SIZE:
                    vald_set[i - TRAIN_SIZE, :], vald_labels[i - TRAIN_SIZE] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten()/255, label  # if the code does not work , remove the /255.. basically do not normalise, if you see weird mean plant image

                elif i < TRAIN_SIZE + VALD_SIZE + TEST_SIZE:
                    test_set[i - TRAIN_SIZE - VALD_SIZE, :], test_labels[i - TRAIN_SIZE - VALD_SIZE] = cv2.cvtColor(
cv2.imread(img_dir+'/'+filename), cv2.COLOR_BGR2GRAY).flatten()/255, label
                '''
                i += 1

        # Give results as training set, validation set and test set respectively.
        return train_set, train_labels, vald_set, vald_labels, test_set, test_labels


    def get_pca_all_acts(self, acts):
        '''Give three matrices (eigenvectors, eigenvalues, mean face for all PCA on actors) and six matrices (training set alphas, training labels, validation set and its labels, test set and its labels)'''
        act_filenames = [act_name for act_name in acts]
        train_set = zeros((TRAIN_SIZE*len(acts), IMG_DIM))
        vald_set = zeros((VALD_SIZE*len(acts), IMG_DIM))
        test_set = zeros((TEST_SIZE*len(acts), IMG_DIM))
        train_labels = zeros(TRAIN_SIZE*len(acts))
        vald_labels = zeros(VALD_SIZE*len(acts))
        test_labels = zeros(TEST_SIZE*len(acts))

        i = 0
        for act in act_filenames:
            train_set[TRAIN_SIZE*i:TRAIN_SIZE*(i+1), :], train_labels[TRAIN_SIZE*i:TRAIN_SIZE*(i+1)], \
            vald_set[VALD_SIZE*i:VALD_SIZE*(i+1), :], vald_labels[VALD_SIZE*i:VALD_SIZE*(i+1)], \
            test_set[TEST_SIZE*i:TEST_SIZE*(i+1), :], test_labels[TEST_SIZE*i:TEST_SIZE*(i+1)] = self.get_random_imgset(act, i, './'+CROPPED)
            i += 1
            
        #V, S, mean_X = pca(train_set)

        mean_X, V = cv2.PCACompute(train_set, np.mean(train_set, axis=0).reshape(1,-1))
        mean_X = mean_X.flatten()

        return V, mean_X, self.get_alpha_basis(V, mean_X, train_set), train_set, train_labels, vald_set, vald_labels, test_set, test_labels
    
    def get_alpha_basis(self, V, mean_im, img_set):
        ''' Calculate a list of alphas from some set using a flattened image.'''
        alphas = np.zeros((img_set.shape[0], V.shape[0]))
        l = 0
        for img in img_set:
            alphas[l, :] = [np.dot(V[i,:], (img-mean_im)) for i in range(V.shape[0])]
            l += 1
        return alphas


    def calculate_alpha_dist(self, train_alpha, vald_alpha):
        '''Calculate the distance between two coordinates'''
        return np.sum((train_alpha - vald_alpha)**2)


    def pca_prediction(self, k, train_alphas, vald_alphas, train_labels, output_best_match=False):
        '''Find the predictions by using k components of pca on all actor models. '''
        prediction = zeros(vald_alphas.shape[0])
        best_match = zeros(vald_alphas.shape[0])
        best_match_err = zeros(vald_alphas.shape[0])

        for v in range(vald_alphas.shape[0]):

            min_dist, label = float('inf'), -1
            for t in range(train_alphas.shape[0]):
                cur_dist = self.calculate_alpha_dist(train_alphas[t, :k], vald_alphas[v, :k])
                if cur_dist <= min_dist:
                    min_dist, label, match, = cur_dist, train_labels[t], t
                        
            prediction[v], best_match[v], best_match_err[v] = label, match, min_dist

        if output_best_match == False:
            return prediction
        else:
            return prediction, best_match, best_match_err
    
    def prediction(self, Imgs):
        testImgs = zeros((Imgs.shape[0], IMG_DIM))
        for i in range(Imgs.shape[0]):
            if len(Imgs.shape)==4:
                img = Imgs[i,:,:,:]
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = Imgs[i,:,:]
                
            img = cv2.resize(img,(FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_LINEAR)
            img = img.flatten()   # delete / add /255 if it does not produce correct visual result
            #img = img.flatten()/255   # delete / add /255 if it does not produce correct visual result
            testImgs[i,:] = img
        
        testAlphas = self.get_alpha_basis(self.eigenvectors, self.mean_plant, testImgs)
        
        prediction = self.pca_prediction(self.best_k, self.train_alphas, testAlphas, self.train_labels)
        return prediction

    def calculate_accuracy(self, prediction, vald_labels):
        if prediction.shape == vald_labels.shape:
            prediction_copy = np.copy(prediction)
            vald_labels_copy = np.copy(vald_labels)
            return 1.0 - np.count_nonzero(prediction_copy - vald_labels_copy) / float(vald_labels.shape[0])


    def show_prediction_err(self, prediction, test_labels, best_match, best_match_err, train_set, test_set, im_shape, train_labels):
        act_filenames = [act_name for act_name in ACT]
        prediction_copy = np.copy(prediction)
        test_labels_copy = np.copy(test_labels)
        prediction_err_set = prediction_copy - test_labels_copy
        num_err = 10

        fig = figure()
        s = 0
        for i in range(prediction_err_set.shape[0]):
            if i > 10 and prediction_err_set[i] != 0 and s < num_err:
                im_reshaped = array(train_set[int(best_match[i]), : ]).reshape(im_shape)
                subplot(num_err, 2, s*2+1)
                plt.xlabel("Prediction: " + act_filenames[int(prediction[i])] + " SSD = %2.0f "%(best_match_err[i]), fontsize=10)
                imshow(im_reshaped, cmap = cm.Greys_r)
                
                frame = plt.gca()
                frame.axes.get_xaxis().set_ticks([])
                frame.axes.get_yaxis().set_ticks([])

                im_reshaped = array(test_set[i, : ]).reshape(im_shape)
                subplot(num_err, 2, s*2+2)
                plt.xlabel("Original: " + act_filenames[int(test_labels[i])], fontsize=10)
                imshow(im_reshaped, cmap = cm.Greys_r)
                
                frame = plt.gca()
                frame.axes.get_xaxis().set_ticks([])
                frame.axes.get_yaxis().set_ticks([])
                s += 1
        fig.subplots_adjust(hspace=.7)
        savefig('error_sample.jpg')  
        show()
