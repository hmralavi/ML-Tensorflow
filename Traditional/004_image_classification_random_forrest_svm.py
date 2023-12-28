"""
in this example, we use traditional machine learning tools for image binary classification.
we use two sets of images: horses and zebras.
then we apply various filters for feature extraction.
then we use random forrest and svm for classification (to detect wether the image is a horse or a zebra)
dataset from: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip
"""
#%% imports
import numpy as np
import cv2
from os import listdir
from matplotlib import pyplot as plt 
from sklearn.model_selection import train_test_split
from skimage.filters import sobel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics

#%% specify path to horse and zebra datasets.
horse_path = r"..\datasets\horse2zebra\trainA\\"
zebra_path = r"..\datasets\horse2zebra\trainB\\"

#%% load images and assign labels
def load_images(path: str):
    imgs = []
    for f in listdir(path)[:100]:
        im = cv2.imread(path+f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        imgs.append(im)
    return np.array(imgs)

horse_imgs = load_images(horse_path)
zebra_imgs = load_images(zebra_path)
horse_labels = np.zeros(horse_imgs.shape[0])
zebra_labels = np.ones(zebra_imgs.shape[0])

X = np.concatenate((horse_imgs, zebra_imgs))
X = np.expand_dims(X, axis=-1)
Y = np.concatenate((horse_labels, zebra_labels))

#%% feature extraction using various filters

# creating gabor kernels: cv2.getGaborKernel((k, k), sigma, theta, lamda, gamma, psi, ktype=cv2.CV_32F)
gabor_kernels = [cv2.getGaborKernel((3, 3), 1, 1*np.pi/4, 1*np.pi/8, 0.05, 0, ktype=cv2.CV_32F),
                 cv2.getGaborKernel((3, 3), 1, 2*np.pi/4, 1*np.pi/8, 0.05, 0, ktype=cv2.CV_32F),
                 cv2.getGaborKernel((3, 3), 1, -1*np.pi/4, 1*np.pi/8, 0.05, 0, ktype=cv2.CV_32F),
                 cv2.getGaborKernel((3, 3), 1, 1*np.pi/4, 1*np.pi/4, 0.05, 0, ktype=cv2.CV_32F)
                ]

class GaborFilt():
    def __init__(self, kernel):
        self.kernel = kernel.copy()
        
    def __call__(self, x):
        return cv2.filter2D(x, cv2.CV_8UC3, self.kernel)
        
    
filtdict = {'SOBEL': lambda x: sobel(x)*255}  # adding sobel filter to our filters banks
for i in range(len(gabor_kernels)):
    filtdict['GABOR%d' % (i+1)] = GaborFilt(gabor_kernels[i]) # adding gabor filters to our filters banks

# applying our filters bank to the images and inserting the result as a new channel to the image
for filt in filtdict.keys(): 
    out = np.zeros_like(X[...,0])[...,np.newaxis] # take original image
    for i in range(X.shape[0]): # a loop that applies our filters to all of the original images
        out[i,...] = filtdict[filt](X[i,:,:,0])[...,np.newaxis]
    X = np.concatenate((X, out), axis=-1)

#%% demonstrate a sample image with the filtering results
# np.random.seed(3)
isample = np.random.randint(0, X.shape[0])
img = X[isample,:,:,0]

fig, axs = plt.subplots(X.shape[-1], 1, figsize=(5, X.shape[-1]*5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title("original image %d (%s)" % (isample, "horse" if Y[isample]==0 else "zebra"))
axs[0].axis('off')
for i in range(1, X.shape[-1]):
    axs[i].imshow(X[isample,:,:,i], cmap='gray')
    axs[i].set_title(list(filtdict)[i-1])
    axs[i].axis('off')
    
#%% create training and testing datasets
xtrain, xtest, ytrain, ytest = train_test_split(X[...,:], Y, test_size=0.2, random_state=100)
xtrain = xtrain.reshape((xtrain.shape[0],-1))
xtest = xtest.reshape((xtest.shape[0],-1))

#%% setup RandomForrest and train it and test it
rf_model = RandomForestClassifier(n_estimators=100, random_state=30)
rf_model.fit(xtrain, ytrain)
print ("RandomForrest's Accuracy = ", metrics.accuracy_score(ytest, rf_model.predict(xtest)))
# result: RandomForrest's Accuracy =  0.85

#%% setup SVM and train it and test it
svm_model = SVC()
svm_model.fit(xtrain, ytrain)
print ("SVM's Accuracy = ", metrics.accuracy_score(ytest, svm_model.predict(xtest)))
# result: SVM's Accuracy =  0.675