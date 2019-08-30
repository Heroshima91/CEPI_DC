import cv2
import numpy as np
import os

image_paths = []
path = "Dataset\\train"
voc = 30

#list of our class names
training_names = os.listdir(path)

training_paths = []
names_path = []
#get full list of all training images
for p in training_names:
    training_paths1 = os.listdir("Dataset\\train\\"+p)
    for j in training_paths1:
        training_paths.append("Dataset\\train\\"+p+"\\"+j)
        names_path.append(p)

sift = cv2.xfeatures2d.SIFT_create()



dictionarySize = 20

BOW = cv2.BOWKMeansTrainer(dictionarySize)

for p in training_paths:
    image = cv2.imread(p)
    kp, dsc= sift.detectAndCompute(image, None)
    BOW.add(dsc)

#dictionary initialization
dictionary = BOW.cluster()


sift2 = cv2.xfeatures2d.SIFT_create()
flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {}) 
bow_extract = cv2.BOWImgDescriptorExtractor(sift2, matcher )
bow_extract.setVocabulary(voc) # the 64x20 dictionary, you made before

traindata = []
trainlabels = []
for p in training_paths:
    image = cv2.imread(p)
    kp, dsc= sift.detectAndCompute(image, None)
    bowsig = bow_extract.compute(image,kp)
    traindata.extend(bowsig)
    if names_path[i]=='ref1':
        train_labels.append(1)
    if names_path[i]=='ref2':
        train_labels.append(2)
    if names_path[i]=='ref3':
        train_labels.append(3)
    if names_path[i]=='ref4':
        train_labels.append(4)
    if names_path[i]=='ref5': 
        train_labels.append(5)
    if names_path[i]=='ref6': 
        train_labels.append(6)


# Training
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setGamma(0.1)
svm.setC(2)
svm.train(np.array(train_desc), cv2.ml.ROW_SAMPLE,np.array(train_labels))

# Prediction
siftkp = sift.detect(img)
bowsig = bow_extract.compute(im, siftkp)
p = svm.predict(bowsig)
