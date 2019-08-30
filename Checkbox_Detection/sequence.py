import numpy as np
import cv2
from matplotlib import pyplot as plt
import math
import time
import os
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
"""
    preprocess gives all contours of the input img
    args :
    - img : the input image
"""
def preprocess(img):
    lp = cv2.Laplacian(img,cv2.CV_64F)
    bw = cv2.adaptiveThreshold(lp, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    return bw

"""
    neigboorL if there is an intersection around the pixel (h,l) on X-axis
    args : 
    - img : the input img
    - h,l : coordinates of the point around which the function checks if there is an intersection
    - decroiss : boolean value refering to what side of the pixel (true = top, false = bottom) the function checks if there is an intersection
"""

def neigboorL(img, h, l, decroiss):
    if(decroiss):
        if((img[h, l] == 255 and img[h-1, l] == 255)or(img[h, l] == 255 and img[h-1, l-1] == 255)or(img[h, l] == 255 and img[h-1, l+1] == 255)):
            return True
        else:
            return False
    else:
        if((img[h, l] == 255 and img[h+1, l] == 255)or(img[h, l] == 255 and img[h+11, l-1] == 255)or(img[h, l] == 255 and img[h+1, l+1] == 255)):
            return True
        else:
            return False

"""
    neigboorH if there is an intersection around the pixel (h,l) on Y-axis
    args : 
    - img : the input img
    - h,l : coordinates of the point around which the function checks if there is an intersection
    - decroiss : boolean value refering to what side of the pixel (true = left, false = right) the function checks if there is an intersection
"""

def neigboorH(img, h, l, decroiss):
    if(decroiss):
        if((img[h, l] == 255 and img[h, l-1] == 255)or(img[h, l] == 255 and img[h-1, l-1] == 255)or(img[h, l] == 255 and img[h+1, l-1] == 255)):
            return True
        else:
            return False
    else:
        if((img[h, l] == 255 and img[h, l+1] == 255)or(img[h, l] == 255 and img[h-1, l+1] == 255)or(img[h, l] == 255 and img[h+1, l+1] == 255)):
            return True
        else:
            return False

"""
    countIt returns the sequence of counted intersections around a specific window for one checkbox
    args : 
    - img : the input img
    - window : array which contains the coordinates of the first window
    - pas : the step of the window expanding at each iteration
    - ite : number of total iterations
    - title : name of the checkbox
"""

def countIt(img, window, pas, ite, title):
    count = []
    for i in range(0, ite):
        min_h = window[0]-(pas*i)
        max_h = window[1]+(pas*i)
        min_l = window[2]-(pas*i)
        max_l = window[3]+(pas*i)
        c = 0
        #count intersections on the y-axis
        for h in range(min_h, max_h+1):
            if(neigboorH(img, h, min_l, True) == True or neigboorH(img, h, max_l, False) == True):
                c = c+1
        #count intersections on the x-axis
        for l in range(min_l, max_l+1):
            if(neigboorL(img, min_h, l, True) == True or neigboorL(img, max_h, l, False) == True):
                c = c+1
        count.append(c)
    return count

"""
    processCount returns a list of all the sequences of counted intersections around a specific window for a list of checkboxes
    args : 
    - img : the input img
    - checkbox : dict of checkboxes referenced by their name and values are sub-images of checkboxes 
    - fn : filename
"""

def processCount(img, checkbox, fn):
    global svm_X
    all_c = []
    for elt in checkbox:
        name = elt["name"]
        count = countIt(img, elt["value"], 1,60, name)
        all_c.append([count,name])
    svm_X.append(all_c)


"""
    checkWindow split the input image into sub-images which are centered on a checkbox
    args :
    - image : the input image
    - fn : filename of the image
"""

def checkWindow(image, fn):

    img = preprocess(image)

    checkbox_M = [91, 171, 1841, 1927]
    checkbox_F = [191, 271, 1841, 1927]

    checkbox_Grossese_Yes = [1153, 1157, 1811, 1815]
    checkbox_Grossese_No = [1153, 1157, 2005, 2009]

    checkbox_Travail_Yes = [1378, 1382, 1536, 1540]
    checkbox_Travail_No = [1378, 1382, 1703, 1707]
    checkbox_Travail_SansPrecision = [1378, 1382, 1893, 1897]

    checkbox_Autopsie_Non = [1482, 1486, 104, 108]
    checkbox_Autopsie_Disponible = [1482, 1486, 354, 358]
    checkbox_Autopsie_NonDisponible = [1541, 1545, 112, 116]

    checkbox_Lieu_Domicile = [60, 63, 115, 118]
    checkbox_Lieu_Hopital = [60, 63, 672, 675]
    checkbox_Lieu_Clinique = [60, 63, 1029, 1032]
    checkbox_Lieu_MaisonRetraite = [144, 147, 115, 118]
    checkbox_Lieu_VoiePublique = [144, 147, 672, 675]
    checkbox_Lieu_Autre = [144, 147, 1029, 1032]

    # List of all checkbox
    checkbox_List = [
        {"name": "checkbox_Grossese_Yes", "value": checkbox_Grossese_Yes},
        {"name": "checkbox_Grossese_No", "value": checkbox_Grossese_No},
        {"name": "checkbox_Travail_Yes", "value": checkbox_Travail_Yes},
        {"name": "checkbox_Travail_No", "value": checkbox_Travail_No},
        {"name": "checkbox_Travail_SansPrecision",
            "value": checkbox_Travail_SansPrecision},
        {"name": "checkbox_Autopsie_Non", "value": checkbox_Autopsie_Non},
        {"name": "checkbox_Autopsie_Disponible",
            "value": checkbox_Autopsie_Disponible},
        {"name": "checkbox_Autopsie_NonDisponible",
            "value": checkbox_Autopsie_NonDisponible},
        {"name": "checkbox_Lieu_Domicile", "value": checkbox_Lieu_Domicile},
        {"name": "checkbox_Lieu_Hopital", "value": checkbox_Lieu_Hopital},
        {"name": "checkbox_Lieu_Clinique", "value": checkbox_Lieu_Clinique},
        {"name": "checkbox_Lieu_MaisonRetraite",
            "value": checkbox_Lieu_MaisonRetraite},
        {"name": "checkbox_Lieu_VoiePublique",
            "value": checkbox_Lieu_VoiePublique},
        {"name": "checkbox_Lieu_Autre", "value": checkbox_Lieu_Autre}
    ]

    checkbox_Sexe = [
        {"name": "checkbox_M", "value": checkbox_M},
        {"name": "checkbox_F", "value": checkbox_F}
    ]

    checkbox_Grossese = [
        {"name": "checkbox_Grossese_Yes", "value": checkbox_Grossese_Yes},
        {"name": "checkbox_Grossese_No", "value": checkbox_Grossese_No}
    ]

    checkbox_Travail = [
        {"name": "checkbox_Travail_Yes", "value": checkbox_Travail_Yes},
        {"name": "checkbox_Travail_No", "value": checkbox_Travail_No},
        {"name": "checkbox_Travail_SansPrecision",
            "value": checkbox_Travail_SansPrecision}
    ]

    checkbox_Autopsie = [
        {"name": "checkbox_Autopsie_Non", "value": checkbox_Autopsie_Non},
        {"name": "checkbox_Autopsie_Disponible",
            "value": checkbox_Autopsie_Disponible},
        {"name": "checkbox_Autopsie_NonDisponible",
            "value": checkbox_Autopsie_NonDisponible}
    ]

    checkbox_Lieu = [
        {"name": "checkbox_Lieu_Domicile", "value": checkbox_Lieu_Domicile},
        {"name": "checkbox_Lieu_Hopital", "value": checkbox_Lieu_Hopital},
        {"name": "checkbox_Lieu_Clinique", "value": checkbox_Lieu_Clinique},
        {"name": "checkbox_Lieu_MaisonRetraite",
            "value": checkbox_Lieu_MaisonRetraite},
        {"name": "checkbox_Lieu_VoiePublique",
            "value": checkbox_Lieu_VoiePublique},
        {"name": "checkbox_Lieu_Autre", "value": checkbox_Lieu_Autre}
    ]

    processCount(img, checkbox_Lieu, fn)

"""
    case_value checks if case is defined or not and cast it into an integer
    args : 
    - case : string which refers to the thicked checkbox number
"""
def case_value(case):
    if(case=="NULL"):
            return 0
    else:
        return int(case)

"""
    load_y loads the label of an image from a csv file
    args :
    - fn : filename of the image 

"""
def load_y(fn):
    global svm_y
    fn = fn[:-1]
    f = open("Lieux_4300-4310.csv", "r")
    for line in f:
        tmp = line.split("\t")
        ref = int(tmp[0])
        if(ref == int(fn)):
            case = tmp[1].split("\n")[0]
            value = case_value(case)
            svm_y.append(value)
            return True
    return False

def main():
    global svm_X, svm_y
    svm_X = []
    svm_y = []
    from glob import glob
    
    for fn in glob('Dataset/*.png'):
        img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        if(img is not None):
            if(load_y(fn.split("\\")[-1].split(".")[0])):
                #compute all sequences for each image in the dataset
                checkWindow(img, fn)
                
    import simplejson
    fd = open('svm_X.txt', 'w')
    simplejson.dump(svm_X, fd)
    fd.close()
    ft = open('svm_y.txt', 'w')
    simplejson.dump(svm_y, fd)
    ft.close()
        


if __name__ == '__main__':
    start = time. time()
    main()
    end = time. time()
    print(end - start)
