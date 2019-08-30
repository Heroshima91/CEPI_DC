import numpy as np
import cv2
from matplotlib import pyplot as plt
import math, time, os

"""
    deskew transforms the input image with the calculed transformation
    args : 
    - skewed_image : image to be deskewed
    - orig_image : reference image
    - M : matrix which contained parameters of the transformation
    - classification : integer which identifies the template
"""
def deskew(skewed_image,orig_image,M,fn,classification):
    im_out = cv2.warpPerspective(skewed_image, np.linalg.inv(M), (orig_image.shape[1], orig_image.shape[0]))
    cv2.imwrite(os.path.join("Center\\"+str(classification),fn),im_out)

"""
    projection finds the best alignment between the image and the reference which matched the image
    args :
    - skewed_image : the image to be aligned
    - kp2 : key points of the image to be aligned
    - orig_image : the reference image
    - kp1 : key points of the reference image
    - good : list of keypoints that are close
    - fn : name of the input image
    - classification : integer which identifies the template
"""
def projection(skewed_image,kp2,orig_image,kp1,good,fn,classification):
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #Find transformation with RANSAC algorithm
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print(M)
        # see https://ch.mathworks.com/help/images/examples/find-image-rotation-and-scale-using-automated-feature-matching.html for details
        ss = M[0, 1]
        sc = M[0, 0]
        scaleRecovered = math.sqrt(ss * ss + sc * sc)
        thetaRecovered = math.atan2(ss, sc) * 180 / math.pi
        print("Calculated scale difference: %.2f\nCalculated rotation difference: %.2f" % (scaleRecovered, thetaRecovered))
        #check if the transformation is not aberrant
        if(scaleRecovered >= 0.85 and scaleRecovered <= 1.2 and thetaRecovered >= -0.3 and thetaRecovered <= 0.3):
            deskew(skewed_image,orig_image,-M,fn,classification)
        else:
            print('Problem')
            cv2.imwrite(os.path.join("Center\\unclassified",fn),im_out)

    else:
        print("Not  enough  matches are found   -   %d/%d" % (len(good), MIN_MATCH_COUNT))

"""
    bestMatch finds the best template for the image
    args : 
    - img : input image
    - detector : cv2 implementation of a detector (Akaze, Surf, Sift, ORB ...)
    - pts : list of tuple (keypoints,descriptors) for all templates
    - fn : name of the input image
    - txt : file txt for log information about classification
    - imgs : all images of template

"""
def bestMatch(img,detector,pts,fn,txt,imgs):
    kp, descs = detector.detectAndCompute(img, None) 
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    ref_sim = []
    final_good = []

    # compare all descriptors between image and each template
    for bestpt in pts:
        matches = bf.knnMatch(bestpt[1],descs, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                good.append(m)
        similiarty = len(good)/len(bestpt[0])
        ref_sim.append(similiarty)
        final_good.append(good)

    #get the best matching template 
    classification = ref_sim.index(max(ref_sim))
    txt.write(fn+" matched for "+str(classification)+" with prob : "+ str(max(ref_sim))+"\n")
    txt.write(str(ref_sim)+"\n")
    
    #check up if there are enough matching between the two images
    if(max(ref_sim)>=0.45):
        projection(img,kp,imgs[classification],pts[classification][0],final_good[classification],fn,classification)
    else:
        cv2.imwrite(os.path.join("Center\\unclassified",fn),img)
    

"""
    createFolder creates folder to store each image deskewed
    args :
    - name : name of the main folder
    - count : number of subfolder to be created (equals to number of image in the folder template)
"""

def createFolder(name,count):
    for i in range(count):
        path = name+"/"+str(i)
        try:  
            os.makedirs(path)
        except OSError:  
            print ("Creation of the directory %s failed" % path)
        else:  
            print ("Successfully created the directory %s " % path)
    try:  
        os.makedirs(name+"/unclassified")
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
        




def main():
    references = []
    detector = cv2.AKAZE_create(threshold = 0.01)
    pts = []
    count = 0
    from glob import glob
    #Compute all key points from templates
    for fn in glob('Template/*.png'):
        ref = cv2.imread(fn)
        kp, descs = detector.detectAndCompute(ref,None)
        references.append(ref)
        pts.append([kp,descs])
        count = count + 1

    createFolder("Center",count)

    txt = open("record.txt","w")

    from glob import glob
    for fn in glob('Dataset/*.png'):
        img = cv2.imread(fn)
        #Find the best match and deskew for each image
        bestMatch(img,detector,pts,fn.split("\\")[1],txt,references)
        #print(fn.split("\\")[1])
    txt.close()


if __name__ == '__main__':
    start = time. time()
    main()
    end = time. time()
    print(end - start)
