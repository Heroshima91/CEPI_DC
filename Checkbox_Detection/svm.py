import simplejson
from matplotlib import pyplot as plt
import time
import os
from sklearn.model_selection import train_test_split  
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score
from sklearn import neighbors
import joblib

def preprocess():
    fd = open('svm_X.txt', 'r')
    X = simplejson.load(fd)
    fd.close()
    ft = open('svm_y.txt', 'r')
    Y = simplejson.load(ft)
    ft.close()
    svm_X = []
    svm_y = []
    for i in range(0,len(X)):
            for j in X[i]:
                svm_X.append(j)
            l = [0,0,0,0,0,0]
            if(Y[i]>0):
                l[Y[i]-1] = 1
            svm_y = svm_y + l 
    return svm_X, svm_y


def main():
    X, y = preprocess()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    #perform grid_search
    C_values = [0.001,0.01,0.1,1,10,100,1000]
    Gamma_values = [0.0001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,100,500,1000]
    best_score = 0
    best_params = {'C':None,'gamma':None}
    
    for C in C_values:
        for gamma in Gamma_values:
            svclassifier = SVC(C=C,gamma=gamma,kernel='rbf')
            svclassifier.fit(X_train, y_train) 
            score = svclassifier.score(X_test,y_test)
            print(score)
            if(score>best_score):
                best_score=score
                best_params['C']=C
                best_params['gamma']=gamma
                #dump the best 
                joblib.dump(svclassifier, "svm.joblib")
    print(best_score)
    print(best_params)



if __name__ == '__main__':
    start = time. time()
    main()
    end = time. time()
    print(end - start)
