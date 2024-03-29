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
from tslearn.metrics import dtw


"""
    preprocess loads the dataset and convert labels into sparse matrix of labels
"""
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
    errors = []
    #get the best k 
    for k in range(1,15):
        knn = neighbors.KNeighborsClassifier(k,algorithm='ball_tree',metric=dtw)
        c = knn.fit(X_train, y_train)
        errors.append(100*(1 - c.score(X_test, y_test)))
        predict = c.predict(X_test)
        #dump each model
        joblib.dump(c, str(k)+"nn_model.joblib")
        print(confusion_matrix(y_test,predict))
        print(classification_report(y_test,predict))
    plt.plot(range(1,15), errors, 'o-')
    plt.show()

if __name__ == '__main__':
    start = time. time()
    main()
    end = time. time()
    print(end - start)
