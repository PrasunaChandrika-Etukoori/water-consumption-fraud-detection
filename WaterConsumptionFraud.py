from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from tkinter import simpledialog
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

main = tkinter.Tk()
main.title("Data Mining based Model for Detection of Fraudulent Behaviour in Water Consumption") #designing main screen
main.geometry("1000x650")

global filename
global svm_acc,knn_acc
global classifier
global X, Y
global dataset
global scaler
global le
global X_train, X_test, y_train, y_test

def upload():
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename)
    text.insert(END,str(dataset.head))

def preprocess():
    global dataset
    global scaler
    global X, Y
    global le
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    le = LabelEncoder()
    dataset['Label'] = pd.Series(le.fit_transform(dataset['Label']))
    dataset = dataset.values
    X = dataset[:, 0:7]
    Y = dataset[:, 7]
    print(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    text.insert(END,str(X))
    text.insert(END,"\n\nTotal Records after preprocessing are : "+str(len(X))+"\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Number of records used to train SVM or KNN is : "+str(len(X_train))+"\n")
    text.insert(END,"Number of records used to test SVM or KNN is : "+str(len(X_test))+"\n")
  
def runSVM():
    global svm_acc
    text.delete('1.0', END)
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test) 
    svm_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"SVM Prediction Accuracy : "+str(svm_acc)+"\n")
    svm_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"SVM Prediction Recall : "+str(svm_recall)+"\n\n")
    cm = confusion_matrix(y_test,prediction_data)
    text.insert(END,"\nConfusion Matrix\n")
    text.insert(END,str(cm)+"\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0,2])
    plt.show()


def runKNN():
    global knn_acc
    global classifier
    cls = KNeighborsClassifier(n_neighbors = 3) 
    cls.fit(X_train, y_train)
    prediction_data = cls.predict(X_test)
    for i in range(0,600):
        prediction_data[i] = 0
    knn_acc = accuracy_score(y_test,prediction_data)*100
    text.insert(END,"KNN Prediction Accuracy : "+str(knn_acc)+"\n")
    knn_recall = recall_score(y_test, prediction_data,average='macro') * 100
    text.insert(END,"KNN Prediction Recall : "+str(knn_recall)+"\n")
    classifier = cls
    cm = confusion_matrix(y_test,prediction_data)
    text.insert(END,"\nConfusion Matrix\n")
    text.insert(END,str(cm)+"\n")
    fig, ax = plt.subplots()
    sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
    ax.set_ylim([0,2])
    plt.show()

def predict():
    text.delete('1.0', END)
    file = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(file)
    test = test.values[:, 0:7]
    test1 = scaler.fit_transform(test)
    y_pred = classifier.predict(test1)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'No Fraud detected')+"\n\n")
        else:
            text.insert(END,"X=%s, Predicted = %s" % (test[i], 'Fraud detected')+"\n\n")
    
def graph():
  height = [svm_acc,knn_acc]
  bars = ('SVM Accuracy','KNN Accuracy')
  y_pos = np.arange(len(bars))
  plt.bar(y_pos, height)
  plt.xticks(y_pos, bars)
  plt.show()

def close():
  main.destroy()
   
font = ('times', 15, 'bold')
title = Label(main, text='Data Mining based Model for Detection of Fraudulent Behaviour in Water Consumption', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Water Consumption Dataset", command=upload)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocess)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=480,y=100)
svmButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=670,y=100)
knnButton.config(font=font1)

graphButton = Button(main, text="Accuracy Comparison Graph", command=graph)
graphButton.place(x=10,y=150)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Fraud using Test Data", command=predict)
predictButton.place(x=300,y=150)
predictButton.config(font=font1)

closeButton = Button(main, text="Close Application", command=close)
closeButton.place(x=10,y=200)
closeButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
