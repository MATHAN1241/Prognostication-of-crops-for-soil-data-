

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from time import sleep

# Importing the dataset
dataset = pd.read_csv('Crop_recommendation.csv')
print(dataset)

dataset.dropna(how="any")

print(dataset)

print(dataset.info())

print(dataset['label'].value_counts())

dataset['label'] = dataset['label'].map({"rice":0,"maize":1,"chickpea":2, "kidneybeans":3,"pigeonpeas":4,"mothbeans":5
                                         ,"mungbean":6,"blackgram":7,"lentil":8,"pomegranate":9,"banana":10
                                         ,"mango":11,"grapes":12,"watermelon":13,"muskmelon":14,"apple":15
                                         ,"orange":16,"papaya":17,"coconut":18,"cotton":19,"jute":20
                                         ,"coffee":21})



print(dataset)

plt.figure(figsize=(10,8))
plt.title("Histogram of outcome")
plt.hist(dataset["label"],rwidth=0.9)
plt.show()

dataset.dropna(inplace=True)
print(dataset.info())



dataset.to_csv('preprocessed dataset.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 121)

model1 = RandomForestClassifier(n_estimators=20)
print("Random Forest Training")
model1.fit(X_train, y_train)
sleep(5)
print("Training Completed")
y_pred = model1.predict(X_test)
print("ypred : ")
print(y_pred)

#confussion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix for Random forest")
print(cm)

cm_df = pd.DataFrame(cm,
                     index = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans","mungbean","blackgram","lentil","pomegranate","banana","mango","grapes","watermelon","muskmelon","apple","orange","papaya","coconut","cotton","jute","coffee"], 
                     columns = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans"
                                         ,"mungbean","blackgram","lentil","pomegranate","banana"
                                         ,"mango","grapes","watermelon","muskmelon","apple"
                                         ,"orange","papaya","coconut","cotton","jute","coffee"])
#Plotting the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix of Random Forest')
plt.show()

rf = accuracy_score(y_test, y_pred)

print("Random Forest accuracy is ")
print(rf)
print("")

testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes, average = 'micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes, average = 'micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average = 'micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

#svm training
print("svm training")
model2 =svm.SVC(degree = 1, kernel = 'sigmoid')
print("SVM Training")
model2.fit(X_train, y_train)
sleep(5);
print("Training Completed")
y_pred = model2.predict(X_test)
print("ypred of SVM ")
print(y_pred)

#confussion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix for SVM")
print(cm)

cm_df = pd.DataFrame(cm,
                     index = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans"
                                         ,"mungbean","blackgram","lentil","pomegranate","banana"
                                         ,"mango","grapes","watermelon","muskmelon","apple"
                                         ,"orange","papaya","coconut","cotton","jute","coffee"], 
                     columns = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans"
                                         ,"mungbean","blackgram","lentil","pomegranate","banana"
                                         ,"mango","grapes","watermelon","muskmelon","apple"
                                         ,"orange","papaya","coconut","cotton","jute","coffee"])
#Plotting the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix of SVM')
plt.show()

svma = accuracy_score(y_test, y_pred)

print("SVM accuracy is ")
print(svma)
print("")

testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes, average = 'micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes, average = 'micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average = 'micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

#Logistic regression training
print("LR training")
model3=LogisticRegression()
print("Logistic Regressioin Training")
model3.fit(X_train, y_train)
sleep(5)
print("Training Completed")
y_pred = model3.predict(X_test)
print("ypred of LR ")
print(y_pred)

#confussion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confussion Matrix for LR")
print(cm)

cm_df = pd.DataFrame(cm,
                     index = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans"
                                         ,"mungbean","blackgram","lentil","pomegranate","banana"
                                         ,"mango","grapes","watermelon","muskmelon","apple"
                                         ,"orange","papaya","coconut","cotton","jute","coffee"], 
                     columns = ["rice","maize","chickpea", "kidneybeans","pigeonpeas","mothbeans"
                                         ,"mungbean","blackgram","lentil","pomegranate","banana"
                                         ,"mango","grapes","watermelon","muskmelon","apple"
                                         ,"orange","papaya","coconut","cotton","jute","coffee"])
#Plotting the confusion matrix
plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True)
plt.title('Confusion Matrix of Logistic Regression')
plt.show()

lrc = accuracy_score(y_test, y_pred)

print("Logistic Regression accuracy is ")
print(lrc)
print("")

testy = y_test
yhat_classes = y_pred
precision = precision_score(testy, yhat_classes, average = 'micro')
print('Precision: %f' % precision)
recall = recall_score(testy, yhat_classes, average = 'micro')
print('Recall: %f' % recall)
f1 = f1_score(testy, yhat_classes, average = 'micro')
print('F1 score: %f' % f1)
 
# kappa
kappa = cohen_kappa_score(testy, yhat_classes)
print('Cohens kappa: %f' % kappa)

data = {'Random forest':rf, 'SVM':svma, 'Logistic Regression':lrc}
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='maroon',
        width = 0.4)
 
plt.xlabel("ML Algorithms")
plt.ylabel("Accuracy")
plt.title("Performance")
plt.show()

if((rf>svma)and(rf>lrc)):
    print("Random forest ALgorithm is the best")
    flg=1
elif((svma>rf)and(svma>lrc)):
    print("SVM is best")
    flg=2
else:
    print("Logistic Regression is best")
    flg=3


import pickle
trPickle = open('training_pickle_file', 'wb')
if(flg==1):
    pickle.dump(model1, trPickle)
elif(flg==2):
    pickle.dump(model2, trPickle)
elif(flg==3):
    pickle.dump(model3, trPickle)
trPickle.close()

