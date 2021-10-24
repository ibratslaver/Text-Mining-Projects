import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import re
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_excel('LDA_TRAINING_VALIDATION.xlsx', sheet_name='Sheet1')
d
# X -> features, y -> label 
x = df.sentence 
y = df.topic

# dividing X, y into train and test data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

#Vectorize the training x
v = TfidfVectorizer()
#v = CountVectorizer(stop_words = {'english'})
x_train_vector = v.fit_transform(x_train)

#re.compile("\s([a-zA-Z]+)\s", re.I))

columns = v.get_feature_names()
df1 = pd.DataFrame(x_train_vector.toarray(), columns=columns)
df1.to_csv('x_train_vector.csv')

# Classifier - Algorithm - SVM
SVM = svm.SVC(C=1.6, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train_vector,y_train)

#transform test data to fit training framework
x_test_vector = v.transform(x_test)

# predict the labels on validation dataset
pred = SVM.predict(x_test_vector)

#confusion matrix
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')

conf = confusion_matrix(y_actu, y_pred, labels=["Exam and Assignment", "Other", "Pedagogy", "Personality"])
cm = pd.DataFrame(conf)

print(cm)

#accuracy and other metrics
print(accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

#cross validation stuff
x_vector = v.fit_transform(x)
cross_val = cross_val_score(SVM, x_vector, y, cv = 10)
print(cross_val)







