# importing necessary libraries 
from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

df = pd.read_excel('LDA_TRAINING_VALIDATION.xlsx', sheet_name='Sheet1')

# X -> features, y -> label 
x = df.sentence 
y = df.topic

# dividing X, y into train and test data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .20, random_state = 0)

#Vectorize the training x
v = CountVectorizer(stop_words = {'english'}, analyzer = 'word', binary = True)
x_train_vector = v.fit_transform(x_train)

columns = v.get_feature_names()
df1 = pd.DataFrame(x_train_vector.toarray(), columns=columns)
df1.to_csv('x_train_vector.csv')

# Initialize naive bayes model 
clf = MultinomialNB().fit(x_train_vector.toarray(), y_train)

#transform test data to fit training framework
x_test_vector = v.transform(x_test)

#do prediction
pred = clf.predict(x_test_vector)

#Create confusion matrix
y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(pred, name='Predicted')

conf = confusion_matrix(y_actu, y_pred, labels=["Exam and Assignment", "Other", "Pedagogy", "Personality"])
cm = pd.DataFrame(conf)

print(cm)

#Accuracy and Other Metrics
print(accuracy_score(y_test, pred))

print(classification_report(y_test, pred))

#Cross Validation
x_vector = v.fit_transform(x)
cross_val = cross_val_score(clf, x_vector, y, cv = 10)
print(cross_val)
 

