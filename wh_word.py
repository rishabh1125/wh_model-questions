#Step1: importing dataset
import pandas as pd
dataset = pd.read_csv('sample.tsv', delimiter = ';', quoting = 3)

#Step2: data perprocessing and form corpus
import re
import nltk
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 11):
    review = re.sub('[^a-zA-Z]', ' ', dataset['sentence'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = ' '.join(review)
    corpus.append(review)

#Step3: Convert corpus to array list of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values    

#Step4: Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

Step5: #Naive Bayes classification model
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

#Step5: Model testing using confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
