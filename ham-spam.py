import os
import pandas as pd
# Loading the data using pandas
filename = "Youtube03-LMFAO.csv"
path = "/Users/garvchhokra/Documents/"
filepath = os.path.join(path, filename)
data = pd.read_csv(filepath)
# Exploring Data
data.shape
data.head(5)
data.dtypes
feature= data[['CONTENT','CLASS']]
# removing punctuations
import string
punctuation  = list(string.punctuation)
def remove_punctuation(text): 
    for punc in punctuation:
        if punc in text:
            text = text.replace(punc, '')
    return text
feature['CONTENT'] = feature['CONTENT'].apply(remove_punctuation)
# removing stopwords
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.append("I")
feature['CONTENT'] = feature['CONTENT'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))
# Shuffling the Dataset
feature = feature.sample(frac=1).reset_index(drop=True)
# Splitting the data for taining and testing in a 75 to 25 ratio
train=feature.sample(frac=0.75,random_state=250)
train = train.reset_index(drop=True)
test = feature.drop(train.index)
test = test.reset_index(drop=True)
# Creating data for training and testing for x and y
y_train = train.drop(columns = ['CONTENT'])
y_test = test.drop(columns = ['CONTENT'])
x_train = train.drop(columns=['CLASS'])
x_test = test.drop(columns = ['CLASS'])
# vectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(x_train.CONTENT)



# 
# Find the most occurant word
# 
# 
cv_feature_names = count_vectorizer.get_feature_names()
feature_count = train_tc.toarray().sum(axis=0)
dict(zip(cv_feature_names, feature_count))
a=dict(zip(cv_feature_names, feature_count))

values_list = list(a.values())
maxFr=max(values_list)
print("One line Code Key value: ", list(a.keys())[list(a.values()).index(maxFr)])

# print(dict(zip(count_vectorizer.get_feature_names(), train_tc.toarray().sum(axis=0))))
# 
# 
# 
# 

x=pd.DataFrame(a.items(), columns=['Words', 'Frequency'])
f = max(x['Frequency'])
import numpy as np
indices = np.where(x['Frequency'] == 110)
print(indices)
indx = indices[0]
x['Words'].iloc[123]
x2 = x["Words"]




# 
# 
# 
# 


print("\nDimensions of training data:", train_tc.shape)
# tf-idf
from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
tfidf_group = tfidf.fit_transform(train_tc)
print("n_samples: %d, n_features: %d" % tfidf_group.shape)
# using MultinomialNB
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB().fit(tfidf_group, y_train)
# Calculating accuracy using Cross Validation score using 5 folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, tfidf_group, y_train, cv=5)
print(scores)
print(scores.mean())

# Now putting testing data for calculating accuracy score
test_tc = count_vectorizer.transform(x_test.CONTENT)
# test_tc = count_vectorizer.transform(x_test.CONTENT)
print("\nDimensions of training data:", test_tc.shape)
# tf-idf
tfidf_group = tfidf.transform(test_tc)
print("n_samples: %d, n_features: %d" % tfidf_group.shape)
# using MultinomialNB and passing testing data into our previous model
test_pred = classifier.predict(tfidf_group)
print(test_pred)
from sklearn.metrics import classification_report
#Print classification report
print(classification_report(y_test, test_pred))
from sklearn.metrics import confusion_matrix
#Confusion Matrix
matrix = confusion_matrix(y_test, test_pred)
print(matrix)

# Creating random comments and testing our model
final_test_X = ['I love your songs','Please follow my youtube channel https://www.youtube.com/watch?v=NQTdcumWo6I&ab_channel=HappyMemories','Awesome song','really liked the song','LMFAO is best', 'If you will like this you will have long lives',]
#Set the labels for the newly created test set
final_test_Y = [0,1,0,0,0,1]
final_test = count_vectorizer.transform(final_test_X)
final_tfidf= tfidf.transform(final_test)
final_pred = classifier.predict(final_tfidf)
print(final_pred)
print(classification_report(final_test_Y,final_pred))
final_matrix = confusion_matrix(final_test_Y, final_pred)
print(final_matrix)
