import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC



# Read data from file
df = pd.read_csv('export_dataframe.csv')


# store review in list x and label in list y
X = df['review']
y = df['label']

# split data to train and test data and test size 30% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



# Build model with linearSVC classifier
txt_lsvc = Pipeline([('tfidf', TfidfVectorizer()),
                     ('clf', LinearSVC()),])
txt_lsvc.fit(X_train, y_train)
y_predicted = txt_lsvc.predict(X_test)


# Calculate accuracy of the model
counter = 0
for i in range(len(y_test)):
    if y_predicted[i] == list(y_test)[i]:
        counter += 1
print("The accuracy of the model is : ",round((counter/len(y_test))*100,2))


# Take input from user
userInput = input('Enter your review: ')
predicted = txt_lsvc.predict([userInput])
if predicted == 1:
    print("It's Positive Review")
else:
    print("It's Negative Review")
