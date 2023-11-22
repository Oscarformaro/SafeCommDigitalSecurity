# Importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Loading the dataset
dataset = pd.read_csv('C:/Users/Super/Documents/GitHub/SafeCommDigitalSecurity/Dataset.csv')

# Calculating the length of each email
dataset['EmailLength'] = dataset['SMS test'].apply(len)

# Vectorizing the email texts
tfidf_vectorizer = TfidfVectorizer(max_features=500)  # Limiting to 500 features for simplicity
X_text = tfidf_vectorizer.fit_transform(dataset['SMS test']).toarray()

# Combining text features with Email Length
X_length = dataset[['EmailLength']].values
X = np.hstack((X_text, X_length))

# Target variable
y = dataset['Fraudolent']

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Creating the Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Computing Accuracy, Precision, Recall, and F1 Score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Outputting the results
print("Confusion Matrix:\n", cm)
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

import matplotlib.pyplot as plt

# Function to plot the results
def plot_results(X, y, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0], y[y == 0], color='green', label='Not Fraudulent', alpha=0.5)
    plt.scatter(X[y == 1], y[y == 1], color='red', label='Fraudulent', alpha=0.5)
    plt.scatter(X[y_pred != y], y_pred[y_pred != y], color='blue', label='Misclassified', alpha=0.5)
    plt.title(title)
    plt.xlabel('Email Length')
    plt.ylabel('Class')
    plt.legend()
    plt.show()

# Predicting for the training set
y_train_pred = classifier.predict(X_train)

# Plotting for the training set
plot_results(X_train[:, -1], y_train, y_train_pred, 'Random Forest - Training Set')

# Plotting for the test set
plot_results(X_test[:, -1], y_test, y_pred, 'Random Forest - Test Set')
