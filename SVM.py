import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Load the dataset
dataset = pd.read_csv('C:/Users/Super/Documents/GitHub/SafeCommDigitalSecurity/Dataset.csv')

# Separating the features and the target variable
X = dataset['SMS test']  # Feature - SMS text
y = dataset['Fraudolent']  # Target - Fraudulent or not

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a text processing and SVM classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),  # Text vectorization
    ('svm', SVC(kernel='linear'))  # SVM classifier with a linear kernel
])

# Training the model
pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = pipeline.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Displaying the results
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
