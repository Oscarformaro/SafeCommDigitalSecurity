#first, we do EDA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
path="./sms.csv"
sms_df = pd.read_csv(path)

#now we check for the missing values
missing_values=sms_df.isnull().sum()
print("Missing values:")
print(missing_values)

column_to_analyze = 'Fraudolent'

sns.histplot(sms_df, x = column_to_analyze, hue='Fraudolent', multiple = 'stack')

#we check if there is a correlation between the lenght of a message and if it is fraudolent or not
# Create a new feature for the length of each message
sms_df['message_length'] = sms_df['SMS test'].apply(len)

# Create a new feature for the number of words in each message
sms_df['word_count'] = sms_df['SMS test'].apply(lambda x: len(str(x).split()))

sns.histplot(sms_df, x='message_length', hue='Fraudolent', multiple='stack')
plt.show()

sns.histplot(sms_df, x='word_count', hue='Fraudolent', multiple='stack')
plt.show()


sms_df['Date and Time'] = pd.to_datetime(sms_df['Date and Time'])
# Create a new feature for the hour of the day
sms_df['hour'] = sms_df['Date and Time'].dt.hour

# Create a new feature for the day of the week
sms_df['day_of_week'] = sms_df['Date and Time'].dt.dayofweek

sns.histplot(sms_df, x='hour', hue='Fraudolent', multiple='stack')
plt.show()

sns.histplot(sms_df, x='day_of_week', hue='Fraudolent', multiple='stack')
plt.show()



from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

# Tokenize the text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sms_df['SMS test'])

# Now, apply OneHotEncoder
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X.toarray())

# X_encoded will now be a numpy array. You can convert it back to DataFrame if you want:
X_encoded_df = pd.DataFrame(X_encoded.toarray())


# Preprocessing the text data
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sms_df['SMS test'])

# Converting the target column to numpy array
y = sms_df['Fraudolent'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Computing Accuracy, Precision, Recall and F1 Score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print("Accuracy =", accuracy_score(y_test,y_pred))
print("Precision =", precision_score(y_test,y_pred))
print("Recall =", recall_score(y_test,y_pred))
print("F1 Score =", f1_score(y_test,y_pred))

print("Confusion Matrix:")
print(cm)