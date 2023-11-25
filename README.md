# SafeCommDigitalSecurity

SafeComm Digital Security Solutions AI/ML Group Project 23/24
Member 1 (captain): Riccardo Aversa 283831
Member 2: Alice Alessandrelli 279371
Member 3: Oscar Maria Formaro 260581

[Section 1]
We were assigned to work on the SafeComm project: a company that aims at eliminating SMS-based fraud. Nowadays we all communicate mainly through text messages on our mobile phones and people have taken the opportunity to scam people through them. 
SafeComm's purpose is exactly that of minimizing these crooked messages or emails that we get. Starting from there we were asked to test three machine learning models that were able to carry out this type of work efficiently. 

[Section 2]
In order to do so, we decided to rely on these three ML models: Logistic Regression, Random Forests and SVM (Support Vector Machine); we'll now go through each one of them and our reasonings behid each choice:
Logistic Regression was our first choice as it is, first of all, a supervised machine learning model. We thought that the best approach for this type of work would be that of supervised training models because giving them examples of what are fraudolent messages and what are not, would be a good baselise to start with. The way it works is by giving the model a tranining set with some given features called predictors, which would detect and score on a scale from 0 to 1 as fraudolent (1) or not fraudolent (0). This way, as soon as the messages are delivered, the model would classify the spam messages as such and directly transfer them to the bin without having the recipient read them. 
Our second choice was Random Forests, another supervised ML model; this model works by builiding a large number of CART trees that vote on the outcome of a new observation and pick the outcome that receives the majority of votes. Random Forests are also relatively easy to use, and they come with fewer hyperparameters to tune compared to some other complex models. We thought it would be a good choice also because of its versatility in data types.
Our third and final choice was SVM (Support Vector Machine). The primary objective of this model is to find a hyperplane that best separates the data into different classes. SVM also works very well in high-dimensional spaces and is effective when the data is not linearly separable by transforming the input features into a higher-dimensional space through a process called the kernel trick, thus allowing SVM to handle complex relationships in the data. Another great feature about SVM, and also one of the reason why it is so popular amongst AI models is that it works well with relatively small and also imbalanced datasets. 

To carry out this job we made use of some libraries such as pandas, used specifically for data manipulation and analysis; and sklearn, in particular we used train_test_split (to split the dataset into subsets that minimize the potential for bias in the evaluation and validation process), TfidfVectorizer (to convert raw documents to a matrix of TF-IDF features), SVC (to fit to the data provided, returning a "best fit" hyperplane that divides, or categorizes the data), classification_report (to compute the accuracy of a classification model based on the values from the confusion matrix), accuracy_score (to calculate the accuracy score for a set of predicted labels against the true labels), confusion_matrix (to evaluate the performance of a machine learning algorithm), and finally Pipeline (to link all steps of data manipulation together to create a pipeline). 
##to add conda envexport and conda list## 
##files too?##

[Section 3]
To build our first model (Logistic Regression), we decided, in a first moment, to base the division of the messages in fraudolent and non-fraudolent on solely the words appearing in said messages. To do so we used the tfid vectoriser. After that we decided to add another parameter that is the length of the messages. What we saw was that adding this new parameter led us to better results, though meaning that the longer the messages are, the more probable is that they are fraudolent.


[Section 4]
*results*

[Section 5] 
*conclusions*
