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
Our initial preference was Logistic Regression primarily because it is a supervised machine learning model. We believed that employing supervised training models would be the most suitable approach for this task, as providing them with examples of fraudulent and non-fraudulent messages could establish a solid baseline for the project. The process involves providing the model with a training set containing specific features known as predictors. These predictors help the model identify and assign a fraudulence score on a scale from 0 to 1, with 1 indicating fraudulent and 0 indicating non-fraudulent. Consequently, upon message delivery, the model promptly classifies spam messages and directs them to the bin, preventing the recipient from reading them.
As our second choice, we opted for Random Forests, an additional supervised machine learning model. The operational mechanism of this model involves constructing a substantial number of Classification and Regression Trees (CART), which collectively cast votes on the outcome of a new observation, ultimately selecting the outcome that gathers the majority of votes. Random Forests display user-friendliness, characterized by ease of use, and they boast a reduced number of hyperparameters in comparison to more intricate models. Furthermore, we found Random Forests to be a compelling option due to their remarkable versatility in handling various data types.
Our ultimate selection was the Support Vector Machine (SVM). The core aim of this model is to identify a hyperplane that optimally divides the data into distinct classes. SVM excels particularly in high-dimensional spaces and proves effective when dealing with non-linearly separable data. It achieves this by employing the kernel trick, a technique that transforms input features into a higher-dimensional space, enabling SVM to effectively manage intricate relationships within the data. Notably, SVM's versatility extends to its capability to perform well with relatively small and imbalanced datasets, contributing to its widespread popularity among various AI models.

To carry out this job we made use of some libraries such as pandas (used specifically for data manipulation and analysis) and sklearn; in particular we used train_test_split (to split the dataset into subsets that minimize the potential for bias in the evaluation and validation process), TfidfVectorizer (to convert raw documents to a matrix of TF-IDF features), SVC (to fit to the data provided, returning a "best fit" hyperplane that divides, or categorizes the data), classification_report (to compute the accuracy of a classification model based on the values from the confusion matrix), accuracy_score (to calculate the accuracy score for a set of predicted labels against the true labels), confusion_matrix (to evaluate the performance of a machine learning algorithm), and finally Pipeline (to link all steps of data manipulation together to create a pipeline). 

(*to add conda envexport and conda list*)
(*files too?*)
*• It may help to include a figure illustrating your ideas, e.g., a flowchart illustrating the steps in your machine learning system(s)*


[Section 3]   (*TO ADD SOME DETAILS ABOUT THE EVALUATION METRICS FOR EACH PARAGRAPH*)
In constructing our initial model, Logistic Regression, our initial approach involved categorizing messages as fraudulent or non-fraudulent based solely on the words present in those messages. Initially, we employed the TF-IDF vectorizer for this purpose and saw that there was correlation between some specific words used and the oucome of the message being fraudolent (*ADD EXAMPLES OF WORDS THAT ARE IN FRAUDOLENT MESSAGES*) Subsequently, we introduced an additional parameter, namely the length of the messages. Notably, incorporating this new parameter yielded improved results, indicating that longer messages are more likely to be fraudulent. Additionally, we introduced a third parameter involving the days of the week to explore potential correlations between specific days and the likelihood of messages being fraudulent. However, the results indicated an absence of any significant correlation in that regard.

Based on the outcomes obtained with our initial model, when transitioning to the second model, we decided to keep just the two parameters that provided significant insights: the presence of specific words and the length of sentences. While the Random Forests model yielded overall improved results compared to our initial model, it displayed a higher count of false negatives than false positives. Additionally, the recall obtained with Random Forests was not as robust as the one we achieved with Logistic Regression.

Our third and ultimate model proved to be the most successful among the three iterations. Despite utilizing the same two parameters as in the previous models—specifically, the presence of certain words and the length of sentences—this time, the model demonstrated a remarkable reduction in errors. The total number of errors amounted to a mere 23 instances, comprising 21 false negatives and only 2 false positives. This noteworthy improvement underscores the enhanced accuracy and precision achieved by the third model, contributing to its effectiveness in minimizing both types of classification errors.

[Section 4]
In conclusion, our journey through model development and refinement unveiled valuable insights into enhancing the efficacy of our fraud detection system. Beginning with Logistic Regression, we initially focused on word-based categorization, identifying specific words correlated with fraudulent messages — *(examples include "phishing," "unauthorized," and "scam.")* The introduction of message length as an additional parameter significantly improved our model's performance, revealing a noteworthy association between longer messages and fraudulence. Subsequently, the exploration of days of the week failed to establish any significant correlation with fraudulent activities.

Transitioning to our second model with Random Forests, we maintained the crucial parameters of word presence and message length. While this model exhibited an overall improvement, its increased false negatives and a less robust recall, compared to Logistic Regression, highlighted the nuanced trade-offs in model selection.

The peak of our efforts, the third and ultimate model, showcased substantial progress. By refining the Logistic Regression framework with the same parameters, our model achieved an outstanding reduction in errors, totaling only 23 instances. Impressively, the model demonstrated a mere 2 false positives and 21 false negatives, accentuating its heightened accuracy and precision. This success underscores the effectiveness of leveraging specific features and parameters, such as word presence and message length, in refining a fraud detection model, thereby minimizing classification errors and enhancing overall performance.

(*TO ADD*)
*• Include at least one placeholder figure and/or table for communicating your findings*
*• All the figures containing results should be generated from the code.*

[Section 5] 
*conclusions*
*List some concluding remarks. In particular:*
*• Summarize in one paragraph the take-away point from your work.*
*• Include one paragraph to explain what questions may not be fully answered by your work as well as natural next steps for this direction of future work*
