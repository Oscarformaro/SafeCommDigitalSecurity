# SafeCommDigitalSecurity

SafeComm Digital Security Solutions AI/ML Group Project 23/24
Member 1 (captain): Riccardo Aversa 283831
Member 2: Alice Alessandrelli 279371
Member 3: Oscar Maria Formaro 260581


[Section 1]
We were assigned to work on the SafeComm project: a company that aims at eliminating SMS-based fraud. Nowadays we all communicate mainly through text messages on our mobile phones and people have taken the opportunity to scam people through them. 
SafeComm's purpose is exactly that of minimizing these crooked messages or emails that we get. Starting from there we were asked to test three machine learning models that were able to carry out this type of work efficiently. 


[Section 2]
In our exploration of the SafeComm project, aimed at combatting SMS-based fraud, we delved into the evaluation of three machine learning models: Logistic Regression, Random Forests, and SVM (Support Vector Machine). The evaluation metrics of focus were precision, recall, accuracy, and F1 score.We'll now go through each one of them and our reasonings behid each choice:
Our initial preference was Logistic Regression primarily because it is a supervised machine learning model. We believed that employing supervised training models would be the most suitable approach for this task, as providing them with examples of fraudulent and non-fraudulent messages could establish a solid baseline for the project. The process involves providing the model with a training set containing specific features known as predictors. These predictors help the model identify and assign a fraudulence score on a scale from 0 to 1, with 1 indicating fraudulent and 0 indicating non-fraudulent. Consequently, upon message delivery, the model promptly classifies spam messages and directs them to the bin, preventing the recipient from reading them.

![alt text](https://github.com/Oscarformaro/SafeCommDigitalSecurity---Riccardo-Aversa-283831/blob/f968c9774f55b726a8872d286cf0274add229a6b/Images/Fraudulent%20amount.png)

As our second choice, we opted for Random Forests, an additional supervised machine learning model. The operational mechanism of this model involves constructing a substantial number of Classification and Regression Trees (CART), which collectively cast votes on the outcome of a new observation, ultimately selecting the outcome that gathers the majority of votes. Random Forests display user-friendliness, characterized by ease of use, and they boast a reduced number of hyperparameters in comparison to more intricate models. Furthermore, we found Random Forests to be a compelling option due to their remarkable versatility in handling various data types.

Our final selection was the Support Vector Machine (SVM). The core aim of this model is to identify a hyperplane that optimally divides the data into distinct classes. SVM excels particularly in high-dimensional spaces and proves effective when dealing with non-linearly separable data. It achieves this by employing the kernel trick, a technique that transforms input features into a higher-dimensional space, enabling SVM to effectively manage intricate relationships within the data. Notably, SVM's versatility extends to its capability to perform well with relatively small and imbalanced datasets, contributing to its widespread popularity among various AI models.

To carry out this job we made use of some libraries such as pandas (used specifically for data manipulation and analysis) and sklearn; in particular we used train_test_split (to split the dataset into subsets that minimize the potential for bias in the evaluation and validation process), TfidfVectorizer (to convert raw documents to a matrix of TF-IDF features), SVC (to fit to the data provided, returning a "best fit" hyperplane that divides, or categorizes the data), classification_report (to compute the accuracy of a classification model based on the values from the confusion matrix), accuracy_score (to calculate the accuracy score for a set of predicted labels against the true labels), confusion_matrix (to evaluate the performance of a machine learning algorithm), and finally Pipeline (to link all steps of data manipulation together to create a pipeline). 
Python 3.11.3 served as our programming environment.


[Section 3]
In constructing our initial model, Logistic Regression, our initial approach involved categorizing messages as fraudulent or non-fraudulent based solely on the words present in those messages. Initially, we employed the TF-IDF vectorizer for this purpose and saw that there was correlation between some specific words used and the oucome of the message being fraudolent - (examples include "WINNER" "PRIZE" and "URGENT"). Subsequently, we introduced an additional parameter, namely the length of the messages. Notably, incorporating this new parameter yielded improved results, indicating that longer messages are more likely to be fraudulent. Additionally, we introduced a third parameter involving the days of the week to explore potential correlations between specific days and the likelihood of messages being fraudulent. However, the results indicated an absence of any significant correlation in that regard. The results we obtained with this first model are:
Accuracy: 0.9871
Precision: 0.9767
Recall: 0.9231
F1 Score: 0.9492

![alt text](https://github.com/Oscarformaro/SafeCommDigitalSecurity---Riccardo-Aversa-283831/blob/f968c9774f55b726a8872d286cf0274add229a6b/Images/Message%20legth.png)


![alt text](https://github.com/Oscarformaro/SafeCommDigitalSecurity---Riccardo-Aversa-283831/blob/f968c9774f55b726a8872d286cf0274add229a6b/Images/Day-Farudulent.png)

Based on the outcomes obtained with our initial model, when transitioning to the second model, we decided to keep just the two parameters that provided significant insights: the presence of specific words and the length of sentences. While the Random Forests model yielded overall improved results compared to our initial model, it displayed a higher count of false negatives than false positives. Additionally, the recall obtained with Random Forests was not as robust as the one we achieved with Logistic Regression. Here are the results obtained with our second model:
Accuracy: 0.9864
Precision: 0.9880
Recall: 0.9066
F1 Score: 0.9456

Our third and final model, while maintaining consistency with the two parameters employed in the previous iterations—specifically, the presence of specific words and the length of sentences—manifested a notable reduction in errors. The total count of errors diminished to a mere 23 instances, consisting of 21 false negatives and only 2 false positives. This substantial enhancement highlights the model's improved accuracy and precision, showcasing its efficacy in mitigating both types of classification errors. Although its precision was higher compated to our first model, its recall did not prove to be better than the one we obtained with Logistic Regression. The results:
Accuracy: 0.9849
Precision: 0.9820
Recall: 0.9011
F1 Score: 0.9398


[Section 4]
In conclusion, our rigorous exploration of three machine learning models for fraud detection unequivocally designates Logistic Regression as the standout performer. While Random Forests and SVM demonstrated commendable performances, achieving accuracies of 98.64% and 98.49% respectively, Logistic Regression outshone them with a more balanced and superior recall, which we thought was the most important parameter to take into consideration for this type of task, emphasizing its proficiency in accurately identifying fraudulent instances.
This evaluation underscores the strategic significance of selecting Logistic Regression in refining fraud detection models, achieving an optimal balance between precision and recall for enhanced performance.


[Section 5] 
Takeaway Point:
In summary, our investigation into fraud detection models underscored the central role of specific features, such as word presence and message length, in optimizing accuracy and precision. The iterative refinement process revealed that Logistic Regression, when augmented with these crucial parameters, emerged as the most successful model in minimizing classification errors. The significance of understanding nuanced relationships within the data became evident, leading to a robust framework for effective fraud detection.

Unanswered Questions and Future Directions:
There remain areas where questions persist and further exploration is warranted. One aspect not fully addressed is the dynamic nature of fraudulent activities, which may evolve over time. Future work could delve into developing models adaptable to emerging fraud patterns. Additionally, investigating the potential impact of contextual factors, such as user behavior and geographic variations, could contribute to a more comprehensive understanding of fraud detection dynamics. Exploring the integration of advanced deep learning architectures and exploring ensemble methods could represent promising avenues for enhancing model robustness and adaptability. As technology and cyber threats evolve, ongoing research and refinement will be essential to stay ahead in the ever-changing landscape of fraud detection.