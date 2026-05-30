# email-spam-detector
A binary text classifier based end to end ML workflow for spam detection using Naive Bayes and Logistic Regression
## Dataset
SMS Spam Collection dataset from https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset. Contains 5572 SMS messages labelled as spam or ham (genuine)
## Workflow
1. Data loading and cleaning: dropped unused columns, renamed vague columns
2. EDA: class distribution analysis (87% ham, 13% spam), message length comparison
3. Data pre-processing: lowercasing, punctuation removal, stopword removal
4. TF-IDF vectorisation of top 3000 features
5. Training set and testing set split (80:20 ratio)
6. Model training and evaluation: Naive Bayes (MultinomialNB) and Logistic Regression
## Results
| Metric | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Accuracy | 0.98 | 0.95 |
| Spam Precision | 1.00 | 0.96 |
| Spam Recall | 0.85 | 0.68 |
| Spam F1 Score | 0.92 | 0.80 |
From the above results, Naive Bayes produced better results than Logistic Regression
## Top words indicative of spam messages
call, free, txt, claim, mobile, stop, text, prize, ur, reply, new, urgent, cash, please, nokia, get, win, service, contact, send
## How to run
1. Load "EmailSpamDetector.ipynb" into Google Colab
2. Load "spam.csv" (download from the above mentioned link) into Google Colab's local storage
3. Run all cells
## Requirements
pandas, numpy, scikit-learn, nltk, matplotlib, seaborn
