__author__ = 'Aleksandar Dimitriev'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from lxml import html
from sklearn import cross_validation
from sklearn import metrics
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as log_reg
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.feature_extraction.text import TfidfVectorizer


def clean(text):
    return html.fromstring(text).text_content().lower().strip()

plot_roc = False

with open("data/endrecord.txt", "r") as my_file:
    lemm = my_file.read()
sentences = lemm.split('endrecord\tendrecord')
sentences = [clean(x) for x in sentences][1:]

lemm = []
for x in sentences:
    l = [y.split('\t')[1] for y in x.split('\n') if len(y.split('\t')) == 3]
    lemm.append(" ".join(l[1:]))
#lemmatized = pd.DataFrame(np.asarray(lemm).T)



# no lemmatization (for now)
data1 = pd.read_csv('data/opinions.csv', delimiter=',')
data = data1[['komentar', 'ocena']]

words = [str(comment) for comment in data['komentar'].values]
sentiments = data['ocena'].values



tfidf = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
#content = tfidf.fit_transform(words)
content = tfidf.fit_transform(lemm)

# Predict positive (5) vs. negative (all else: 1, 2, 3, 4)
print "Because the dataset is very imbalanced we merge all the classes except 5 to get a ~50/50 split."
print Counter(sentiments)
only_5 = [1 if x == 5 else 0 for x in sentiments]
print Counter(only_5)

print "For the moment, we achieve the following accuracy:"
logistic = log_reg()
logistic.fit(content, only_5)
predictions = logistic.predict_proba(content)

binary_preds = logistic.predict(content)
acc = sum([1 for i in range(len(only_5)) if binary_preds[i] == only_5[i]])*1.0/len(only_5)
print "Accuracy on training set: %.3f" % acc

auc = roc_auc_score(only_5, [x[1] for x in predictions])
fpr, tpr, thresholds = metrics.roc_curve(only_5, [x[1] for x in predictions])

logistic2 = log_reg()
scores = cross_validation.cross_val_score(logistic2, content, np.asarray(only_5), cv=10)
print "Accuracy on test set (i.e. using 10-fold CV): %.3f " % np.mean(scores)

if plot_roc:
    # Plot a ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve and AUC for a 5 vs. rest logistic regression')
    plt.legend(loc="lower right")
    plt.show()

# TO DO: try a SVM, tune params, try lasso or ridge for logReg? deep learning?

# Save the lemmatized to a CSV
pd.DataFrame(np.asarray([data1['profesor'], data['ocena'],
                         np.asarray([u.encode('utf-8') for u in lemm])]).T).to_csv('data/lemmatized.csv',
                                                                      index=False, header=["profesor", "ocena", "komentar"])