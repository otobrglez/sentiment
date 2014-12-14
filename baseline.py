__author__ = 'Aleksandar Dimitriev'

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from lxml import html
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.feature_extraction.text import TfidfVectorizer


def clean(text):
    """ Return a parsed string without html and extra spaces, and with only lowercase letters. """
    return html.fromstring(text).text_content().lower().strip()


def negate_comments(text_list):
    """ Input: a list of comments/reviews
        Output: same list with 'NOT_' prepended to each word after 'negation words' (see below)
                until a break point (.,-?!;)
        Negation words: ne, ni
    """
    output = []
    for sentence in text_list:
        regex = re.findall(r" (?:ne|ni) .*?(?=[.,\-!?;])", sentence)
        for x in regex:
            negated = " ".join(['NOT_' + y for y in x.split(' ')[2:-1]])
            sentence = sentence.replace(x, ' ' + negated + ' ')
        output.append(sentence)
    return output


def plot_roc(fpr, tpr, auc):
    plt.figure()
    plt.plot(fpr, tpr, label=str(auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve and AUC')
    plt.legend(loc="lower right")
    plt.show()

# MAIN
lemmatize = True

data1 = pd.read_csv('data/opinions.csv', delimiter=',')
data2 = pd.read_csv('data/lemmatized.csv', delimiter=',')
data = data1[['komentar', 'ocena']]

words = [str(comment) for comment in data['komentar'].values]
sentiments = data['ocena'].values
lemm = [str(comment) for comment in data2['komentar'].values]

tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
if lemmatize:
    content = tfidf.fit_transform(lemm)
else:
    content = tfidf.fit_transform(words)

# Predict positive (5) vs. negative (all else: 1, 2, 3, 4)
print "Because the dataset is very imbalanced we merge all the classes except 5 to get a ~50/50 split."
print Counter(sentiments)
only_5 = [1 if x == 5 else 0 for x in sentiments]
print Counter(only_5)

print "For the moment, we achieve the following accuracy:"
logistic = logReg()
logistic.fit(content, only_5)
predictions = logistic.predict_proba(content)

binary_preds = logistic.predict(content)
acc = sum([1 for i in range(len(only_5)) if binary_preds[i] == only_5[i]])*1.0/len(only_5)
print "Accuracy on training set: %.3f" % acc

auc = roc_auc_score(only_5, [x[1] for x in predictions])
fpr, tpr, thresholds = metrics.roc_curve(only_5, [x[1] for x in predictions])

logistic2 = logReg()
scores = cross_validation.cross_val_score(logistic2, content, np.asarray(only_5), cv=10)
print "Accuracy on test set (i.e. using 10-fold CV): %.3f " % np.mean(scores)
#plot_roc(fpr, tpr, auc)







# TO DO: try a SVM, tune params, try lasso or ridge for logReg? deep learning?



# UNUSED:

# Save the lemmatized to a CSV
#pd.DataFrame(np.asarray([data1['profesor'], data['ocena'],
#                         np.asarray([u.encode('utf-8') for u in lemm])]).T).to_csv('data/lemmatized.csv',
#                                                   index=False, header=["profesor", "ocena", "komentar"])

# Save the negated to a CSV
#negated = negate_comments(lemm)
#pd.DataFrame(np.asarray([data1['profesor'], data['ocena'],
#                         np.asarray(negated)]).T).to_csv('data/negated.csv',
#                                                   index=False, header=["profesor", "ocena", "komentar"])

def print_negations(txt_list):
    """ Prints negations and the number of them """
    counter = 0
    for com in txt_list:
        regex = re.findall(r"(?:ne|ni) \w+", com)
        if len(regex) > 0:
            counter += 1
            print regex
    print "number of negations:", counter

#print_negations(lemm)
