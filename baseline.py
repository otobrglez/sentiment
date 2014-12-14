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
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.linear_model import LinearRegression as lin_reg
from sklearn.feature_extraction.text import TfidfVectorizer
import re


def clean(text):
    return html.fromstring(text).text_content().lower().strip()

data1 = pd.read_csv('data/opinions.csv', delimiter=',')
data2 = pd.read_csv('data/lemmatized.csv', delimiter=',')
data = data1[['komentar', 'ocena']]

words = [str(comment) for comment in data['komentar'].values]
sentiments = data['ocena'].values
lemm = [str(comment) for comment in data2['komentar'].values]

def read_lexicon(file):
    return open(file, 'r').read().splitlines()

negative_words = read_lexicon("lexicon/slovene/negative-words.txt")
positive_words = read_lexicon("lexicon/slovene/positive-words.txt")

def word_score(word):
    """scores the given word with +1 if the word is positive, -1 if it is
       negative and 0 otherwise. Looks at prefix instead of just whole string
       so that it can spot words that should be in lexicon but are not there."""
    if len(word) < 5:
        if word in positive_words: return 1
        if word in negative_words: return -1
        return 0
    min_stem_len, max_postfix_len = 4, 4
    interesting_positive = [w for w in positive_words if w.startswith(word[0:4])]
    interesting_negative = [w for w in negative_words if w.startswith(word[0:4])]
    for p in range(len(word), max(len(word) - max_postfix_len, min_stem_len), -1):
        prefix = word[0:p]
        pos = any(dict_word.startswith(prefix) for dict_word in interesting_positive)
        neg = any(dict_word.startswith(prefix) for dict_word in interesting_negative)
        if pos and neg: return 0
        if pos: return 1
        if neg: return -1
    return 0

def sentence_score(sentence):
    "calculates score of a given sentence"
    return np.sign(sum([word_score(word) for word in sentence.split()]))

def opinion_score(opinion):
    "calculates total score of a certain opinion"
    sentences = re.split(r' \.|\!|\? ', opinion)
    return sum([sentence_score(sentence) for sentence in sentences])

opinion_scores = [opinion_score(opinion) for opinion in lemm]

tfidf = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
#content = tfidf.fit_transform(words)
content = tfidf.fit_transform(lemm)

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

plot_roc = False
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

# Save the lemmatized to a CSV
#pd.DataFrame(np.asarray([data1['profesor'], data['ocena'],
#                         np.asarray([u.encode('utf-8') for u in lemm])]).T).to_csv('data/lemmatized.csv',
#                                                                      index=False, header=["profesor", "ocena", "komentar"])

# TO DO: try a SVM, tune params, try lasso or ridge for logReg? deep learning?
