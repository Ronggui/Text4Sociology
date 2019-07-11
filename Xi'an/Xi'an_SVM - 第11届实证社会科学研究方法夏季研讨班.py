# -*- coding: utf-8 -*-
"""
Created on 12 July 2019
Copyright (C) 2019 Ronggui Huang <rghuang@fudan.edu.cn>

Huang Ronggui
Department of Sociology
Fudan  University

第11届实证社会科学研究方法夏季研讨班
授课老师：黄荣贵（复旦大学社会学系）
"""

import jieba
import numpy as np
import os
import pandas as pd
import sklearn.metrics as metrics
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_score

# from sklearn.model_selection import train_test_split


d = pd.read_csv("Weibo_XKD_500_Texts_SVM.csv")
d = d.dropna()

cvr = CountVectorizer(input="content",
                      strip_accents=None,
                      tokenizer=jieba.lcut)

X_tf = cvr.fit_transform(d.text)

tfidf = TfidfTransformer()
X_tfidf = tfidf.fit_transform(X_tf)

clf = svm.LinearSVC().fit(X_tfidf, d.y)

Y_pred = clf.predict(X_tfidf)
print(metrics.confusion_matrix(d.y, Y_pred))
print(metrics.classification_report(d.y, Y_pred))
print(metrics.accuracy_score(d.y, Y_pred))

val_scores = cross_val_score(svm.LinearSVC(),
                             X_tfidf, d.y,
                             cv=5
                             )

print(val_scores.mean())
