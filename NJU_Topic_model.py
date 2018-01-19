# -*- coding: utf-8 -*-
"""
@author:
Huang Ronggui
Department of Sociology
Fudan  University
ronggui.huang@gmail.com
"""


import csv
import jieba
import os
import re
import sys
import zhon.hanzi as hanzi
from gensim import corpora
from gensim import models
import itertools
import matplotlib.pylab as plt
import numpy as np
from sqlitedict import SqliteDict


os.chdir('南大寒假课程/实例')
csv.field_size_limit(sys.maxsize)


def word_tokenizer(doc):
    pat_uid = re.compile(u"@[a-zA-Z0-9一-龥-_]{2,30}")
    doc = pat_uid.sub('', doc)
    # remove user name before tokenizing
    doc = re.compile("(http:)[a-zA-Z0-9.\\\\]*").sub('', doc)
    # remove short urls
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    # remove single character
    tokens = [el for el in tokens if el not in hanzi.punctuation]
    # remove Chinese punctuation
    pat_num = re.compile("[0-9a-zA-Z]+")
    return [el for el in tokens if pat_num.sub('', el) != '']
    # remove all number and letters


def get_perplexity(model, corpus):
    p = np.exp2(- model.log_perplexity(corpus))
    return p


file_path = "南大课程2018.csv"
file = open(file_path, errors="replace", encoding="utf8")
reader = csv.reader(file)
reader4d, reader4c = itertools.tee(reader, 2)  # 2 independent iterators


# dictionary
corpus4dict = (word_tokenizer(line[0]) for line in reader4d)
dictionary = corpora.Dictionary()
dictionary.add_documents(corpus4dict)
dictionary.filter_extremes(no_below=20, no_above=0.5)
filtered_tokens = ['微博', '转发', '回复']
dictionary.filter_tokens(bad_ids=[dictionary.token2id[k] for k in filtered_tokens])
dictionary.compactify()
print(dictionary)
dictionary.save("NJU_dictionary")
# dictionary = corpora.Dictionary.load("NJU_dictionary")


# serialized corpus
corpus = (dictionary.doc2bow(word_tokenizer(line[0])) for line in reader4c
          if dictionary.doc2bow(word_tokenizer(line[0])))
corpora.MmCorpus.serialize("南大课程2018.mm", corpus)
# load the serialized corpus
mm_corpus = corpora.MmCorpus("南大课程2018.mm")
print(mm_corpus)


# LDA topic model
model_dict = SqliteDict("ladmodels.sqlite", autocommit=True)
for k in range(2, 16):
    model = models.LdaModel(corpus=mm_corpus, num_topics=k, id2word=dictionary)
    # model evaluation
    cm = models.CoherenceModel(model=model, corpus=mm_corpus, dictionary=dictionary, coherence="u_mass")
    umass = cm.get_coherence()
    model_dict[k] = (model, umass)
    print(k, "is finished", "u-mass is", umass)


# visualization of coherence scores
plt.plot(range(2, 16), [model_dict[e][1] for e in range(2, 16)], "ko-")


perplexity = []
for k in range(2, 16):
    print("calculating perplexity for model with %s topics" % k)
    model = model_dict[k][0]
    p = get_perplexity(model, mm_corpus)
    perplexity.append(p)


plt.plot(range(2, 16), perplexity, "ko-")


# cohrence of each topic
m5 = model_dict[5][0]
m5t = m5.top_topics(mm_corpus)
[t[1] for t in m5t]
m7 = model_dict[7][0]
m7t = m7.top_topics(mm_corpus)
[t[1] for t in m7t]

# choose the final model
model = model_dict[5][0]

# key words of topics
model.show_topics()

# topic probabilities of docs
doc0 = mm_corpus[0]
model[doc0]
