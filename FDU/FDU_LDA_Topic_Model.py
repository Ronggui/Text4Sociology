# -*- coding: utf-8 -*-
"""
Copyright (C) 2018 Ronggui Huang <rghuang@fudan.edu.cn>

Huang Ronggui
Department of Sociology
Fudan  University

“2018年上海社会科学研究方法前沿”研究生暑期学校

“社会科学研究中的大数据分析方法”
授课老师：黄荣贵（复旦大学社会学系）

Note: gensim 3.2.0 does not work with numpy 1.13.3 from anaconda
      need to install numpy using > pip install numpy
"""

import jieba
import os
import re
from zhon import hanzi
from gensim import corpora
from gensim import models
import itertools
import matplotlib.pylab as plt
import numpy as np
from sqlitedict import SqliteDict
import logging
from collections import defaultdict
import csv

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


os.chdir('/培训讲义/复旦大学暑期学校')


def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


def word_tokenizer(doc, stopwords=None):
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    # remove single character
    tokens = [el for el in tokens if el not in hanzi.punctuation]
    # remove Chinese punctuation
    pat_num = re.compile("[0-9a-zA-Z]+")
    tokens = [el for el in tokens if pat_num.sub('', el) != '']
    # remove all number and letters
    if stopwords is not None:
        tokens = [w for w in tokens if w not in stopwords]
    return tokens


def get_perplexity(model, corpus):
    p = np.exp2(- model.log_perplexity(corpus))
    return p


stp = get_words("停用词.txt")
file_path = "文本.txt"


# read the file
file = open(file_path, errors="replace", encoding="utf8")
texts = file.readlines()


# texts in the form of tokens
texts_as_tokens = [word_tokenizer(text, stp) for text in texts]


# dictionary
dictionary = corpora.Dictionary()
dictionary.add_documents(texts_as_tokens)
dictionary.filter_extremes(no_below=2, no_above=0.9)
# filtered_tokens = ['本文', '研究', '分析']
# dictionary.filter_tokens(bad_ids=[dictionary.token2id[k] for k in filtered_tokens])
dictionary.save("FDU_dictionary")
# dictionary = corpora.Dictionary.load("FDU_dictionary")


# corpus
mycorpus = [dictionary.doc2bow(text) for text in texts_as_tokens]


"""
# if the file is very large
file4d = open(file_path, errors="replace", encoding="utf8")
texts_as_tokens = (word_tokenizer(text, stp) for text in file4d)
dictionary = corpora.Dictionary()
dictionary.add_documents(texts_as_tokens)
dictionary.filter_extremes(no_below=2, no_above=0.9)
file4d.close()


file4c = open(file_path, errors="replace", encoding="utf8")
mycorpus = (dictionary.doc2bow(word_tokenizer(text, stp)) for text in file4c)
# serialized corpus
corpora.MmCorpus.serialize("FDU2018.mm", mycorpus)
file4c.close()
# load the serialized corpus
mycorpus = corpora.MmCorpus("FDU2018.mm")
print(mycorpus)
"""


# LDA topic model
model = models.LdaModel(corpus=mycorpus, num_topics=3, id2word=dictionary)
model.print_topics()


# model selection
model_dict = SqliteDict("FDU_ladmodels.sqlite", autocommit=True)

num_topics = [1, 2, 3, 4, 5, 6, 7]

for k in num_topics:
    model = models.LdaModel(corpus=mycorpus, num_topics=k, id2word=dictionary, passes=50, iterations=100)
    # model evaluation
    cm = models.CoherenceModel(model=model, corpus=mycorpus, dictionary=dictionary, coherence="u_mass")
    umass = cm.get_coherence()
    perplexity = get_perplexity(model, mycorpus)
    model_dict[k] = (model, umass, perplexity)
    print(k, " topic model is finished")


# visualization of coherence/perplexity scores
umass = [model_dict[e][1] for e in num_topics]
perplexity = [model_dict[e][2] for e in num_topics]

plt.plot(num_topics, umass, "ko-")
plt.xlabel("number of topics")
plt.ylabel("umass coherence")
plt.savefig("FDU_umass.pdf")
plt.close()

plt.plot(num_topics, perplexity, "ko-")
plt.xlabel("number of topics")
plt.ylabel("perplexity")
plt.savefig("FDU_perplexity.pdf")
plt.close()


# choose the final model
mod_final = model_dict[2][0]

# key words of topics
mod_final.show_topics()


# topic probabilities of docs
doc = """我国社会结构的剧烈变迁加深了社区人口的异质性,这种结构性的变化将对城市邻里的社会资本产生重大影响。关于社区内部异质性对邻里社会资本的作用,国内外学者看法不一。在剖析争论的基础上,本文提出了一种分类研究框架,认为社区内部异质性对两种性质的社会资本具有不同的影响。社区内部异质性的增大会抑制整合性的社会资本,但有可能促进链合性的社会资本。"""
doc_tokens = word_tokenizer(doc, stp)
doc_corpus = dictionary.doc2bow(doc_tokens)

doc_topics = mod_final[doc_corpus]

[defaultdict(int, doc_topics).get(_) for _ in range(2)]


# get topic prob for each document, for further processing
def get_doc_topics(model, corpus, outfile):
    num_topic = model.num_topics
    out = open(outfile, "w")
    writer = csv.writer(out)
    for doc_topics in mod_final[corpus]:
        topics = [defaultdict(int, doc_topics).get(_) for _ in range(num_topic)]
        writer.writerow(topics)
    out.close()


get_doc_topics(mod_final, mycorpus, "topics.csv")
