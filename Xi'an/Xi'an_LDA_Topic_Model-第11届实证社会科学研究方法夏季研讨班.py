# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Ronggui Huang <rghuang@fudan.edu.cn>
10-11 July 2019

Huang Ronggui
Department of Sociology
Fudan  University

第11届实证社会科学研究方法夏季研讨班
授课老师：黄荣贵（复旦大学社会学系）
"""

import jieba
import os
import re
# from zhon import hanzi
from gensim import corpora
from gensim import models
import itertools
import matplotlib.pylab as plt
import numpy as np
# from sqlitedict import SqliteDict
import logging
from collections import defaultdict
import csv
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results
import matplotlib.pylab as plt

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


# os.chdir(r'实例所在文件夹路径')


def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


def word_tokenizer(doc, stopwords=None):
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    # remove single character
    if stopwords is not None:
        tokens = [w for w in tokens if w not in stopwords]
    return tokens


stp = get_words("停用词.txt")
file_path = "文本.txt"


# read the file
file = open(file_path, errors="replace", encoding="utf8")
texts = file.readlines()
texts = [e.strip() for e in texts]


# texts in the form of tokens
texts_as_tokens = [word_tokenizer(text, stp) for text in texts]


# dictionary
dictionary = corpora.Dictionary()
dictionary.add_documents(texts_as_tokens)
dictionary.filter_extremes(no_below=2, no_above=0.9)
# filtered_tokens = ['本文', '研究', '分析']
# dictionary.filter_tokens(bad_ids=[dictionary.token2id[k] for k in filtered_tokens])
dictionary.save("dictionary.bin")
# dictionary = corpora.Dictionary.load("dictionary.bin")


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
corpora.MmCorpus.serialize("Corpus2019.mm", mycorpus)
file4c.close()
# load the serialized corpus
mycorpus = corpora.MmCorpus("Corpus2019.mm")
print(mycorpus)
"""


# LDA topic model
model = models.LdaModel(corpus=mycorpus, num_topics=3, id2word=dictionary, passes=10)
model.print_topics()


# model comparison and selection
ks = list(range(1, 10, 1))  # 话题数
varying_params = [dict(num_topics=k) for k in ks]
print(varying_params)


eval_results = \
    tm_gensim.evaluate_topic_models(data=(dictionary, mycorpus),
                                    varying_parameters=varying_params,
                                    metric=('perplexity',
                                            'cao_juan_2009',
                                            'coherence_gensim_u_mass'
                                            )
                                    )


plt.style.use('ggplot')
results_by_n_topics = results_by_parameter(eval_results, 'num_topics')
plot_eval_results(results_by_n_topics,
                  xaxislabel='num. topics k',
                  title='Evaluation results',
                  figsize=(8, 6))


# choose the final model
mod_final = models.LdaModel(corpus=mycorpus, num_topics=2, id2word=dictionary, passes=10)
mod_final.print_topics()


# topic probabilities of docs
doc_corpus = mycorpus[1]
# or
doc = """我国社会结构的剧烈变迁加深了社区人口的异质性,这种结构性的变化将对城市邻里的社会资本产生重大影响。关于社区内部异质性对邻里社会资本的作用,国内外学者看法不一。在剖析争论的基础上,本文提出了一种分类研究框架,认为社区内部异质性对两种性质的社会资本具有不同的影响。社区内部异质性的增大会抑制整合性的社会资本,但有可能促进链合性的社会资本。"""
doc_tokens = word_tokenizer(doc, stp)
doc_corpus = dictionary.doc2bow(doc_tokens)

doc_topics = mod_final[doc_corpus]

doc_prob = [defaultdict(int, doc_topics).get(_) for _ in range(2)]


# get topic prob for each document, for further processing
def get_doc_topics(model, corpus, outfile):
    num_topic = model.num_topics
    out = open(outfile, "w")
    writer = csv.writer(out)
    for doc_topics in mod_final[corpus]:
        topics = [defaultdict(int, doc_topics).get(_) for _ in range(num_topic)]
        writer.writerow(topics)
    out.close()


get_doc_topics(mod_final, mycorpus, "doc_topics_prob.csv")
