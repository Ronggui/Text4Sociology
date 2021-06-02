"""
第X章 网络舆论表达中的文本挖掘与主题模型，2021
黄荣贵（复旦大学社会学系）
"""

# 加载所需要的模块
import os
import re

import jieba
import matplotlib.pylab as plt
from gensim import corpora
from gensim import models
from tmtoolkit.topicmod import tm_gensim
from tmtoolkit.topicmod.evaluate import results_by_parameter
from tmtoolkit.topicmod.visualize import plot_eval_results

import numpy as np
import pandas as pd
import csv
from collections import defaultdict

import statsmodels.formula.api as smf


# 定义函数，用于读入停用词列表
def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


# 定义函数对文本进行分词和停用词过滤操作，可根据研究的需要进行相应的修改
def word_tokenizer(doc, stopwords=None):
    tokens = jieba.cut(doc)
    tokens = [el for el in tokens if len(el) > 1]
    # remove single character
    pat_num = re.compile("[0-9a-zA-Z]+")
    tokens = [el for el in tokens if pat_num.sub('', el) != '']
    # remove all number and letters
    if stopwords is not None:
        tokens = [w for w in tokens if w not in stopwords]
    return tokens


# 读入停用词，在“停用词.txt”文件，一个停用词对应一行
# 已包括通用停用词、平台相关停用词、以及地名等实体命名词
stp = get_words("停用词.txt")


# 读入文本数据，数据格式为csv
# 数据含有微博文（webo_text）、博文日期、博主昵称变量。
data = pd.read_csv("Corpus/劳工关注社群.csv") 
texts = data.weibo_text.tolist()  # 从数据数据集中抽取微博文数据，以列表的形式来表示
texts_as_tokens = [word_tokenizer(text,  stp) for text in texts] 
# 分词与停用词过滤，分词后每条博文将表示为词汇的列表


dictionary = corpora.Dictionary()
dictionary.add_documents(texts_as_tokens)
# dictionary.filter_extremes(no_below=2, no_above=0.9)  # 过滤非常稀有和常见的词汇
dictionary.save("dictionary")  # 将创建的字典对象保存到电脑硬盘中的文件中去
dictionary = corpora.Dictionary.load("dictionary")  #重新加载所保存的字典对象


# gensim corpus
mycorpus = [dictionary.doc2bow(text) for text in texts_as_tokens]


# 使用tm_gensim模块进行主题模型比较
ks = list(range(2, 41, 1))   # 话题数列表
varying_params = [dict(num_topics=k) for k in ks]


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


mymodel = models.LdaModel(corpus=mycorpus, num_topics=13, id2word=dictionary)
mymodel.print_topics()


# 估算每个文档的主题概率，用于后续统计/可视化分析
def get_doc_topics(model, corpus, outfile):
    num_topic = model.num_topics
    out = open(outfile, "w")
    writer = csv.writer(out)
    for doc_topics in model[corpus]:
        topics = [defaultdict(int, doc_topics).get(_) for _ in range(num_topic)]
        writer.writerow(topics)
    out.close()


get_doc_topics(mymodel, mycorpus, "doc_topics_prob.csv") 
#主题概率数据将保存在"doc_topics_prob.csv"文件中


df_topics = pd.read_csv("doc_topics_prob.csv ", header=None)
df_combined = pd.concat([data, df_topics], axis=1)  # 合并两个数据集
df_combined.fillna(0, inplace=True)  # 将缺失值的文档概率替换为0


df_combined.rename(columns={ 0:'城市融入', 1:'农民工问题'}, inplace=True)

community= pd.read_csv("communities.csv ", header=True)
df_combined = pd.merge(df_combined, community, on="doc_ID ")  # 匹配并合并


S1 = df_combined.loc[(df_combined.community_1 ==1),  ['城市融入', '农民工问题']].sum()
S1.plot(kind='bar')
plt.show()


# S1是pandas.DataFrame数据集对象
# 其中'城市融入'、'社群3'、'社群1与3'是数据集中的三个变量
mod = smf.ols('城市融入~ 社群3 + 社群1与3', data=S1).fit()
print(mod.summary())
#                            OLS Regression Results
# ==============================================================================
# Dep. Variable:                   城市融入   R-squared:                       0.020
# Model:                            OLS   Adj. R-squared:                  0.020
# Method:                 Least Squares   F-statistic:                     168.2
# Date:                Tue, 01 Jun 2021   Prob (F-statistic):           5.01e-73
# Time:                        16:26:44   Log-Likelihood:                 4840.9
# No. Observations:               16303   AIC:                            -9676.
# Df Residuals:                   16300   BIC:                            -9653.
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept      0.1402      0.002     78.136      0.000       0.137       0.144
# 社群3           -0.0488      0.004    -12.915      0.000      -0.056      -0.041
# 社群1与3         -0.0565      0.004    -15.713      0.000      -0.064      -0.049
# ==============================================================================
# Omnibus:                     5607.496   Durbin-Watson:                   1.743
# Prob(Omnibus):                  0.000   Jarque-Bera (JB):            16590.320
# Skew:                           1.830   Prob(JB):                         0.00
# Kurtosis:                       6.321   Cond. No.                         3.16
# ==============================================================================

print(mod.wald_test("社群3 =社群1与3"))
# <F test: F=array([[2.85453924]]), p=0.09113546641949251, df_denom=1.63e+04, df_num=1>

