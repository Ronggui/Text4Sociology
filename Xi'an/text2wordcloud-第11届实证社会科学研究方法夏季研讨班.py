# -*- coding: utf-8 -*-
"""
Copyright (C) 2019 Ronggui Huang <rghuang@fudan.edu.cn>

Huang Ronggui
Department of Sociology
Fudan  University

第11届实证社会科学研究方法夏季研讨班
授课老师：黄荣贵（复旦大学社会学系）
"""

from wordcloud import WordCloud
import collections
import matplotlib.pylab as plt
import jieba
import argparse


def get_words(file):
    with open(file, encoding="utf-8") as f:
        stopwords = f.readlines()
        stopwords = [w.strip() for w in stopwords]
    return stopwords


def text2wordcloud(text_path, word_cloud_path, stopwords=None, new_words=None, font_path=None):
    if new_words is not None:
        for w in new_words:
            jieba.add_word(w)
    freq = collections.Counter()
    texts = open(text_path, encoding="utf-8", errors="replace")
    for text in texts:
        tokens = jieba.lcut(text)
        tokens = [w for w in tokens if len(w) > 1]
        if stopwords is not None:
            tokens = [w for w in tokens if w not in stopwords]
        for w in tokens:
            freq[w] += 1

    cloud = WordCloud(font_path=font_path,
                    width=1600, height=800, background_color='white')
    # /Library/Fonts/Arial Unicode.ttf
    cloud.generate_from_frequencies(freq)


    plt.figure(figsize=(20, 10))
    plt.imshow(cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig("%s.pdf" % word_cloud_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text_path", help="path of a text file")
    parser.add_argument("word_cloud_path", help="path of the word cloud file")
    parser.add_argument("-stopwords", help="path of the stop words file")
    parser.add_argument("-new_words", help="path of the new words file")
    parser.add_argument("-font_path", help="path of a ttf font")
    arg = parser.parse_args()

    stopwords = None
    newwords = None

    if arg.stopwords is not None:
        stopwords = get_words(arg.stopwords)
    if arg.new_words is not None:
        newwords = get_words(arg.new_words)
    font_path = arg.font_path
    text2wordcloud(arg.text_path, arg.word_cloud_path, stopwords, newwords, font_path)
