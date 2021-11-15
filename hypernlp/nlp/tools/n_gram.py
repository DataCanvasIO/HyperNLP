import os
import codecs

import jieba
import hypernlp.nlp.dataset
from hypernlp.config import *


class NGram(object):

    def __init__(self, dataset):
        self.data = dataset
        self.model = {}

    def cut(self, line):
        segs = jieba.cut_for_search(line)
        seq = " ".join(segs)
        print(seq)


if __name__ == '__main__':
    ngram = NGram(None)
    ngram.cut('我爱北京天安门，天安门上太阳升！')

