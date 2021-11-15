'''
https://github.com/zhanlaoban/EDA_NLP_for_Chinese
'''

import os
import jieba
import synonyms
import random
from random import shuffle


class EdaChinese(object):

    def __init__(self, num_aug):
        self.stop_words = list()
        with open(os.path.abspath(os.path.dirname(os.path.abspath(__file__)) + os.path.sep + ".") + '/'
                  + 'hit_stopwords.txt') as f:
            for stop_word in f:
                self.stop_words.append(stop_word[:-1])
        self.num_aug = num_aug

    def synonym_replacement(self, words, n):
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word not in self.stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break

        sentence = ' '.join(new_words)
        new_words = sentence.split(' ')

        return new_words

    def get_synonyms(self, word):
        return synonyms.nearby(word)[0]

    def random_insertion(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            self.add_word(new_words)
        return new_words

    def add_word(self, new_words):
        synonyms = []
        counter = 0
        while len(synonyms) < 1:
            random_word = new_words[random.randint(0, len(new_words) - 1)]
            synonyms = self.get_synonyms(random_word)
            counter += 1
            if counter >= 10:
                return
        random_synonym = random.choice(synonyms)
        random_idx = random.randint(0, len(new_words) - 1)
        new_words.insert(random_idx, random_synonym)

    def random_swap(self, words, n):
        new_words = words.copy()
        for _ in range(n):
            new_words = self.swap_word(new_words)
        return new_words

    def swap_word(self, new_words):
        random_idx_1 = random.randint(0, len(new_words) - 1)
        random_idx_2 = random_idx_1
        counter = 0
        while random_idx_2 == random_idx_1:
            random_idx_2 = random.randint(0, len(new_words) - 1)
            counter += 1
            if counter > 3:
                return new_words
        new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
        return new_words

    def random_deletion(self, words, p):
        if len(words) == 1:
            return words

        new_words = []
        for word in words:
            r = random.uniform(0, 1)
            if r > p:
                new_words.append(word)

        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return [words[rand_int]]

        return new_words

    def __call__(self, sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1):
        seg_list = jieba.cut(sentence)
        seg_list = " ".join(seg_list)
        words = list(seg_list.split())
        num_words = len(words)

        augmented_sentences = []
        num_new_per_technique = int(self.num_aug / 4) + 1
        n_sr = max(1, int(alpha_sr * num_words))
        n_ri = max(1, int(alpha_ri * num_words))
        n_rs = max(1, int(alpha_rs * num_words))

        for _ in range(num_new_per_technique):
            a_words = self.synonym_replacement(words, n_sr)
            augmented_sentences.append(' '.join(a_words).replace(' ', ''))

        for _ in range(num_new_per_technique):
            a_words = self.random_insertion(words, n_ri)
            augmented_sentences.append(' '.join(a_words).replace(' ', ''))

        for _ in range(num_new_per_technique):
            a_words = self.random_swap(words, n_rs)
            augmented_sentences.append(' '.join(a_words).replace(' ', ''))

        for _ in range(num_new_per_technique):
            a_words = self.random_deletion(words, p_rd)
            augmented_sentences.append(' '.join(a_words).replace(' ', ''))

        shuffle(augmented_sentences)

        if self.num_aug >= 1:
            augmented_sentences = augmented_sentences[:self.num_aug]
        else:
            keep_prob = self.num_aug / len(augmented_sentences)
            augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

        augmented_sentences.append(seg_list)

        return augmented_sentences


if __name__ == '__main__':
    line = '您好请讲。有什么可以帮到您呢？那您稍等一下这边还得给您转咱们这边网络专线给您查一下保额是两个人是啊。'
    eda = EdaChinese(1)
    sentence = eda(line)
    print(sentence[0], sentence[1], line)