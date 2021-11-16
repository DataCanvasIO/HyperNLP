import abc
import random
import sys

import numpy as np
import tensorflow as tf
import torch
from progressbar import ProgressBar, Percentage, Bar

from hypernlp.config import Config
from hypernlp.nlp.data_process.format_dataset import TXTDataset
from utils.gpu_status import is_gpu_available
from utils.logger import logger


class DatasetBase(object):

    def __init__(self, data, max_len, tokenizer, batch_size, shuffle):

        self.data = data  # dataframe
        self.max_len = max_len  # max length of sequence
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.shuffle = shuffle

    @abc.abstractmethod
    def _data_argument(self):
        pass

    @abc.abstractmethod
    def _create_data(self):
        pass

    def _squeeze(self, x):
        if Config.framework == "pytorch":
            return torch.squeeze(x)

        elif Config.framework == "tensorflow":
            return tf.squeeze(x)
        else:
            raise ValueError("Unsupported framework: {}!".format(Config.framework))

    @abc.abstractmethod
    def _reset(self):
        pass

    @abc.abstractmethod
    def get_batch_data(self):
        pass

    @abc.abstractmethod
    def get_full_data(self):
        pass

    @abc.abstractmethod
    def tokenize_data(self, d):
        pass

    def tf_distribute_data(self, inputs):
        assert Config.framework == "tensorflow"
        # if not isinstance(inputs[0], np.ndarray):
        if not isinstance(inputs[0], list):
            raise TypeError("inputs type must be numpy ndarray!")
        x = tf.data.Dataset.from_tensor_slices(inputs).batch(self.batch_size)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        x = x.with_options(options)
        _datas = Config.strategy.experimental_distribute_dataset(x)
        return _datas


class DatasetSeq(DatasetBase):

    def __init__(self, data, max_len, tokenizer, batch_size, n_sampling=False, shuffle=True, with_labels=True,
                 EDA=None):
        super(DatasetSeq, self).__init__(data, max_len, tokenizer, batch_size, shuffle)
        self.eda = EDA
        self.with_labels = with_labels
        if self.eda is not None and with_labels is True:
            self.data = self._data_argument()
        self._index = 0
        self.epoch_length = len(self.data) // self.batch_size if len(self.data) % self.batch_size == 0 \
            else len(self.data) // self.batch_size + 1
        self.tokenized_data = []
        self.epoch_end = False
        self.n_sampling = n_sampling
        self._reset()

    def _data_argument(self):
        _data = []
        n = 0
        logger.info("Processing data argument: {}".format(len(self.data)))
        for d in self.data:
            _sequences = []
            for l in d[:-1]:
                _sequences.append(self.eda(l))
            _data.append(d)
            for i in range(10):
                try:
                    _d = []
                    for s in range(len(_sequences)):
                        _d.append(_sequences[s][i])
                    _d.append(d[-1])
                    _data.append(_d)
                except Exception:
                    pass
            sys.stdout.write("Processing evaluation: {}/{}".format(n, len(self.data)) + '\r')
            sys.stdout.flush()
            n += 1
        return TXTDataset(_data, range(len(_data[0]))[:-1], label_index=len(_data[0]) - 1)

    '''
    Increase the n_sample sampling rate,
    Need fix the resampled conflict samples.
    '''

    def _create_random_n_samples(self):
        anchor = self.data[self._index][0]
        index = random.randint(0, len(self.data) - 1)
        if index == self._index:
            index = random.randint(0, len(self.data) - 1)
        res_sample = [anchor, self.data[index][0], '0']
        return res_sample

    def tokenize_data(self, d):
        encoded_pair = self.tokenizer(d)
        token_data = [self._squeeze(encoded_pair['input_ids']),
                      self._squeeze(encoded_pair['attention_mask']), self._squeeze(encoded_pair['token_type_ids'])]

        if self.with_labels:
            if len(d) >= 2:
                try:
                    token_data.append(int(d[-1]))
                except Exception:
                    raise ValueError("Please make sure label column is correct!")
            else:
                raise IndexError("No label data is founded, make sure the dataset correct!")
        return token_data

    def tokenize_full_data(self):
        progress = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#')]).start()
        logger.info("Processing dataset tokenization: {}".format(len(self.data)))
        for i in progress(range(len(self.data))):
            d = self.data[i]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            token_data = self.tokenize_data(d)
            self.tokenized_data.append(token_data)
            progress.update()

    def _reset(self):
        self.epoch_end = False
        self._index = 0
        if self.shuffle is True:
            self.data.shuffle()

    def _create_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, label_dataset = [], [], [], []
        for i in range(self.batch_size):
            if self.n_sampling is True:
                if random.random() > 0.5:
                    raw_data = self._create_random_n_samples()
                else:
                    raw_data = self.data[self._index]
            else:
                raw_data = self.data[self._index]
            token_data = self.tokenize_data(raw_data)
            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])
            if self.with_labels:
                if Config.framework == "tensorflow":
                    label_dataset.append([token_data[3]])
                elif Config.framework == "pytorch":
                    label_dataset.append(torch.tensor([token_data[3]]))
                else:
                    raise ValueError("Unsupported framework: {}".format(Config.framework))
            self._index += 1
            if self._index == len(self.data):
                self._index = 0
                self.epoch_end = True

        if Config.framework == "tensorflow":
            if self.with_labels:
                return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset, label_dataset))
            return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset))
        elif Config.framework == "pytorch":
            if self.with_labels:
                if is_gpu_available():
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size,
                                                                        self.max_len).cuda(), torch.cat(
                        label_dataset, 0).reshape(self.batch_size).cuda()
                else:
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                        label_dataset, 0).reshape(self.batch_size, self.max_len)
            if is_gpu_available():
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda()
            else:
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size)
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))

    def __len__(self):  # return sample count
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_end is True:
            self._reset()
            raise StopIteration()
        return self._create_data()

    def get_batch_data(self):  # get tokenized sample by index
        if self.epoch_end is True:
            self._reset()
        return self._create_data()

    def get_full_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, label_dataset = [], [], [], []
        for i in range(len(self.data)):
            token_data = self.tokenize_data(self.data[i])
            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])
            if self.with_labels:
                if Config.framework == "tensorflow":
                    label_dataset.append([token_data[3]])
                elif Config.framework == "pytorch":
                    label_dataset.append(torch.tensor([token_data[3]]))
                else:
                    raise ValueError("Unsupported framework: {}".format(Config.framework))

        if Config.framework == "tensorflow":
            if self.with_labels:
                return tf.convert_to_tensor(token_ids_dataset), tf.convert_to_tensor(
                    attn_masks_dataset), tf.convert_to_tensor(token_type_ids_dataset), tf.convert_to_tensor(
                    label_dataset)
            return tf.convert_to_tensor(token_ids_dataset), tf.convert_to_tensor(
                attn_masks_dataset), tf.convert_to_tensor(
                token_type_ids_dataset)
        elif Config.framework == "pytorch":
            data_size = len(self.data)
            if self.with_labels:
                if is_gpu_available():
                    return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len).cuda(), torch.cat(
                        attn_masks_dataset, 0).reshape(data_size, self.max_len).cuda(), \
                           torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len).cuda(), torch.cat(
                        label_dataset, 0).reshape(data_size).cuda()
                else:
                    return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len), torch.cat(
                        attn_masks_dataset, 0).reshape(data_size, self.max_len), \
                           torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len), torch.cat(
                        label_dataset, 0).reshape(data_size)
            if is_gpu_available():
                return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len).cuda(), torch.cat(
                    attn_masks_dataset, 0).reshape(data_size, self.max_len).cuda(), \
                       torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len).cuda()
            else:
                return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len), torch.cat(
                    attn_masks_dataset, 0).reshape(data_size, self.max_len), \
                       torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len)
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))


class DatasetSep(DatasetBase):

    def __init__(self, data, max_len, tokenizer, batch_size, batch_rate, shuffle=True, with_labels=True):
        super(DatasetSep, self).__init__(data, max_len, tokenizer, batch_size, shuffle)
        self.data_length = len(self.data)
        self.with_labels = with_labels  # data with labels or not
        self._index = {}
        self.tokenized_data = {}
        self.seperated_data = {}
        self.epoch_end = False
        self.batch_rate = batch_rate
        if not sum(self.batch_rate) == 1:
            raise ValueError("Please make sure batch rate summary is 1 vs {}!".format(sum(self.batch_rate)))
        self._seperate_data()
        self._reset()
        self.epoch_length = len(self.data) // self.batch_size if len(self.data) % self.batch_size == 0 \
            else len(self.data) // self.batch_size + 1
        self._iter = 0
        del self.data

    def _gen_random_cls(self):
        rand = random.uniform(0, 1)
        cls = 0
        for i in range(0, len(self.batch_rate)):
            res = sum(self.batch_rate[:i + 1])
            if rand <= res:
                return str(i)
        return str(cls)

    def tokenize_data(self, d):
        encoded_pair = self.tokenizer(d)
        token_data = [self._squeeze(encoded_pair['input_ids']),
                      self._squeeze(encoded_pair['attention_mask']), self._squeeze(encoded_pair['token_type_ids'])]

        if self.with_labels:
            if len(d) >= 2:
                try:
                    token_data.append(int(d[-1]))
                except Exception:
                    raise ValueError("Please make sure label column is correct!")
            else:
                raise IndexError("No label data is founded, make sure the dataset correct!")
        return token_data

    def tokenize_full_data(self):
        progress = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#')]).start()
        logger.info("Processing dataset tokenization: {}".format(len(self.data)))
        for i in progress(range(len(self.data))):
            d = self.data[i]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            token_data = self.tokenize_data(d)
            if d[-1] in self.tokenized_data.keys():
                self.tokenized_data[d[-1]].append(token_data)
            else:
                self.tokenized_data[d[-1]] = [token_data]
            progress.update()

    def _seperate_data(self):
        progress = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#')]).start()
        logger.info("Seperate dataset: {}".format(len(self.data)))
        for i in progress(range(len(self.data))):
            d = self.data[i]
            if d[-1] in self.seperated_data.keys():
                self.seperated_data[d[-1]].append(d)
            else:
                self.seperated_data[d[-1]] = [d]
                self._index[d[-1]] = 0
            progress.update()

    def _reset(self, cls=None):
        self.epoch_end = False
        if self.shuffle is True:
            print(self.seperated_data["0"][0], self.seperated_data["1"][0])
            if cls is None:
                for d in self.seperated_data.keys():
                    random.shuffle(self.seperated_data[d])
                for d in self.seperated_data.keys():
                    self._index[d] = 0
            else:
                random.shuffle(self.seperated_data[cls])
                self._index[cls] = 0
            print(self.seperated_data["0"][0], self.seperated_data["1"][0])

    def _create_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, label_dataset = [], [], [], []
        for i in range(self.batch_size):
            cls = self._gen_random_cls()

            token_data = self.tokenize_data(self.seperated_data[cls][self._index[cls]])
            self._index[cls] += 1
            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])
            if self.with_labels:
                if Config.framework == "tensorflow":
                    label_dataset.append([token_data[3]])
                elif Config.framework == "pytorch":
                    label_dataset.append(torch.tensor([token_data[3]]))
                else:
                    raise ValueError("Unsupported framework: {}".format(Config.framework))
            self._iter += 1
            if self._iter == self.data_length:
                self.epoch_end = True
            if self._index[cls] == len(self.seperated_data[cls]):
                self._index[cls] = 0
                self._reset(cls=cls)

        if Config.framework == "tensorflow":
            if self.with_labels:
                return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset, label_dataset))
            return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                            token_type_ids_dataset))
        elif Config.framework == "pytorch":
            if self.with_labels:
                if is_gpu_available():
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size,
                                                                        self.max_len).cuda(), torch.cat(
                        label_dataset, 0).reshape(self.batch_size).cuda()
                else:
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                        label_dataset, 0).reshape(self.batch_size, self.max_len)
            if is_gpu_available():
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda()
            else:
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size)
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))

    def __len__(self):  # return sample count
        return self.data_length

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_end is True:
            raise StopIteration()
        return self._create_data()

    def get_batch_data(self):  # get tokenized sample by index
        return self._create_data()


class DatasetLM(DatasetBase):
    '''
    Language Model Dataset
    '''

    def __init__(self, data, max_len, tokenizer, batch_size, shuffle=True, with_labels=True, EDA=None):

        super(DatasetLM, self).__init__(data, max_len, tokenizer, batch_size, shuffle)
        self.eda = EDA
        if self.eda is not None and with_labels is True:
            self.data = self._data_argument()
        self.with_labels = with_labels  # data with labels or not
        self._index = 0
        self.epoch_length = len(self.data) // self.batch_size if len(self.data) % self.batch_size == 0 \
            else len(self.data) // self.batch_size + 1
        self.tokenized_data = []
        self.epoch_end = False
        self._reset()

    def _data_argument(self):
        _data = []
        n = 0
        logger.info("Processing data argument: {}".format(len(self.data)))
        for d in self.data:
            _sequences = []
            for l in d[:-1]:
                _sequences.append(self.eda(l))
            _data.append(d)
            for i in range(10):
                try:
                    _d = []
                    for s in range(len(_sequences)):
                        _d.append(_sequences[s][i])
                    _d.append(d[-1])
                    _data.append(_d)
                except Exception:
                    pass
            sys.stdout.write("Processing evaluation: {}/{}".format(n, len(self.data)) + '\r')
            sys.stdout.flush()
            n += 1
        return TXTDataset(_data, range(len(_data[0]))[:-1], label_index=len(_data[0]) - 1)

    def __random_mlm(self, seqs, p=0.85):
        labels = []
        inputs = []

        for seq in seqs:
            size = np.where(seq == 0)[0]
            if size.shape[0] == 0:
                size = self.max_len
            else:
                size = size[0]

            label = np.ones(self.max_len) * -100
            input = np.zeros(self.max_len)

            for index in range(size):
                if Config.framework == "tensorflow":
                    element = tf.identity(seq[index])
                elif Config.framework == "pytorch":
                    element = seq[index].clone()
                else:
                    raise ValueError("Unsupported framework: {}".format(Config.framework))
                input[index] = element
                # sample selection
                if element <= self.tokenizer.mask_token_id():
                    label[index] = element
                    continue
                if random.random() > p:
                    rand = random.random()
                    if rand < 0.8:
                        label[index] = element
                        if Config.framework == "tensorflow":
                            input[index] = self.tokenizer.mask_token_id()
                        elif Config.framework == "pytorch":
                            seq[index] = self.tokenizer.mask_token_id()
                        else:
                            raise ValueError("Unsupported framework: {}".format(Config.framework))
                    elif rand > 0.8 and rand < 0.9:
                        label[index] = element
                        if Config.framework == "tensorflow":
                            input[index] = random.randint(self.tokenizer.mask_token_id() + 1,
                                                          self.tokenizer.vocab_size() - 1)
                        elif Config.framework == "pytorch":
                            seq[index] = random.randint(self.tokenizer.mask_token_id() + 1,
                                                        self.tokenizer.vocab_size() - 1)
                        else:
                            raise ValueError("Unsupported framework: {}".format(Config.framework))
                    else:
                        pass
            labels.append(label.astype(np.int32))
            if Config.framework == 'tensorflow':
                inputs.append(tf.convert_to_tensor(input.astype(np.int32)))
        if Config.framework == 'tensorflow':
            return inputs, labels
        return seqs, np.array(labels).astype(np.int)

    def tokenize_data(self, d):
        encoded_pair = self.tokenizer(d)
        token_data = [self._squeeze(encoded_pair['input_ids']),
                      self._squeeze(encoded_pair['attention_mask']), self._squeeze(encoded_pair['token_type_ids'])]

        if self.with_labels:
            if len(d) >= 2:
                try:
                    token_data.append(int(d[-1]))
                except Exception:
                    raise ValueError("Please make sure label column is correct!")
            else:
                raise IndexError("No label data is founded, make sure the dataset correct!")
        return token_data

    def tokenize_full_data(self):
        progress = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#')]).start()
        logger.info("Processing dataset tokenization: {}".format(len(self.data)))
        for i in progress(range(len(self.data))):
            d = self.data[i]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            token_data = self.tokenize_data(d)
            self.tokenized_data.append(token_data)
            progress.update()

    def _reset(self):
        self.epoch_end = False
        self._index = 0
        if self.shuffle is True:
            self.data.shuffle()

    def _create_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, mlm_labels = [], [], [], []
        for i in range(self.batch_size):
            data_cell = self.data[self._index]
            token_data = self.tokenize_data(data_cell)

            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])

            self._index += 1
            if self._index == len(self.data):
                self._index = 0
                self.epoch_end = True

        if Config.framework == "tensorflow":
            if self.with_labels:
                token_ids_dataset, mlm_labels = self.__random_mlm(token_ids_dataset)
                return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset, mlm_labels))
            return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                            token_type_ids_dataset))
        elif Config.framework == "pytorch":
            token_ids_dataset = torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len)
            attn_masks_dataset = torch.cat(attn_masks_dataset, 0).reshape(self.batch_size, self.max_len)
            token_type_ids_dataset = torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size, self.max_len)

            if self.with_labels:
                token_ids_dataset, mlm_labels = self.__random_mlm(token_ids_dataset)
                mlm_labels = torch.from_numpy(mlm_labels)
                if is_gpu_available():
                    return token_ids_dataset.cuda(), attn_masks_dataset.cuda(), \
                           token_type_ids_dataset.cuda(), mlm_labels.cuda()
                else:
                    return token_ids_dataset, attn_masks_dataset, \
                           token_type_ids_dataset, mlm_labels

            if is_gpu_available():
                return token_ids_dataset.cuda(), attn_masks_dataset.cuda(), \
                       token_type_ids_dataset.cuda()
            else:
                return token_ids_dataset, attn_masks_dataset, \
                       token_type_ids_dataset
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))

    def __len__(self):  # return sample count
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_end is True:
            self._reset()
            raise StopIteration()
        return self._create_data()

    def get_batch_data(self):  # get tokenized sample by index
        if self.epoch_end is True:
            self._reset()
        return self._create_data()


class DatasetCustom(DatasetBase):

    def __init__(self, data, max_len, tokenizer, batch_size, data_column, shuffle=True, with_label=False):
        super(DatasetCustom, self).__init__(data, max_len, tokenizer, batch_size, shuffle)
        self.data_column = data_column
        self._index = 0
        self.epoch_length = len(self.data) // self.batch_size if len(self.data) % self.batch_size == 0 \
            else len(self.data) // self.batch_size + 1
        self.tokenized_data = []
        self.epoch_end = False
        self.with_label = with_label
        self._reset()

    def tokenize_data(self, data):
        d, indicators = [], []
        for t in self.data_column[0]:
            assert t < len(data)
            d.append(data[t])
        for t in self.data_column[1]:
            assert t < len(data)
            indicators.append(data[t])
        encoded_pair = self.tokenizer(d)
        token_data = [self._squeeze(encoded_pair['input_ids']),
                      self._squeeze(encoded_pair['attention_mask']), self._squeeze(encoded_pair['token_type_ids']),
                      indicators]

        return token_data

    def tokenize_full_data(self):
        progress = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#')]).start()
        logger.info("Processing dataset tokenization: {}".format(len(self.data)))
        for i in progress(range(len(self.data))):
            d = self.data[i]
            # Tokenize the pair of sentences to get token ids, attention masks and token type ids
            token_data = self.tokenize_data(d)
            self.tokenized_data.append(token_data)
            progress.update()

    def _reset(self):
        self.epoch_end = False
        self._index = 0
        if self.shuffle is True:
            self.data.shuffle()

    def _create_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, indicator_dataset = [], [], [], []
        for i in range(self.batch_size):
            token_data = self.tokenize_data(self.data[self._index])
            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])

            if self.with_label is True:
                indicator_dataset.append(token_data[3])

            self._index += 1
            if self._index == len(self.data):
                self._index = 0
                self.epoch_end = True

        if Config.framework == "tensorflow":
            if self.with_label is True:
                return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset, indicator_dataset))
            return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                                token_type_ids_dataset))
        elif Config.framework == "pytorch":
            if self.with_label is True:
                if is_gpu_available():
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size,
                                                                        self.max_len).cuda(), indicator_dataset
                else:
                    return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                        attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                           torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size), indicator_dataset
            if is_gpu_available():
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len).cuda(), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size,
                                                                    self.max_len).cuda()
            else:
                return torch.cat(token_ids_dataset, 0).reshape(self.batch_size, self.max_len), torch.cat(
                    attn_masks_dataset, 0).reshape(self.batch_size, self.max_len), \
                       torch.cat(token_type_ids_dataset, 0).reshape(self.batch_size)
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))

    def __len__(self):  # return sample count
        return len(self.data)

    def __iter__(self):
        return self

    def __next__(self):
        if self.epoch_end is True:
            self._reset()
            raise StopIteration()
        return self._create_data()

    def get_batch_data(self):  # get tokenized sample by index
        if self.epoch_end is True:
            self._reset()
        return self._create_data()

    def get_full_data(self):
        token_ids_dataset, attn_masks_dataset, token_type_ids_dataset, indicator_dataset = [], [], [], []
        for i in range(len(self.data)):
            token_data = self.tokenize_data(self.data[i])
            token_ids_dataset.append(token_data[0])
            attn_masks_dataset.append(token_data[1])
            token_type_ids_dataset.append(token_data[2])
            indicator_dataset.append(token_data[3])

        if Config.framework == "tensorflow":
            return self.tf_distribute_data((token_ids_dataset, attn_masks_dataset,
                                            token_type_ids_dataset, indicator_dataset))
        elif Config.framework == "pytorch":
            data_size = len(self.data)
            if is_gpu_available():
                return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len).cuda(), torch.cat(
                    attn_masks_dataset, 0).reshape(data_size, self.max_len).cuda(), \
                       torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len).cuda(), indicator_dataset
            else:
                return torch.cat(token_ids_dataset, 0).reshape(data_size, self.max_len), torch.cat(
                    attn_masks_dataset, 0).reshape(data_size, self.max_len), \
                       torch.cat(token_type_ids_dataset, 0).reshape(data_size, self.max_len), indicator_dataset
        else:
            raise ValueError("Unsupported framework: {}".format(Config.framework))


if __name__ == '__main__':
    from hypernlp.nlp.data_process.reader import CSVReader
    from utils.string_utils import generate_model_name, home_path
    from hypernlp.dl_framework_adaptor.configs.config import bert_models_config
    from hypernlp.nlp.tokenizer import TokenizerNSP
    from utils.gpu_status import environment_check

    environment_check()

    data = CSVReader("/home/luhf/dataset/", None).train_data(["s1", "s2"], "class_label")

    nsp_tokenizer = TokenizerNSP(model_path=home_path() + bert_models_config[
        generate_model_name("bert", Config.framework,
                            "chinese")]["BASE_MODEL_PATH"], max_len=128)

    train_data = DatasetLM(data.train_data, 128, nsp_tokenizer,
                           batch_size=8,
                           with_labels=True, EDA=None)

    d = next(iter(train_data.get_batch_data()))

    print(d[0], d[1], d[2], d[3])