import abc
import random


class FormatData(object):
    def __init__(self, data, columns, with_labels=True):
        super(FormatData, self).__init__()
        self.data = data  # dataframe
        self.with_labels = with_labels  # data with labels or not
        self.columns = columns
        self._index = 0

    def __len__(self):  # return sample count
        return len(self.data)

    def __getitem__(self, index):
        pass

    def __iter__(self):
        return self

    def __setitem__(self, k, v):
        self.k = v

    def __next__(self):
        pass

    @abc.abstractmethod
    def shuffle(self):
        pass


class CSVDataset(FormatData):
    def __init__(self, data, columns, label_name, with_labels=True):
        super(CSVDataset, self).__init__(data, columns, with_labels)
        self.label_name = label_name
        if self.label_name is None:
            assert with_labels is False

    def __getitem__(self, index):
        # Selecting a sentence at the specified index in the data frame
        _data = []
        for c in self.columns:
            _data.append(str(self.data.loc[index, c]))

        if self.with_labels:  # True if the dataset has labels (when training or validating)
            _data.append(self.data.loc[index, self.label_name])
        return _data

    def __next__(self):
        if self._index == len(self.data):
            raise StopIteration()
        _data = []
        for c in self.columns:
            _data.append(str(self.data.loc[self._index, c]))

        if self.with_labels:  # True if the dataset has labels (when training or validating)
            _data.append(self.data.loc[self._index, self.label_name])
        self._index += 1
        return _data

    def shuffle(self):
        self.data = self.data.sample(frac=1).reset_index(drop=True)


class TXTDataset(FormatData):
    def __init__(self, data, columns, label_index, with_labels=True):
        super(TXTDataset, self).__init__(data, columns, with_labels)
        self.label_index = label_index
        if self.label_index is None:
            assert with_labels is False

    def __getitem__(self, index):
        # Selecting a sentence at the specified index in the data frame
        _data = []
        for c in self.columns:
            _data.append(str(self.data[index][c]))

        if self.with_labels:  # True if the dataset has labels (when training or validating)
            _data.append(self.data[index][self.label_index])
        return _data

    def __next__(self):
        if self._index == len(self.data):
            raise StopIteration()
        _data = []
        for c in self.columns:
            _data.append(str(self.data[self._index][c]))

        if self.with_labels:  # True if the dataset has labels (when training or validating)
            _data.append(self.data[self._index][self.label_index])
        self._index += 1
        return _data

    def shuffle(self):
        random.shuffle(self.data)
