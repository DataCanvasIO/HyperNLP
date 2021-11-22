import abc
import os

from utils.csv_utils import process_data_csv
from utils.txt_utils import process_data_txt
from hypernlp.nlp.data_process.format_dataset import CSVDataset, TXTDataset
from utils.logger import logger


class Reader(object):
    def __init__(self, data_folder, classes2idx):
        self.data_folder = data_folder
        self.classes2idx = classes2idx

        self._test_data = None
        self._train_data = None
        self._validate_data = None

    @abc.abstractmethod
    def test_data(self, columns, label_name, with_label=False):
        pass

    @abc.abstractmethod
    def train_data(self, columns, label_name, with_label=True):
        pass

    @abc.abstractmethod
    def validate_data(self, columns, label_name, with_label=True):
        pass


class CSVReader(Reader):
    def __init__(self, data_folder, classes2idx):
        super(CSVReader, self).__init__(data_folder, classes2idx)
        if os.path.exists("".join([data_folder, "test.csv"])):
            self._test_data = process_data_csv("".join([data_folder, "test.csv"]))
            logger.info("Finish loading test data.")
        else:
            self._test_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "test.csv"])))

        if os.path.exists("".join([data_folder, "train.csv"])):
            self._train_data = process_data_csv("".join([data_folder, "train.csv"]))
            logger.info("Finish loading train data.")
        else:
            self._train_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "train.csv"])))

        if os.path.exists("".join([data_folder, "validation.csv"])):
            self._validate_data = process_data_csv("".join([data_folder, "validation.csv"]))
            logger.info("Finish loading validation data.")
        else:
            self._validate_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "validation.csv"])))

    def test_data(self, columns, label_name=None, with_label=False):
        return CSVDataset(self._test_data, columns, label_name, with_label)

    def train_data(self, columns, label_name, with_label=True):
        if self.classes2idx is not None:
            df = self._train_data.replace({label_name: self.classes2idx})
            return CSVDataset(df, columns, label_name, with_label)
        return CSVDataset(self._train_data, columns, label_name, with_label)

    def validate_data(self, columns, label_name, with_label=True):
        if self.classes2idx is not None:
            df = self._validate_data.replace({label_name: self.classes2idx})
            return CSVDataset(df, columns, label_name, with_label)
        return CSVDataset(self._validate_data, columns, label_name, with_label)


class TXTReader(Reader):
    def __init__(self, data_folder, classes2idx, spliter, skip_title=True):
        super(TXTReader, self).__init__(data_folder, classes2idx)
        if os.path.exists("".join([data_folder, "test.txt"])):
            self._test_data = process_data_txt("".join([data_folder, "test.txt"]),
                                               spliter=spliter, skip_title=skip_title)
            logger.info("Finish loading test data.")
        else:
            self._test_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "test.txt"])))

        if os.path.exists("".join([data_folder, "train.txt"])):
            self._train_data = process_data_txt("".join([data_folder, "train.txt"]),
                                                spliter=spliter, skip_title=skip_title)
            logger.info("Finish loading train data.")
        else:
            self._train_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "train.txt"])))

        if os.path.exists("".join([data_folder, "validation.txt"])):
            self._validate_data = process_data_txt("".join([data_folder, "validation.txt"]),
                                                   spliter=spliter, skip_title=skip_title)
            logger.info("Finish loading validation data.")
        else:
            self._validate_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "validation.txt"])))

    def test_data(self, columns, label_index=None, with_label=False):
        return TXTDataset(self._test_data, columns, label_index, with_label)

    def train_data(self, columns, label_index, with_label=True):
        df = self._train_data
        if self.classes2idx is not None:
            for i in range(len(df)):
                df[i][label_index] = self.classes2idx[df[i][label_index]]
        return TXTDataset(df, columns, label_index, with_label)

    def validate_data(self, columns, label_index, with_label=True):
        df = self._validate_data
        if self.classes2idx is not None:
            for i in range(len(df)):
                df[i][label_index] = self.classes2idx[df[i][label_index]]
        return TXTDataset(df, columns, label_index, with_label)


if __name__ == '__main__':

    CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
    IDX2CLS = {2: '负向', 1: '正向', 0: '中立'}

    train = CSVReader("../data/", CLS2IDX).train_data(["content"], label_name='label')
    test = CSVReader("../data/", CLS2IDX).test_data(["content"])
    validate = CSVReader("../data/", CLS2IDX).validate_data(["content"], label_name='label')
    print(train[0])
    print(test[0])
    print(validate[0])

    train = TXTReader("../data/", IDX2CLS, spliter=",").train_data([2], label_index=1)
    test = TXTReader("../data/", IDX2CLS, spliter=",").test_data([2])
    validate = TXTReader("../data/", IDX2CLS, spliter=",").validate_data([2], label_index=1)
    print(train[0])
    print(test[0])
    print(validate[0])
