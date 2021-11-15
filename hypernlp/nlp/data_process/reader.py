import os

import pandas as pd

from hypernlp.nlp.data_process.format_dataset import CSVDataset, TXTDataset
from utils.logger import logger

CLS2IDX = {'负向': 2, '正向': 1, '中立': 0}
IDX2CLS = {'2': '负向', '1': '正向', '0': '中立'}


def process_data_csv(filename, classes2idx):
    df = pd.read_csv(filename, encoding='utf-8')
    if classes2idx is not None:
        df = df.replace({'class_label': classes2idx})  # mapping
    return df


def process_data_txt(filename, classes2idx, spliter, skip_title, label_index):
    df = []
    with open(filename) as f:
        for line in f:
            if skip_title:
                skip_title = False
                continue
            if line is None:
                continue
            df.append(line.strip().split(spliter))
            if classes2idx is not None:
                df[-1][label_index] = classes2idx[df[-1][label_index]]
    return df


class Reader(object):
    def __init__(self, data_folder, columns, classes2idx):
        self.data_folder = data_folder
        self.columns = columns
        self.classes2idx = classes2idx


class CSVReader(Reader):
    def __init__(self, data_folder, columns, classes2idx):
        super(CSVReader, self).__init__(data_folder, columns, classes2idx)
        if os.path.exists("".join([data_folder, "test.csv"])):
            self.test_data = CSVDataset(process_data_csv("".join([data_folder, "test.csv"]), None), columns,
                                        False)
            logger.info("Finish loading test data.")
        else:
            self.test_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "test.csv"])))

        if os.path.exists("".join([data_folder, "train.csv"])):
            self.train_data = CSVDataset(process_data_csv("".join([data_folder, "train.csv"]), classes2idx),
                                         columns,
                                         True)
            logger.info("Finish loading train data.")
        else:
            self.train_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "train.csv"])))

        if os.path.exists("".join([data_folder, "validation.csv"])):
            self.validate_data = CSVDataset(
                process_data_csv("".join([data_folder, "validation.csv"]), classes2idx),
                columns,
                True)
            logger.info("Finish loading validation data.")
        else:
            self.validate_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "validation.csv"])))


class TXTReader(Reader):
    def __init__(self, data_folder, columns, classes2idx, label_index, spliter, skip_title=True):
        super(TXTReader, self).__init__(data_folder, columns, classes2idx)
        if os.path.exists("".join([data_folder, "test.txt"])):
            self.test_data = TXTDataset(process_data_txt("".join([data_folder, "test.txt"]),
                                                         None, spliter=spliter, skip_title=skip_title, label_index=0),
                                        columns, 0)
            logger.info("Finish loading test data.")
        else:
            self.test_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "test.txt"])))

        if os.path.exists("".join([data_folder, "train.txt"])):
            self.train_data = TXTDataset(process_data_txt("".join([data_folder, "train.txt"]),
                                                          classes2idx, spliter=spliter, skip_title=skip_title,
                                                          label_index=label_index),
                                         columns,
                                         label_index)
            logger.info("Finish loading train data.")
        else:
            self.train_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "train.txt"])))

        if os.path.exists("".join([data_folder, "validation.txt"])):
            self.validate_data = TXTDataset(process_data_txt("".join([data_folder, "validation.txt"]),
                                                             classes2idx, spliter=spliter, skip_title=skip_title,
                                                             label_index=label_index),
                                            columns,
                                            label_index)
            logger.info("Finish loading validation data.")
        else:
            self.validate_data = None
            logger.warning("Cannot find '{}' file!".format("".join([data_folder, "validation.txt"])))




if __name__ == '__main__':
    data = CSVReader("../data/", ["content"], CLS2IDX)
    print(data.train_data[0])
    print(data.test_data[0])
    print(data.validate_data[0])

    data = TXTReader("../data/", [2], IDX2CLS, 1, spliter=",")
    print(data.train_data[0])
    print(data.test_data[0])
    print(data.validate_data[0])
