import os
import random

from sklearn.utils import shuffle as reset


def train_validate_split(data_df, validate_size=0.2, shuffle=True, random_state=None):

    if shuffle:
        data_df = reset(data_df, random_state=random_state)

    train = []
    validate = []

    for d in data_df:
        if random.random() < validate_size:
            validate.append(d)
        else:
            train.append(d)
    return train, validate


def create_train_validate(full_data, validate_size=0.2, shuffle=True, random_state=None, with_column_name=True):
    data_df = []
    if not os.path.exists(full_data):
        raise FileExistsError("File '{}' not exists!".format(full_data))
    with open(full_data, 'r', encoding='UTF-8') as f:
        if with_column_name is True:
            for line in f:
                column_name = line
                break
        for line in f:
            data_df.append(line)
    train, validate = train_validate_split(data_df, validate_size, shuffle, random_state)
    data_path = os.path.dirname(full_data)
    validate_path = data_path + "/validation." + full_data.split(".")[-1]
    train_path = data_path + "/train." + full_data.split(".")[-1]
    if os.path.exists(validate_path):
        raise FileExistsError("File '{}' already exists!".format(validate_path))
    if os.path.exists(train_path):
        raise FileExistsError("File '{}' already exists!".format(train_path))

    with open(validate_path, "w", encoding='UTF-8') as writer:
        if with_column_name is True:
            writer.write(column_name)
        for d in validate:
            writer.write(d)
    with open(train_path, "w", encoding='UTF-8') as writer:
        if with_column_name is True:
            writer.write(column_name)
        for d in train:
            writer.write(d)


if __name__ == '__main__':

    from utils.string_utils import home_path

    create_train_validate(home_path() + "hypernlp/nlp/data/all.csv")
