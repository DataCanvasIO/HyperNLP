import pandas as pd


def process_data_csv(filename):
    df = pd.read_csv(filename, encoding='utf-8')
    return df


def combine(csv1, csv2, columns1, columns2):
    data1 = process_data_csv(csv1)
    data2 = process_data_csv(csv2)

    data = []

    for index in range(len(data1)):
        d = []
        for col in columns1:
            d.append(data1.loc[index, col])
        for col in columns2:
            d.append(data2.loc[index, col])

        data.append(d)
    return data




