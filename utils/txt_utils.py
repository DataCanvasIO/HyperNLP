import os


def process_data_txt(filename, spliter, skip_title):
    df = []
    with open(filename) as f:
        for line in f:
            if skip_title:
                skip_title = False
                continue
            if line is None:
                continue
            df.append(line.strip().split(spliter))
    return df