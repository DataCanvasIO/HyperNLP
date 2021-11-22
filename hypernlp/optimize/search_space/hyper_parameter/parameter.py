import os


class Param(object):

    def __init__(self, choice):
        self.choice = choice
        self.index = 0

    def __getitem__(self, index):
        if index < len(self.choice):
            return self.choice[index]
        else:
            raise ValueError("Index is out of range: {} >= len ({}).".format(index, len(self.choice)))

    def __iter__(self):
        return self

    def __next__(self):
        if self.index == len(self.choice):
            self.index = 0
            raise StopIteration()
        data = self.choice[self.index]
        self.index += 1
        return data

    def sorted(self, reverse=False):
        self.choice.sort(reverse=reverse)

