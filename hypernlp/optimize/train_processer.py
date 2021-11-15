import abc

from hypernlp.config import Config


class TrainProcesser(object):

    def __init__(self, dataset, losses):
        self.dataset = dataset
        self.losses = losses
        assert isinstance(self.losses, list) or isinstance(self.losses, tuple)
        assert len(self.losses) == 2

    @abc.abstractmethod
    def process(self, model, x, y, index):
        pass

    @abc.abstractmethod
    def step(self, model, batch_data, index):
        pass


class TFTrainProcesser(TrainProcesser):

    def __init__(self, data, losses):
        super(TFTrainProcesser, self).__init__(data, losses)

    def step(self, model, batch_data, index):
        x_batch, y_batch = batch_data[:3], batch_data[-1]
        return self.process(model, x_batch, y_batch, index)

    def process(self, model, x, y, index):
        # Wrap computation inside a GradientTape for automatic differentiation.
        # Forward pass

        pred = model(x, training=True)

        if isinstance(pred, tuple):
            assert len(pred) > index
            loss = self.losses[0](pred[index], y) * self.losses[1]
        else:
            loss = self.losses[0](pred, y) * self.losses[1]

        return loss


class PTTrainProcesser(TrainProcesser):

    def __init__(self, data, losses):
        super(PTTrainProcesser, self).__init__(data, losses)

    def step(self, model, batch_data, index):
        x_batch, y_batch = batch_data[:3], batch_data[-1]
        return self.process(model, x_batch, y_batch, index)

    def process(self, model, x, y, index):
        # Forward pass
        pred = model(x)

        if isinstance(pred, tuple):
            loss = self.losses[0](pred[index], y) * self.losses[1]
        else:
            loss = self.losses[0](pred, y) * self.losses[1]

        return loss


def train_processer(dataset, losses):
    if Config.framework == "tensorflow":
        return TFTrainProcesser(dataset, losses)
    elif Config.framework == "pytorch":
        return PTTrainProcesser(dataset, losses)
    else:
        raise ValueError("Unsupported framework: {}".format(Config.framework))


