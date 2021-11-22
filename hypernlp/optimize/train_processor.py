import abc

from hypernlp.framework_config import Config


class TrainProcessor(object):

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


if Config.framework == 'tensorflow':

    class TFTrainProcessor(TrainProcessor):

        def __init__(self, data, losses):
            super(TFTrainProcessor, self).__init__(data, losses)

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

elif Config.framework == 'pytorch':

    class PTTrainProcessor(TrainProcessor):

        def __init__(self, data, losses):
            super(PTTrainProcessor, self).__init__(data, losses)

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
else:
    raise TypeError("Unsupported framework: '{}'".format(Config.framework))


def train_processor(dataset, losses):
    if Config.framework == "tensorflow":
        return TFTrainProcessor(dataset, losses)
    elif Config.framework == "pytorch":
        return PTTrainProcessor(dataset, losses)
    else:
        raise TypeError("Unsupported framework: {}".format(Config.framework))


