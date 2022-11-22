import abc


class MetricCalculator(abc.ABC):
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.get_names())

    def get_names(self):
        raise NotImplementedError
