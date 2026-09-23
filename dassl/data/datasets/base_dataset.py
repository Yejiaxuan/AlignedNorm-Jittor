import random
from collections import defaultdict

from dassl.utils import check_isfile


class Datum:
    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)
        assert check_isfile(impath)
        self._impath = impath
        self._label = int(label)
        self._domain = int(domain)
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DatasetBase:
    dataset_dir = ""
    domains = []

    def __init__(self, train_x=None, train_u=None, val=None, test=None):
        self._train_x = train_x or []
        self._train_u = train_u or []
        self._val = val or []
        self._test = test or []
        self._num_classes = self.get_num_classes(self._train_x)
        self._lab2cname, self._classnames = self.get_lab2cname(self._train_x)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_u(self):
        return self._train_u

    @property
    def val(self):
        return self._val

    @property
    def test(self):
        return self._test

    @property
    def lab2cname(self):
        return self._lab2cname

    @property
    def classnames(self):
        return self._classnames

    @property
    def num_classes(self):
        return self._num_classes

    @staticmethod
    def get_num_classes(data_source):
        labels = {item.label for item in data_source}
        return max(labels) + 1 if labels else 0

    @staticmethod
    def get_lab2cname(data_source):
        mapping = {item.label: item.classname for item in data_source}
        labels = sorted(mapping)
        return mapping, [mapping[label] for label in labels]

    def generate_fewshot_dataset(self, *data_sources, num_shots=-1, repeat=False):
        if num_shots < 1:
            return data_sources[0] if len(data_sources) == 1 else data_sources
        print(f"Creating a {num_shots}-shot dataset")
        output = []
        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)
            dataset = []
            for items in tracker.values():
                if len(items) >= num_shots:
                    dataset.extend(random.sample(items, num_shots))
                elif repeat:
                    dataset.extend(random.choices(items, k=num_shots))
                else:
                    dataset.extend(items)
            output.append(dataset)
        return output[0] if len(output) == 1 else output

    @staticmethod
    def split_dataset_by_label(data_source):
        output = defaultdict(list)
        for item in data_source:
            output[item.label].append(item)
        return output
