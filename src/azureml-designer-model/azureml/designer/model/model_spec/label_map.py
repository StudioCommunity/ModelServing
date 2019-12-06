import pandas as pd

from ..utils import ioutils


# TODO: Add unittest
class LabelMap(object):

    _index_to_label = {}

    def __init__(self):
        pass

    def save(self, save_to, overwrite_if_exists=True):
        ioutils.validate_overwrite(save_to, overwrite_if_exists)
        index_list = list(self._index_to_label.keys())
        label_list = list(self._index_to_label.values())
        df = pd.DataFrame(index=index_list, data={"label": label_list})
        df.to_csv(save_to)

    @classmethod
    def create_from_csv(cls, file_path):
        df = pd.read_csv(file_path, index_col=0)
        label_map = LabelMap()
        label_map.index_to_label_dict = df.to_dict()["label"]
        return label_map

    @classmethod
    def create_from_dict(cls, label_dict):
        label_map = LabelMap()
        label_map.index_to_label_dict = label_dict
        return label_map

    @classmethod
    def create_from_list(cls, label_list):
        label_map = LabelMap()
        label_map.index_to_label_dict = dict(enumerate(label_list))
        return label_map
    
    # TODO: Add input validation here
    @classmethod
    def create(cls, param):
        if isinstance(param, str):
            return cls.create_from_csv(param)
        if isinstance(param, dict):
            return cls.create_from_dict(param)
        if isinstance(param, list):
            return cls.create_from_list(param)
    
    @property
    def index_to_label_dict(self):
        return self._index_to_label

    @index_to_label_dict.setter
    def index_to_label_dict(self, value):
        self._index_to_label = value

    # TODO: implement this
    def transform(self):
        pass

    # TODO: Support list input
    def inverse_transform(self, label_ids):
        """
        label_id -> label_name
        :return:
        """
        return [self._index_to_label.get(i, i) for i in label_ids]
