import os

import pandas as pd

from ..utils import ioutils
from .. import constants

# TODO: Add validation here
class LabelMap(object):

    _index_to_label = {}

    def save(self, artifact_path, overwrite_if_exists=True):
        save_to = os.path.join(artifact_path, constants.LABEL_MAP_FILE_NAME)
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
    
    @property
    def index_to_label_dict(self):
        return self._index_to_label
    
    @index_to_label_dict.setter
    def index_to_label_dict(self, value):
        self._index_to_label = value