import os

from .task_type import TaskType
from .label_map import LabelMap
from ..constants import ModelSpecConstants


class Task(object):
    """
    Specify machine learning task and additional info need for the task
    """
    task_type: TaskType = None
    label_map: LabelMap = LabelMap()
    ground_truth_column_name: str = None

    def __init__(
            self,
            task_type: TaskType = None,
            label_map: LabelMap = LabelMap(),
            ground_truth_column_name: str = None
    ):
        self.task_type = task_type
        self.label_map = label_map
        self.ground_truth_column_name = ground_truth_column_name

    def save(self, artifact_path: str, overwrite_if_exists=True) -> dict:
        """
        Save task-related info to artifact_path, return "task" field in model_spec
        :param artifact_path:
        :param overwrite_if_exists:
        :return:
        """
        task_conf = {
            ModelSpecConstants.TASK_TYPE_KEY: self.task_type.name if self.task_type else None,
            ModelSpecConstants.GROUND_TRUTH_COLUMN_NAME_KEY: self.ground_truth_column_name
        }
        if self.label_map:
            label_map_path = os.path.join(artifact_path, ModelSpecConstants.LABEL_MAP_FILE_NAME)
            self.label_map.save(label_map_path, overwrite_if_exists=overwrite_if_exists)
            task_conf[ModelSpecConstants.LABEL_MAP_FILE_KEY] = ModelSpecConstants.LABEL_MAP_FILE_NAME
        return task_conf

    @classmethod
    def load(cls, artifact_path: str, task_conf: dict):
        task_type = None
        label_map = LabelMap()
        if ModelSpecConstants.TASK_TYPE_KEY in task_conf and task_conf[ModelSpecConstants.TASK_TYPE_KEY]:
            task_type = TaskType[task_conf[ModelSpecConstants.TASK_TYPE_KEY]]
        if ModelSpecConstants.LABEL_MAP_FILE_KEY in task_conf:
            label_map = LabelMap.create(os.path.join(artifact_path, task_conf[ModelSpecConstants.LABEL_MAP_FILE_KEY]))
        ground_truth_column_name = task_conf.get(ModelSpecConstants.LABEL_MAP_FILE_KEY, None)
        return cls(task_type=task_type, label_map=label_map, ground_truth_column_name=ground_truth_column_name)
