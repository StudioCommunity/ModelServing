from .task_type import TaskType
from .label_map import LabelMap


class Task(object):
    """
    Specify machine learning task and additional info need for the task
    """

    def __init__(
            self,
            task_type: TaskType = None,
            label_map: LabelMap = None,
            ground_truth_column_name: str = None
    ):
        pass

    def save(self, artifact_path: str, overwrite_if_exists=True):
        pass

    @classmethod
    def load(cls, artifact_path: str, task_conf: dict):
        pass
