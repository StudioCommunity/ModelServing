from enum import Enum

class TaskType(Enum):
    Regression = 0
    BinaryClassification = 1
    MultiClassification = 2
    Clustering = 3
    ImageGeneration = 4
    TextGeneration = 5