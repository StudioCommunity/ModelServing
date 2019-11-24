CONDA_FILE_NAME = "conda.yaml"
CONDA_ENV_NAME = "project_environment"
MODEL_SPEC_FILE_NAME = "model_spec.yaml"
LOCAL_DEPENDENCIES_PATH = "local_dependencies"
CUSTOM_MODEL_DIRECTORY = "model"
CUSTOM_MODEL_FLAVOR_NAME = "custom"
PYTORCH_MODEL_FILE_NAME = "model.pkl"
LABEL_MAP_FILE_NAME = "index_to_label.csv"

# Copied from azureml.studio.modules.ml.common.constants
# TODO: Support image generation and other task type
class ScoreColumnConstants:
    # Label and Task Type Region
    BinaryClassScoredLabelType = "Binary Class Assigned Labels"
    MultiClassScoredLabelType = "Multi Class Assigned Labels"
    RegressionScoredLabelType = "Regression Assigned Labels"
    ClusterScoredLabelType = "Cluster Assigned Labels"
    ScoredLabelsColumnName = "Scored Labels"
    ClusterAssignmentsColumnName = "Assignments"
    # Probability Region
    CalibratedScoreType = "Calibrated Score"
    ScoredProbabilitiesColumnName = "Scored Probabilities"
    ScoredProbabilitiesMulticlassColumnNamePattern = "Scored Probabilities"
    # Distance Region
    ClusterDistanceMetricsColumnNamePattern = "DistancesToClusterCenter no."

# temp solution, would remove later
DATA_TYPE_FILE_NAME = "data_type.json"
DATA_ILEARNER_FILE_NAME = "data.ilearner"
