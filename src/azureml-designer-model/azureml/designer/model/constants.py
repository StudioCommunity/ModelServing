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


class ModelSpecConstants:
    # Top level keys in model_spec
    FLAVOR_KEY = "flavor"
    MODEL_FILE_KEY = "model_file"
    CONDA_FILE_KEY = "conda_file"
    LOCAL_DEPENDENCIES_KEY = "local_dependencies"
    INPUTS_KEY = "inputs"
    OUTPUTS_KEY = "outputs"
    TASK_TYPE_KEY = "task_type"
    LABEL_MAP_FILE_KEY = "label_map_file"
    SERVING_CONFIG_KEY = "serving_config"
    DESCRIPTION_KEY = "description"
    TIME_CREATED_KEY = "time_created"

    # Flavor specified keys in model_spec
    FLAVOR_NAME_KEY = "name"
    SERIALIZATION_METHOD_KEY = "serialization_method"
    MODEL_CLASS_KEY = "class"
    MODEL_MODULE_KEY = "module"
    IS_CUDA_KEY = "is_cuda"
    INIT_PARAMS_KEY = "init_params"

    # Others
    DEFAULT_ARTIFACT_SAVE_PATH = "./AzureMLModel"
    CONDA_FILE_NAME = "conda.yaml"
    CONDA_ENV_NAME = "project_environment"
    MODEL_SPEC_FILE_NAME = "model_spec.yaml"
    LOCAL_DEPENDENCIES_PATH = "local_dependencies"
    CUSTOM_MODEL_FLAVOR_NAME = "custom"
    CUSTOM_MODEL_DIRECTORY = "model"
    PICKLE_MODEL_FILE_NAME = "model.pkl"
    PYTORCH_STATE_DICT_FILE_NAME = "state_dict.pt"
    LABEL_MAP_FILE_NAME = "index_to_label.csv"
