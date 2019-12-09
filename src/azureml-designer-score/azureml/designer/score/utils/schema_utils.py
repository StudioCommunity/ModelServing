from ..logger import get_logger

logger = get_logger(__name__)


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


def _filter_column_names_with_prefix(name_list, prefix=''):
    # if prefix is '', all string.startswith(prefix) is True.
    if prefix == '':
        return name_list
    return [column_name for column_name in name_list if column_name.startswith(prefix)]


# TODO: Support other task types
def generate_score_column_meta(predict_df):
    score_columns = {x: x for x in _filter_column_names_with_prefix(
        predict_df.columns.tolist(), prefix=ScoreColumnConstants.ScoredProbabilitiesMulticlassColumnNamePattern)}
    score_columns[ScoreColumnConstants.MultiClassScoredLabelType] = ScoreColumnConstants.ScoredLabelsColumnName
    logger.info("Multi-class Classification Model Scored Columns are: ")
    return score_columns
