from ..logger import get_logger

logger = get_logger(__name__)
logger.info(f"in {__file__}")

def add_column_to_dataframe(input_df, results, target_column):
    if target_column in input_df.columns:
        logger.info(f"writing to column {target_column}")
        input_df[target_column] = results
    else:
        logger.info(f"append new column {target_column}")
        input_df.insert(len(input_df.columns), target_column, results, False)
