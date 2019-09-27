import logging
import numpy as np

logging.info(f"in {__file__}")
logger = logging.getLogger(__name__)


def from_df_column_to_array(col):
    if(len(col)==0):
        return []
    
    if(col.dtype == 'O'):
        shape = []
        shape.append(len(col))
        shape.append(len(col[0]))
        values = np.zeros(shape)
        for i in range(len(col)):
            values[i] = col[i]
        return values
    else:
        return col.values