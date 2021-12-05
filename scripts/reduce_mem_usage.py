# sourced from: https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

import pandas as pd
from tqdm import tqdm
import numpy as np

def reduce_mem_usage(df):
    """
    reduce_mem_usage
    Function to reduce the memory usage of a dataframe by changing the data type of each column.

    Args:
        df (pandas.DataFrame): the dataframe to be reduced

    Returns:
        pandas.DatFrame: the reduced dataframe
    """    
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("\tMemory usage of dataframe is :",round(start_mem_usg,2)," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for i in tqdm(
        range(len(df.columns)), 
        desc='Processing columns', 
        unit='row'
    ):
        col = df.columns[i]
        if df[col].dtype != object:  # Exclude strings
            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
    
    # Print final result
    print("\t___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("\tMemory usage is: ", round(mem_usg,2)," MB")
    print("\tThis is ",round(100*mem_usg/start_mem_usg,2),"% of the initial size")
    return df, NAlist
