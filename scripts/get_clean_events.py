"""
This file contains a function to get the clean events dataframe for a game.

Event data is read from a pkl or csv file, processed and then returned as a zipped pickle file.
"""

from functools import partial
import os
import pandas as pd
import numpy as np
from datetime import datetime

def _separate_players(row, columns, player_id):
    """
    Helper function to create indices for player1 and player2.
    
    Args:
        row (pandas.Series): Row of dataframe.
        columns (list): List of columns in dataframe that include pid info
        player_id (str): Player id.

    Returns:
        bool: True if player_id is in columns, False otherwise.
    """

    # check if player_id is in any of the columns
    for col in columns:
        # skip if col is not in columns
        if col not in row.index:
            continue

        # check if player_id is in col
        if row[col] == player_id:
            return True

    return False


def separate_player_dfs(df):
    """
    Separate the player1 and player2 dataframes from the combined dataframe.

    Args:
        df (pandas.DataFrame): Combined dataframe.

    Returns:
        pandas.DataFrame: Player1 dataframe.
        pandas.DataFrame: Player2 dataframe.
    """
    
    
    pid_checklist = [
        'pid',
        'control_pid',
        'upkeep_pid',
        'killer_pid'
    ]

    # get all rows where any column in pid_checklist is 1
    func_part = partial(_separate_players, columns=pid_checklist, player_id=1)
    player1_index = df.apply(func_part, axis=1).values

    # get all rows where any column in pid_checklist is 2
    func_part = partial(_separate_players, columns=pid_checklist, player_id=2)
    player2_index = df.apply(func_part, axis=1).values

    # create an index of all cases where neither player1_index or player2_index is 1
    remainder_index = np.where(
        (player1_index == 0) & (player2_index == 0),
        True,
        False
    )

    # use indices to create the 3 dfs
    player1_df = df.loc[player1_index]
    player2_df = df.loc[player2_index]
    remainder_df = df.loc[remainder_index]

    # check that unit_id column exists in player1_df and player2_df
    if 'unit_id' not in player1_df.columns or \
        'unit_id' not in player2_df.columns:
        print('WARNING: unit_id column does not exist in player1_df or player2_df')
        return None, None



    # match rows of remainder_df to either player1_df or player2_df
    # use unit_id as the matching key
    player1_unit_id = player1_df['unit_id'].unique()
    player2_unit_id = player2_df['unit_id'].unique()

    # assign correct pid to each row in remainder_df using np.where
    remainder_df.loc[:, 'pid'] = np.where(
        remainder_df['unit_id'].isin(player1_unit_id),
        1,
        np.where(
            remainder_df['unit_id'].isin(player2_unit_id), 2, np.nan
        )
    )

    # add rows with pid 1 to player1_df and pid 2 to player2_df
    player1_df = player1_df.append(remainder_df.loc[remainder_df['pid'] == 1])
    player2_df = player2_df.append(remainder_df.loc[remainder_df['pid'] == 2])

    # we now have 2 dfs, one for each player
    # we don't need the remainder_df anymore
    return player1_df, player2_df


def _clean_columns_and_rows(
    df, 
    df_id, 
    cols,
    events,
    log_file="info/event_warnings.txt"):
    """
    This is a helper function which stores the list of events and columns to be 
    removed from a dataframe.
    It receives a dataframe, processes it and returns a dataframe with only the 
    columns and rows to keep.
    Expected columns and rows that are not found are logged to a file per 
    dataframe. 

    Args:
        df (dataframe): dataframe to be cleaned
        df_id (str): id of the dataframe, used to identify the dataframe in logs
        cols (list): list of columns to keep
        events (list): list of events to keep
        log_file (str): path to the file where warnings are logged

    Returns:
        dataframe: dataframe with only the columns and rows to keep

    """

    # generate keep_set a list of column from cols which are in df
    keep_set = set(cols).intersection(df.columns)

    # missing columns
    missing_columns = set(cols) - keep_set
        
    # check if the dataframe contains the necessary columns
    if len(missing_columns) > 0:
        # if not, log the event and columns to be removed to a file
        
        with open(log_file, 'a') as file:
            # get timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # log the event by timestamp and df_id
            file.write(f'{timestamp} - {df_id}\n')
            file.write(f'Dateframe shape - {df.shape}\n')
            file.write(f'Missing columns - {list(missing_columns)}\n\n')

    # drop the columns that are not needed
    df = df[keep_set]

    # keep only rows with names in events
    df = df[df['name'].isin(events)]

    # return the dataframe with only the columns to keep
    return df


def clean_player_events(df, raw_columns_list, filehash, dummies_list_location):
    """
    This function cleans the player events dataframe.

    Any new dummies that are created which are not already in dummies_list, are added to dummies_list. If dummies_list is not present at the location specified it is created

    Args:
        df (pandas.Dataframe): dataframe to be cleaned
        raw_columns_list (pandas.Dataframe): index of columns and how to treat them
        filehash (string): hash of the file currently being processed, for logging purposes
        dummies_list_location (string): location of the dummies_list file, relative or absolute
    
    Returns:
        pandas.DataFrame: the cleaned dataframe
    """    
    # start by removing index=0 rows
    df.drop(0, inplace=True, errors='ignore')

    events_to_keep = [
        'PlayerStatsEvent',
        'UnitBornEvent',
        'UnitDiedEvent',
        'UnitDoneEvent',
        'UpgradeCompleteEvent'
    ]

    # extract column_name from raw_columns_list where clean_bool == 1
    columns_to_keep = raw_columns_list.loc[
        raw_columns_list['clean_bool'] == 1,
        'column_name'
    ].tolist()

    # clean the dataframes
    df = _clean_columns_and_rows(
        df,
        filehash,
        columns_to_keep,
        events_to_keep
    )

    # create a list of original column names in df
    original_columns = df.columns.to_list()

    # list of object column targets
    object_columns = ['upgrade_type_name', 'name', 'unit_type_name']
    # check that object columns are in df
    if not set(object_columns).issubset(set(df.columns)):
        print('WARNING: object columns are not in df')
        return None

    # convert object columns to numeric
    df = pd.get_dummies(df, columns=object_columns)

    # # create a list of column names in df that are not in original_columns
    # dummy_columns = pd.Series([
    #     col for col in df.columns.to_list() if col not in original_columns
    # ])   

    # # a dummies list is used to track all the possible values for each dummy
    # # get the location of the dummies list from settings
    # dummies_list_location = dummies_list_location

    # # append dummy_columns series to csv file
    # with open(dummies_list_location, 'a') as file:
    #     dummy_columns.to_csv(file, header=False, index=False)

    return df


def _aggregate_frames(df, max_frame, frame_size, columns_list):
    """
    Function to aggregate the frames in a dataframe.

    Args:
        df (pandas.DataFrame): The dataframe to aggregate.
        frame_size (int): The number of frames to aggregate.

    Returns:
        df (pandas.DataFrame): The aggregated dataframe.
        index_range(list): The range of indices to keep.
    """

    col_settings = columns_list.copy()

    # set col_settings index to column_name
    col_settings.set_index('column_name', inplace=True)

    # create two lists, one for summation columns and one for mean agg types
    summation_columns = []
    mean_agg_columns = []

    # iterate over columns, checking their values in col_settings
    for col in df.columns:
        
        if col in col_settings.index:
            
            if col_settings.loc[col, 'agg_type'] == 'mean':
                mean_agg_columns.append(col)

        else:
            summation_columns.append(col)

    # create a new df with only the columns for each agg type
    df_summation = df[summation_columns]
    df_mean = df[mean_agg_columns]

    # aggregate the summation columns
    df_summation = df_summation.groupby(df_summation.index).sum()

    # aggregate the mean columns
    df_mean = df_mean.groupby(df_mean.index).mean()

    # create index range
    index_range = range(0, max_frame + 1)

    if frame_size > 1:
        # calculate the number of bins that would be required 
        # to aggregate the dataframe into the desired frame_size
        num_bins = int(max_frame / frame_size) + 1

        # use pd.cut to group the dataframe by the number of bins
        df_summation = df_summation.groupby(
            pd.cut(df_summation.index, num_bins)
        ).sum()

        df_mean = df_mean.groupby(
            pd.cut(df_mean.index, num_bins)
        ).mean()

        # create a list of starting index for each bin
        bin_ends = [(i + 1)*frame_size - 1 for i in range(num_bins)]
        # check that the last bin ends at the max frame
        if bin_ends[-1] > max_frame:
            bin_ends[-1] = max_frame

        # index both dataframes by the bin starts
        df_summation.index = bin_ends
        df_mean.index = bin_ends

        # create a new index range
        index_range = bin_ends

    # concatenate the two dataframes
    df = pd.concat([df_summation, df_mean], axis=1)

    return df, index_range


def _remove_nan_values(df, fill_type):
    """
    Function to remove nan values from a dataframe.
    For dummy columns fill nan with 0 for other columns use fill_type

    Args:
        df (pandas.DataFrame): The dataframe to clean.

    Returns:
        df (pandas.DataFrame): The cleaned dataframe.
    """

    dummy_prefixes = []
    # create a list of prefixes for dummy columns
    for p in ['p1_', 'p2_']:
        for c in ['upgrade_type_name', 'name', 'unit_type_name']:
            dummy_prefixes.append(p + c)

    # find any columns that start with a dummy prefix
    dummy_columns = [c for c in df.columns if c.startswith(tuple(dummy_prefixes))]

    # if there are dummy columns, fill them with 0
    if dummy_columns:
        df.fillna(0, inplace=True)

    # fill df with fill_type
    df.fillna(method=fill_type, inplace=True)

    return df


def fill_missing_frames_and_agg(
    df, 
    max_frame, 
    frame_size, 
    columns_list,
    fill_type='ffill'
):
    """
    Forward fill missing frames in a dataframe. And aggregate the dataframe by frame_size.

    Args:
        df (pandas.DataFrame): Dataframe to fill.
        max_frame (int): The maximum frame number. This is specified outside 
        the function because it is used to process player1 and 2 in parallel.
        fill_type (str): Type of fill to use.

    Returns:
        pandas.DataFrame: Dataframe with missing frames filled.
    """

    # assert that df is a dataframe
    assert isinstance(df, pd.DataFrame), 'df must be a pandas dataframe'

    # assert that df is indexed by frame
    assert 'frame' in df.index.names, 'df must be indexed by frame'

    # assert that fill_type is a string
    assert isinstance(fill_type, str), 'fill_type must be a string'
        
    # assert that fill_type is either 'ffill' or 'bfill'
    assert fill_type in ['ffill', 'bfill'], \
        f'fill_type must be either ffill or bfill, not {fill_type}'

    # assert that max_frame is an integer
    assert isinstance(max_frame, int), 'max_frame must be an integer'

    # insert frame 0 with all zeros
    df.loc[0] = 0

    # sort the dataframe by frame
    df = df.sort_index()

    # aggregate duplicate frames
    df, index_range = _aggregate_frames(df, max_frame, frame_size, columns_list)

    df = _remove_nan_values(df, fill_type)

    # fill in the missing frames
    df = df.reindex(index_range, method=fill_type)

    # return the dataframe
    return df

def get_clean_events(
    filename, 
    return_df=False, 
    output_dir='data/clean_events',
    columns_list_location='info/raw_columns_list.csv',
    dummies_list_location='info/dummies_list.csv',
    frame_size=1
):
    """
    Get the clean events dataframe for a game from the raw dataframe. Stores the dataframe as a pickle file, using filehash as the filename.

    Args:
        filename (str): The filename (absolute or relative path) of the raw dataframe.
        return_df (bool): Return the dataframe if True, otherwise return nothing.
        output_dir (str): The directory to save the clean events dataframe.
        columns_list_location (str): The location of the raw_columns_list.csv file.
        dummies_list_location (str): The location of the dummies_list.csv file.
        frame_size (int): The number of frames to aggregate.

    Returns:
        pandas.DataFrame: The clean events dataframe if return_df is True.
    """

    # assert that file exists
    assert os.path.exists(filename), f'{filename} does not exist'

    # assert that return_df is a boolean
    assert isinstance(return_df, bool), 'return_df must be a boolean'

    # assert that output_dir is a string and exists
    assert isinstance(output_dir, str), 'output_dir must be a string'
    assert os.path.exists(output_dir), f'{output_dir} does not exist'

    # if dummies_list_location.csv does not exist, create it
    if not os.path.exists(dummies_list_location):
        # create blank dummies_list.csv
        with open(dummies_list_location, 'w') as f:
            pass

    # read the raw dataframe
    if filename.endswith('.pkl') or filename.endswith('.zip'):
        event_df = pd.read_pickle(filename)
    elif filename.endswith('.csv'):
        event_df = pd.read_csv(filename)
    else:
        raise ValueError(f'{filename} is not a valid filetype')

    # set index to frame
    event_df.set_index('frame', inplace=True)

    # get the filehash from the filename
    filehash = filename.split('/')[-1].split('.')[0]

    # get the location of raw_columns_list.csv and read it in
    raw_columns_list_path = columns_list_location

    # assert that the file exists
    assert os.path.exists(raw_columns_list_path), f'{raw_columns_list_path} does not exist'
    raw_columns_list = pd.read_csv(raw_columns_list_path)

    # separate the dataframes into player1 and player2
    player1_df, player2_df = separate_player_dfs(event_df)
    
    # check if either is None
    if player1_df is None or player2_df is None:
        return

    # clean the dataframes
    player1_df = clean_player_events(
        player1_df, 
        raw_columns_list, 
        filehash, 
        dummies_list_location
    )
    player2_df = clean_player_events(
        player2_df, 
        raw_columns_list, 
        filehash,
        dummies_list_location
    )
    # check if either is None
    if player1_df is None or player2_df is None:
        return

    # get max frames from player1 and player2
    max_frames = max(player1_df.index.max(), player2_df.index.max())

    # print warning if max_frames is not an integer or is less than 1
    if not isinstance(max_frames, int) or max_frames < 1:
        print(f'filehash: {filehash} - SKIPPING')
        print('WARNING: max_frames is not an integer or is less than 1')
        return

    # aggregate frames according to settings
    num_frames = frame_size

    # fill missing frames and aggregate
    player1_df = fill_missing_frames_and_agg(
        player1_df, 
        max_frames, 
        num_frames,
        raw_columns_list
    )
    player2_df = fill_missing_frames_and_agg(
        player2_df, 
        max_frames, 
        num_frames,
        raw_columns_list
    )

    # combine the dataframes
    player1_df.columns = ['p1_' + col for col in player1_df.columns]
    player2_df.columns = ['p2_' + col for col in player2_df.columns]

    # concatenate the dataframes
    event_df = pd.concat([player1_df, player2_df], axis=1)

    # define filename as zip for automatic pickle compression
    output_filename = f'{output_dir}/{filehash}.zip'

    # save the dataframe
    event_df.to_pickle(output_filename)

    # return the dataframe
    if return_df:
        return event_df
