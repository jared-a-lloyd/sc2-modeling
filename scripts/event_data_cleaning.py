"""
This file contains a function that will be called to clean event data stored in a csv or pkl file and create a dataframe from it.
"""

def clean_event_data(event_data_file):
    """
    This function receives a filename (absolute path) to a csv or pkl file 
    containing event data and cleans it. A dataframe is then returned.

    Args:
        event_data_file (str): Absolute path to the event data file

    Returns:
        pandas.DataFrame: A dataframe containing the cleaned event data
    """
    import pandas as pd
    import os

    # assert that a file exists at the given path
    assert os.path.exists(event_data_file), f'File {event_data_file} does not exist.'

    # if the file is a csv, read it as a dataframe
    if event_data_file.endswith('.csv'):
        df = pd.read_csv(event_data_file)
    else:
        df = pd.read_pickle(event_data_file)

    # set the index to frame number
    df.set_index('frame', inplace=True)

    # get the total frame range of the replay (i.e. the number of frames)
    frame_range = range(max(df.index)+1)

    # list of necessary events that should be kept
    necessary_events = [
        'TargetPointCommandEvent',
        'UnitBornEvent',
        'PlayerStatsEvent',
        'UnitDiedEvent',
        'UnitInitEvent',
        'UnitDoneEvent',
        'TargetUnitCommandEvent',
        'BasicCommandEvent',
        'UpgradeCompleteEvent'
    ]

    # remove all rows where 'name' is in the unnecessary_events list
    df = df[df['name'].isin(necessary_events)]

    # list of unnecessary columns
    unnecessary_columns = [
        'count',
        'second',
        'killer_pid',
        'ff_minerals_lost_army',
        'ff_minerals_lost_economy',
        'ff_minerals_lost_technology',
        'ff_vespene_lost_army',
        'ff_vespene_lost_economy',
        'ff_vespene_lost_technology'
    ]

    # drop unnecessary columns if they exist (i.e. ignore errors)
    df.drop(columns=unnecessary_columns, inplace=True, errors='ignore')

    # player sprays are considered "upgrades" these are not needed
    unnecessary_upgrades = set(
        [x for x in df['upgrade_type_name'] if 'spray' in str(x).lower()]
    )

    # drop unnecessary upgrades
    df = df[~df['upgrade_type_name'].isin(unnecessary_upgrades)]

    # create dummies of event name and upgrade type name
    df = pd.get_dummies(df, columns=['name', 'upgrade_type_name'])


    ## <<<< CLEANING PLAYER DATA >>>> ##
    # create a separate df for each player
    player1_df = df[df['pid'] == 1].copy()
    player2_df = df[df['pid'] == 2].copy()

    # drop pid column from new dataframes
    player1_df.drop(columns=['pid'], inplace=True)
    player2_df.drop(columns=['pid'], inplace=True)

    # drop 'unit_type_name' column from player1_df and player2_df 
    # because all relevant data will be in the df dataframe
    player1_df.drop(columns=['unit_type_name'], inplace=True)
    player2_df.drop(columns=['unit_type_name'], inplace=True)

    # rename columns to player_1_ and player_2_
    player1_df.columns = ['player_1_' + x for x in player1_df.columns]
    player2_df.columns = ['player_2_' + x for x in player2_df.columns]

    # drop duplicate rows
    player1_df.drop_duplicates(inplace=True)
    player2_df.drop_duplicates(inplace=True)

    # for all missing value rows in player1_df and player2_df, fill with above row
    player1_df.fillna(method='ffill', inplace=True)
    player2_df.fillna(method='ffill', inplace=True)

    # drop duplicate rows again (ffill may have created duplicates)
    player1_df.drop_duplicates(inplace=True)
    player2_df.drop_duplicates(inplace=True)

    ## <<<< CLEANING GENERAL DATA >>>> ##
    # we now have 3 dataframes: player1_df, player2_df, and df
    # drop all the columns in df that are emptied into player1_df and player2_df

    # get all numeric columns in df
    numeric_columns = [x for x in df.columns if df[x].dtype != 'object']

    # get all numeric columns where sum is 0 in df
    zero_columns = [x for x in numeric_columns if df[x].sum() == 0]

    # drop all columns in df that are in zero_columns (i.e. empty)
    df.drop(columns=zero_columns, inplace=True)

    # create dummies of unit type name from df
    # first drop unnecessary unit type names (i.e. those with 'beacon' or 'nan')
    unique_units = df['unit_type_name'].unique()
    unnecessary_units =[]
    for unit in unique_units:
        if ('beacon' in str(unit).lower()) or ('nan' in str(unit).lower()):
            unnecessary_units.append(unit)

    # drop all rows in df where 'unit_type_name' is in unnecessary_units
    df = df[~df['unit_type_name'].isin(unnecessary_units)]

    # dummify unit_type_name
    df = pd.get_dummies(df, columns=['unit_type_name'])

    ## <<<< COMBINING DATA FRAMES >>>> ##
    # make new df with index of frame_range
    new_df = pd.DataFrame(index=frame_range)

    # add all dfs to new_df
    new_df = new_df.join(player1_df)
    new_df = new_df.join(player2_df)
    new_df = new_df.join(df)

    # drop all rows with index = 0
    new_df.drop(index=0, inplace=True)

    return new_df

    # insert single row with index = 0 with all 0 values
    new_df.loc[0] = 0

    # for each column get first non-nan value
    for col in new_df.columns:
        first_non_nan = new_df[col].first_valid_index()
        new_df.loc[0, col] = new_df.loc[first_non_nan, col]
    
    # sort new_df by index
    new_df.sort_index(inplace=True)

    # get all columns from player1_df and player2_df
    player_columns_mask = player1_df.columns.tolist() \
    + player2_df.columns.tolist()

    # forward fill all new_df mask columns 
    new_df[player_columns_mask] = new_df[
        player_columns_mask
    ].fillna(method='ffill').copy()

    # fillna with 0 for all other columns
    new_df.fillna(0, inplace=True)

    # return new_df
    return new_df
