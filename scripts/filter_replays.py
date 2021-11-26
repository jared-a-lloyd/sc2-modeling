"""
This script contains the function filter_replays that can be used to filter replays in a dataframe according to a specified (or default) set of criteria.

A number of supporting functions are also included, these can be identified by an underscore at the beginning of the function name.
"""
import pandas as pd
import numpy as np

def _check_filter_inputs(filter_dict):
    """
    check_filter_inputs
    Check the input filters to ensure they are all in the correct format.

    Args:
        filter_dict (dict): A dictionary containing the filter criteria

    Returns:
        None
    """

    # assert that 'df' is a dataframe
    assert isinstance(filter_dict['df'], pd.DataFrame), \
        'df should be a dataframe'
    
    # assert that all bool values are True or False
    bool_keys = [
        'check_game_length',
        'check_player_mmr',
        'check_player_race',
        'check_1v1',
        'check_region',
        'check_fps',
        'check_is_ladder',
        'check_game_speed'
    ]
    for k in bool_keys:
        assert isinstance(filter_dict[k], bool), \
            f"{k} should be a boolean (True or False)"

    # assert that valid regions is a list of strings
    assert isinstance(filter_dict['valid_regions'], list), \
        'valid_regions should be a list of strings'
    for v in filter_dict['valid_regions']:
        assert isinstance(v, str), \
            'valid_regions should be a list of strings'

    # assert that numeric values are positive ints
    numeric_keys = [
        'min_game_length',
        'max_game_length',
        'min_player_mmr',
        'max_player_mmr'
    ]
    for k in numeric_keys:
        assert isinstance(filter_dict[k], int), \
            f"{k} should be a positive integer"
        assert filter_dict[k] > 0, \
            f"{k} should be a positive integer"


def _check_game_length(df, min_game_length, max_game_length):
    """
    check_game_length
    Check the game length of a replay.

    Args:
        df (pandas.DataFrame): A dataframe containing replay data
        min_game_length (int): The minimum game length
        max_game_length (int): The maximum game length

    Returns:
        pandas.DataFrame: A filtered dataframe
    """

    # get the length of the dataframe
    original_length = df.shape[0]

    # remove rows where the game length is outside the specified range
    df = df[
        (df["game_length"] >= min_game_length) &
        (df["game_length"] <= max_game_length)
    ]

    # print the number of replays that were filtered
    print(f"Game Length: {original_length - df.shape[0]} replays were filtered")

    return df


def _check_player_mmr(df, min_player_mmr, max_player_mmr):
    """
    check_player_mmr
    Check the MMR of both players is valid.

    Args:
        df (pandas.DataFrame): A dataframe containing replay data
        min_player_mmr (int): The minimum player MMR
        max_player_mmr (int): The maximum player MMR

    Returns:
        pandas.DataFrame: A filtered dataframe
    """

    # get the length of the dataframe
    original_length = df.shape[0]

    # remove rows where the player MMR is outside the specified range
    df = df[
        (df["player1_mmr"] >= min_player_mmr) &
        (df["player1_mmr"] <= max_player_mmr) &
        (df["player2_mmr"] >= min_player_mmr) &
        (df["player2_mmr"] <= max_player_mmr)
    ]

    # print the number of replays that were filtered
    print(f"Player MMR: {original_length - df.shape[0]} replays were filtered")

    return df


def _check_player_races(df):
    """
    _check_player_races
    Check that both players are either Protoss, Terran or Zerg

    Args:
        df (pandas.DataFrame): A dataframe containing replay data

    Returns:
        pandas.DataFrame: A filtered dataframe
    """

    # get the length of the dataframe
    original_length = df.shape[0]

    # define valid races
    valid_races = [
        'Protoss',
        'Terran',
        'Zerg'
    ]

    # check that both player1_race and player2_race are in valid_races
    df = df[df['player1_race'].isin(valid_races)]
    df = df[df['player2_race'].isin(valid_races)]

    # print the number of replays that were filtered
    print(f"Player Race: {original_length - df.shape[0]} replays were filtered")

    return df


def _check_region(df, regions):
    """
    _check_region
    Check that both players are either Protoss, Terran or Zerg

    Args:
        df (pandas.DataFrame): A dataframe containing replay data
        regions (list): A list of regions to check for

    Returns:
        pandas.DataFrame: A filtered dataframe
    """

    # get the length of the dataframe
    original_length = df.shape[0]

    # check that region is in regions
    df = df[df['region'].isin(regions)]

    # print the number of replays that were filtered
    print(f"Region: {original_length - df.shape[0]} replays were filtered")

    return df


def _check_other_booleans(df, bools):
    """
    _check_other_boolean
    Check relevant values for all other "check_" options. For each option a relevant value is created to be checked for.

    Args:
        bools (list): A list of booleans to check

    Returns:
        pandas.DataFrame: A filtered dataframe
    """

    # construct a list of tuples for each boolean
    # tuple contains:
    #   - the name of the boolean
    #   - relevant column in the df
    #   - the relevant value to check against
    bool_checks = [
        ('check_1v1', 'game_type', '1v1'),
        ('check_fps', 'fps', 16.0),
        ('check_is_ladder', 'is_ladder', True),
        ('check_game_speed', 'game_speed', 'Faster'),
    ]

    for i, check in enumerate(bool_checks):
        # get the length of the dataframe
        original_length = df.shape[0]

        # check if bools[i] is True
        if bools[i]:
            # filter the dataframe based on the relevant column and value
            col = check[1]
            val = check[2]
            df = df[df[col] == val]

            # print the number of replays that were filtered
            print(f"{check[0].capitalize()}: {original_length - df.shape[0]} replays were filtered")

    return df


def filter_replays(
    df,
    check_game_length=True,
    min_game_length=int(5*60),
    max_game_length=int(45*60),
    check_player_mmr=True,
    min_player_mmr=2000,
    max_player_mmr=8000,
    check_player_races=True,
    check_1v1=True,
    check_region=True,
    valid_regions=['eu', 'us', 'kr', 'cn'],
    check_fps=True,
    check_is_ladder=True,
    check_game_speed=True
):
    """
    filter_replays
    Filter replays based on the specified options. Prints the number of replays that were filtered at each step, and returns a filtered dataframe.

    Args:
        df (pandas.Dataframe): The dataframe containing replay data to filter
        check_game_length (bool, optional): Whether game length should be checked.
        If True, min_game_length and max_game_length are used. Defaults to True.
        min_game_length (int, optional): Minimum game length in seconds. 
        Defaults to int(5*60) or 5 minutes.
        max_game_length (int, optional): Maximum game length in seconds. 
        Defaults to int(45*60) or 45 minutes.
        check_player_mmr (bool, optional): Whether player MMR should be checked.
        If True, min_player_mmr and max_player_mmr are used. Defaults to True.
        min_player_mmr (int, optional): Minimum valid player MMR. Defaults to 2000.
        max_player_mmr (int, optional): Maximum valid player MMR. Defaults to 8000.
        check_player_races (bool, optional): Whether player races should be checked.
        If True player races should be "Protoss", "Terran" or "Zerg". Defaults to True.
        check_1v1 (bool, optional): Whether game_type should be 1v1. Defaults to True.
        check_region (bool, optional): Whether game_region should be checked. 
        If True, valid_regions is used. Defaults to True.
        valid_regions (list, optional): List of regions that will be considered valid. 
        Defaults to ['eu', 'us', 'kr', 'cn'].
        check_fps (bool, optional): If True only replays with FPS of 16 will be returned.
        Defaults to True.
        check_is_ladder (bool, optional): If True only replays with is_ladder = True are returned. 
        Defaults to True.
        check_game_speed (bool, optional): If true only replays where game_speed is "faster are returned.
        Defaults to True.

    Returns:
        pandas.DataFrame: The filtered dataframe
    """    
    
    # add all arguments to the filter_dict for validation
    filter_dict = {
        'df': df,
        'check_game_length': check_game_length,
        'min_game_length': min_game_length,
        'max_game_length': max_game_length,
        'check_player_mmr': check_player_mmr,
        'min_player_mmr': min_player_mmr,
        'max_player_mmr': max_player_mmr,
        'check_player_race': check_player_races,
        'check_1v1': check_1v1,
        'check_region': check_region,
        'valid_regions': valid_regions,
        'check_fps': check_fps,
        'check_is_ladder': check_is_ladder,
        'check_game_speed': check_game_speed
    }

    # run asserts to ensure that the filter_dict is valid
    _check_filter_inputs(filter_dict)

    # print the length of the dataframe before filtering
    print(f"Length of dataframe before filtering: {df.shape[0]}")

    # filter by game length if check_game_length is True
    if check_game_length:
        df = _check_game_length(df, min_game_length, max_game_length)

    # filter by player mmr if check_player_mmr is True
    if check_player_mmr:
        df = _check_player_mmr(df, min_player_mmr, max_player_mmr)

    # filter by player race if check_player_race is True
    if check_player_races:
        df = _check_player_races(df)

    # filter by region if check_region is True
    if check_region:
        df = _check_region(df, valid_regions)

    # filter by other booleans
    df = _check_other_booleans(
        df,
        [
            check_1v1, 
            check_fps, 
            check_is_ladder, 
            check_game_speed
        ]
    )

    return df
