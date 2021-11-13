"""
Functions that require separate modules (generally for multiprocessing) are ..found here

Currently, these are:
    - get_all_events - load a replay using sc2reader and run _get_event_data on each event

    - _get_event_data - extract the data from an event and return it as a dictionary
"""

# function to extract all attribute value from a single replay event
# create function to get the event data so that multiprocessing can be used
def _get_event_data(event):
    """
    get_event_data
    Extracts the data from the event and returns it as a dictionary
    Ignores events that start with '_', i.e., special attributes and dunder types

    Args:
        event (sc2reader.event): Event object extracted from sc2reader.events

    Returns:
        [dict]: A dictionary containing the event data
    """
    # ignore attributes that are not needed (special or dunder)
    event_attributes = [attr for attr in dir(event) if not attr.startswith('_')]

    # initialize a dictionary to store the values of each attribute
    d = dict()

    # loop through each attribute and store the value in the dictionary
    for attr in event_attributes:
        # ignore attributes if they do not contain a value type
        if type(getattr(event, attr)) in [int, float, str, bool]:
            d[attr] = getattr(event, attr)

    return d

# function to extract all events using get_event_data
def get_all_events(filename, output_dir='data/events'):
    """
    ____________________________________________________________________________

    Extracts all events from a replay file and stores them in a pickle file

    Returns the dataframe of the extracted events

    Args:
        filename (string): Absolute path to the replay file
        output_dir (str, optional): Directory where output csv should be stored. Defaults to 'data/events'.

    Returns:
        pandas.DataFrame: A dataframe containing all events from the replay file
    """
    import sc2reader
    import os
    import pandas as pd

    # use sc2reader to extract replay data, load_level=4
    replay = sc2reader.load_replay(filename, load_level=4)

    # get events as a list from replay object
    events = replay.events

    # loop through each event and extract the data
    event_data = [_get_event_data(event) for event in events]

    # convert event_data to a dataframe
    df = pd.DataFrame(event_data)

    # get the list of columns to be kept
    columns_checklist = pd.read_csv(
        'C:/Users/jared/Gits/sc2-modeling/info/raw_columns_list.csv'
    )
    column_keep_df = columns_checklist.loc[
        columns_checklist['include_bool'] == 1
    ].drop(['include_bool'], axis=1)

    # create a list of columns to be kept
    columns_to_keep = column_keep_df['column_name'].tolist()
    columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    df = df[columns_to_keep]

    # remove all rows where pid not NaN and pid = 1 or 2
    df = df.loc[~(
            (df['pid'].notnull())
            & (df['pid'] != 1)
            & (df['pid'] != 2)
        )
    ]

    # name replay csv with the filehash of the replay
    output_name = replay.filehash + '.pkl'
    output_path = os.path.join(output_dir, output_name)

    # save the dataframe to a csv file
    df.to_pickle(output_path)

    return df