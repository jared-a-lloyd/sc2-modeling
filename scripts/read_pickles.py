import pandas as pd

def read_pickles(directory, filehash, metadata_df):
    """
    Reads in pickles from the directory.

    Args:
        directory (str): the directory from which to read the pickles
        filehash (string): list of filehashes to read in
        metadata_df (df): the metadata df

    Returns:
        pandas.Dataframe: the dataframe of the pickles
    """

    # get the filepath
    filepath = directory+filehash+'.zip'

    # if the file exists
    if os.path.exists(filepath):
        # read df from pickle
        df = pd.read_pickle(filepath)
    else:
        return None

    # check that df has no nans
    if df.isnull().values.any():
        return None

    # write the filehash to the df
    df['filehash'] = filehash

    # look up the winner in the metadata df
    winner = metadata_df.loc[
        metadata_df['filehash'] == filehash,
        'game_winner'
    ].values[0]

    # write the winner to the df
    if winner == 1:
        df['winner'] = 1
    else:
        df['winner'] = 0

    # reset the index
    df.reset_index(inplace=True)

    # rename the index to 'frame'
    df.rename(columns={'index': 'frame'}, inplace=True)

    # return the df
    return df
