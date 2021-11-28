"""
This script contains a function that processes each replay, extracts the maximum number of frames, and calculates the game length. The game length is then stored in a dict, indexed by the filehash to allow for ease of updating the Metadata `csv` file.
"""


def fetch_calculate_game_length(filepath):
    """
    Function to calculate the game length of a replay and store it in a dict.

    Args:
        filepath (str): filepath to the replay

    Returns:
        dict: {filehash: game length}
    """
    import sc2reader

    # read the replay
    replay = sc2reader.load_replay(filepath)

    # get the filehash
    filehash = replay.filehash

    # iterate backwards through the events to find the last frame event
    for event in reversed(replay.events):
        try:
            frame = event.frame
        except:
            continue
            
        if (isinstance(frame, int)) & (frame > 0):
            #break out of the loop
            break

    # calculate the game length
    game_speed_factor = 1.4
    game_length = round(frame/(16*game_speed_factor), 0)

    # return the filehash and game length
    return {filehash: game_length}
