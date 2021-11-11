# this file is used to process replays and extract the data
# it is used as a separate script to allow for parallel processing
# settings can be found in replay_settings.json
import json
import os
import sys
import multiprocessing as mp
import random
import sc2reader
from ..classes import ReplayInfo
import pandas as pd

def process_replay(filename, filters=None):
    """
    process_replay
    ___________________________________________________________________________
    Processes a replay file and returns a ReplayInfo object if the replay is valid according to the filters. Else returns None.

    Args:
        filename (string): Absolute path to the replay file.
        filters (dict): A dictionary of filters to apply to the replay. Dictionary keys are the attributes of ReplayInfo, and values are those that should be excluded.

    Returns:
        [type]: [description]
    """
    # load replay
    try:
        replay = sc2reader.load_replay(
            filename,
            load_level=2 # level 2 is all that is required for metadata
            )
    except: # catch exceptions created by sc2reader
        return None

    replay_object = ReplayInfo(replay)

    # TODO: check if replay is valid using filters from settings

    return replay_object

if __name__ == "__main__":
    # load settings
    with open("replay_settings.json", "r") as f:
        settings = json.load(f)

    # get replay directory from settings
    replay_dir = settings["replay_dir"]

    # get sample size from settings
    sample_size = settings["sample_size"]

    # get random seed from settings
    # check if random_seed key exists
    if "random_seed" in settings:
        random_seed = settings["random_seed"]
    else:
        random_seed = None

    replays_list = []
    # loop through replay directory and get list of .SC2Replay files
    for dirpath, dirnames, filenames in os.walk(replay_dir):
        for filename in filenames:
            if filename.endswith('.SC2Replay'):
                filepath = os.path.join(dirpath, filename)
                replays_list.append(filepath)

    # take a random sample of replays
    random.seed(random_seed)
    replays_list = random.sample(replays_list, sample_size)

    # process replays
    with mp.Pool(mp.cpu_count()) as pool:
        replay_collection = pool.map(
            process_replay,
            replays_list,
            [settings["filters"]]*len(replays_list))

    # remove all None from replay_collection
    replay_collection = [x for x in replay_collection if x is not None]

    # convert replay collection to dataframe
    replay_df = pd.DataFrame({
        'filename':[x.filename for x in replay_collection],
        'map':[x.map_hash for x in replay_collection],
        'player1_race':[x.player_races[0] for x in replay_collection],
        'player2_race':[x.player_races[1] for x in replay_collection],
        'player1_mmr':[x.player_mmrs[0] for x in replay_collection],
        'player2_mmr':[x.player_mmrs[1] for x in replay_collection],
        'game_length':[x.game_length for x in replay_collection],
        'game_type':[x.game_type for x in replay_collection],
        'game_speed':[x.game_speed for x in replay_collection],
        'game_winner':[x.game_winner for x in replay_collection],
        'timestamp':[x.timestamp for x in replay_collection],
        'fps':[x.fps for x in replay_collection],
        'is_ladder':[x.is_ladder for x in replay_collection],
        'region':[x.region for x in replay_collection]
    })

    # write replay_collection to csv with no index
    replay_df.to_csv('../data/replays.csv', index=False)

    print(f"Replays processed = {replay_df.shape[0]}.")
