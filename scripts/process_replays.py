"""
This file is used to process replays and extract the data
It is used as a separate script to allow for parallel processing, and for portability.
The ReplayInfo class is required to be located at scripts/classes/ReplayInfo.py for processing

Settings are specified in a scripts/replay_settings.json file. The following settings are available:
        - replay_dir: The directory where replays are located, searches through any child folders as well
        - sample_size: The number of replays to attempt to extract. Any that fail will reduce this amount
        - n_jobs: Number of multiprocessing threads to use. -1 sets the maximum available, less one. Any positive int
                    will be used as is
        - random_seed: The random seed to be used if sampling is selected.
"""
import json
import os
import multiprocessing as mp
import random
import sc2reader
from scripts.classes import ReplayInfo
import pandas as pd
import time
import tqdm

def process_replay(filename):
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
    # use try, except to not derail the entire process if a single file fails
    try:
        replay = sc2reader.load_replay(
            filename,
            load_level=2 # level 2 is all that is required for metadata
            )
    except:
        return None

    try:
        replay_object = ReplayInfo(replay)
    except: # catch exceptions created by ReplayInfo
        return None

    return replay_object

if __name__ == "__main__":

    # suppress warnings when running within Notebooks
    __spec__ = None 
    
    # start timer
    timer = time.time()

    # load settings
    with open("scripts/replay_settings.json", "r") as file:
        settings = json.load(file)

    # get replay directory from settings
    replay_dir = settings["replay_dir"]

    # get sample size from settings
    sample_size = settings["sample_size"]

    # get n_jobs from settings
    n_jobs = settings["n_jobs"]

    # get output_file from settings
    output_file = settings["output_file"]
    if output_file == "":
        output_file = 'data/replays.csv'

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

    # if the user has chosen to take a sample
    if sample_size != -1:
        # take a random sample of replays
        random.seed(random_seed)
        replays_list = random.sample(replays_list, sample_size)

    # set the number of processes to use
    if n_jobs == -1:
        cpu_total = mp.cpu_count() - 1 # cpu_count() returns total number of cores
    else:
        cpu_total = n_jobs

    print(f'Processing {len(replays_list)} replays')

    # set up and run multiprocessing pool
    # imap is used to allow tqdm to calculate a progress bar
    with mp.Pool(processes=cpu_total) as pool:
        replay_collection = list(tqdm(pool.imap(
                process_replay,
                replays_list,
                chunksize=10
            ),
            total=len(replays_list)
        ))

    # remove all None from replay_collection (failed replays)
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
        'region':[x.region for x in replay_collection],
        'player1_highest_league':[
            x.highest_league[0] for x in replay_collection
        ],
        'player2_highest_league':[
            x.highest_league[1] for x in replay_collection
        ],
        'filehash':[x.filehash for x in replay_collection]
    })

    # remove rows with duplicate filehashes
    replay_df = replay_df.drop_duplicates(subset='filehash')

    # write replay_collection to csv with no index
    replay_df.to_csv(output_file, index=False)

    # found x valid replays
    print(f'Found {replay_df.shape[0]} unique valid replays')
    # print time elapsed as HH:MM:SS
    print(f'Time elapsed: {time.strftime("%H:%M:%S", time.gmtime(time.time() - timer))}')
