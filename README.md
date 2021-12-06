# SC2 Game Prediction Model

This project is an attempt to create a predictor for the outcome of a StarCraft 2 (SC2) game. The idea is to be able to predict the outcome of the game as early as possible based only on the events within the game.

SC2 replays are stored as `.SC2Replay` files. These files do not contain any information about the game state at any point, but rather only the events that occurred during the game. This means that the game state is not known at any point. The model will attempt to use the occurrence and sequence of events to predict the outcome of the game. The game will divided up into manageable chunks of time, and the model will attempt to predict the outcome of each chunk of time, with information from all previous chunks being included.

It is hoped that a secondary outcome of the modeling process will be an prediction of victory at time = 0, which may be able to give information about the discrepancies in balance between different races, or maps that favor a certain race.

Data are extracted using the `sc2reader` python package, which can be install by PiP, or found __[here](https://github.com/ggtracker/sc2reader)__. In the raw form, `sc2reader` provides events at each frame of the game (a frame is about a 16th of an in-game second - and an in-game second is 1&divide;1.4 &asymp; 0.714 real-time seconds).

Replays were downloaded from __[SpawningTool](https://lotv.spawningtool.com/replays/)__ and __[Blizzard](https://github.com/Blizzard/s2client-proto#downloads)__.

The project is constructed as a series of Jupyter notebooks as part of my Brainstation capstone project.

## Contents
* `Notebook 1 - Data Collection.ipynb`
* `Notebook 2 - Metadata Analysis and Modeling.ipynb`
* `Notebook 3 - Event Data Analysis and Modeling.ipynb`
* `Technical Report.pdf` - A summary of the work done, and findings of this project.
* `env/` - A folder containing the yml files which can be used to generate the environments for the project.
* `img/` - A folder containing the images used in the project.
* `info/` - General information used in the project
    * `clean_master_columns_list.csv` - A list of all unique columns in all cleaned event data files. Used to ensure all model inputs are identical.
    * `raw_columns_list.csv` - A csv detailing how each column of the dataframe output by `process_events.py` is handled.
* `scripts/` - A collection of scripts used to generate the data.
    * `calculate_game_length.py` - A script to calculate the length of a game from the number of frames in the replay.
    * `event_data_cleaning.py` - Clean event data and store as .zip pickles.
    * `filter_replays.py` - Filter replays based on game length, and other Metadata.
    * `get_clean_events.py` - Read pickle or csv files of extracted event data and clean it. Storing as .zip pickles.
    * `process_events.py` - Use `sc2reader` to process replays and extract and store event data.
    * `process_replays.py` - Use `sc2reader` to process replays and extract and store Metdata.
    * `reduce_mem_usage.py` - Reduce memory usage of a pandas dataframe. (Currently unused)
    * `repay_settings.json` - Settings for `process_replays.py`.
    * `classes/` - A collection of classes used in this project
        * `BatchGenerator.py` - A class to handle reading npy files and generating batches of data for a Keras model.
        * `ReplayInfo.py` - A class to extract replay metadata from an `sc2reader` replay object.
* `.gitignore` - A file containing a list of files to ignore when committing to git.
* `README.md` - A file containing a summary of the project.

