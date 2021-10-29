# SC2 Keyboard Hotkey Modeling

This project is an attempt to create a predictor for the outcome of a StarCraft 2 (SC2) game. The idea is to be able to predict the outcome of the game as early as possible based only on the events within the game.

SC2 replays are stored as `.SC2Replay` files. These files do not contain any information about the game state at any point, but rather only the events that occurred during the game. This means that the game state is not known at any point. The model will attempt to use the occurrence and sequence of events to predict the outcome of the game. The game will divided up into manageable chunks of time, and the model will attempt to predict the outcome of each chunk of time, with information from all previous chunks being included.

It is hoped that a secondary outcome of the modeling process will be an prediction of victory at time = 0, which may be able to give information about the descrepancies in balance between different races, or maps that favor a certain race.

Data are extracted using the `sc2reader` python package, which can be install by PiP, or found __[here](https://github.com/ggtracker/sc2reader)__. In the raw form, `sc2reader` provides events at each frame of the game (a frame is about a 16th of an in-game second - and an in-game second is $1/1.2 \approx 0.83$ real-time seconds).

Replays were downloaded from __[SpawningTool](https://lotv.spawningtool.com)__.

The project is constructed as a series of Jupyter notebooks as part of my Brainstation capstone project. I hope to be able to link all the notebooks to simulate a workflow from one notebook to the next.

## Contents
* `web_scraping/`
    * `download_spawning.ipynb` - Automating the download of replay files as .zip.
    * `extract_and_map.ipynb` - Creating a json map of the files that are downloaded, and unzipping all .zip to their own folders.
* `get_replay_data/`
    * `explore_data_structure.ipynb` - Unpacking an individual replay into a json dict and mapping out the generalised layout of data within the replay using `sc2reader`. Uses multithreading to speed up the process.
* `info/`
    * `A tutorial on sc2reader_ events and units _ MGD’s blog.pdf' - A basic tutorial on interfacing with `sc2reader` found online
    * `event_attribute_dict.json` - A json map of the layout of data within the events section of a `sc2reader` output.
    * `replay_json.json` - The general layout of an `sc2reader` object from the top level down.
    * `overall_process_flow.drawio` - A map of the process that is used from data download to modeling.