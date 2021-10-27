# SC2 Keyboard Hotkey Modeling

This project is an attempt to use available resources to extract and map the most common actions that users take while playing StarCraft 2. The goal is an attempt to automate improving of hotkey layout for the most ergonomic layout (i.e., least average travel time for the fingers)

The initial attempt is using the `sc2reader` python package, which can be install by PiP, or found __[here](https://github.com/ggtracker/sc2reader)__ to extract all events from a replay. Events will then be binned according to type (keyboard, mouse, no user interaction), and the keyboard related actions mapped to their default hotkeys. Upon analyzing all replays available the most common hotkey combinations (overall and by race) will be used to attempt rearranging the default hotkey setup assuming that the 'home keys' are WASD, as is the case for most games.

Ultimately, this will be an attempt to generalize the means of observing games in order to automatically determine optimal hotkey layouts. But the potential would exist for this ability to extend to other programs such as Excel or IDEs - where hotkey use is prefered.

Replays were downloaded from __[SpawningTool](https://lotv.spawningtool.com)__.

The project is constructed as a series of Jupyter notebooks as part of my Brainstation capstone project.

## Contents
* `web_scraping/`
    * `download_spawning.ipynb` - Automating the download of replay files as .zip.
    * `extract_and_map.ipynb` - Creating a json map of the files that are downloaded, and unzipping all .zip to their own folders.
* `get_replay_data/`
    * `explore_data_structure.ipynb` - Unpacking an individual replay into a json dict and mapping out the generalised layout of data within the replay using `sc2reader`.
* `info/`
    * `A tutorial on sc2reader_ events and units _ MGD’s blog.pdf' - A basic tutorial on interfacing with `sc2reader` found online
    * `event_attribute_dict.json` - A json map of the layout of data within the events section of a `sc2reader` output.
    * `replay_json.json` - The general layout of an `sc2reader` object from the top level down.
    * `overall_process_flow.drawio` - A map of the process that is used from data download to modeling.