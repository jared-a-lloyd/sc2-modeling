import re

class ReplayInfo:

    def __init__(self, replay):
        # self.__Replay = replay
        self.map_hash = replay.map_hash
        self.player_races = self._get_player_races(replay)
        self.filename = replay.filename
        self.player_mmrs = self._get_player_mmrs(replay)
        self.game_length = self._get_game_length(replay)
        self.game_winner = self._get_winner(replay)
        self.timestamp = replay.unix_timestamp
        self.game_type = replay.type
        self.game_speed = replay.speed
        self.fps = replay.game_fps
        self.is_ladder = replay.is_ladder
        self.region = replay.region
        self.highest_league = replay.highest_league


    def _get_game_length(self, replay):

        # this converts to minutes.seconds
        length_string = str(replay.game_length)

        # use regex to extract all numbers from length_string
        minutes = length_string.split('.')[0]
        seconds = length_string.split('.')[1]

        # convert to int in seconds
        return int(minutes)*60 + int(seconds)


    def _get_winner(self, replay):

        winner_string = str(replay.winner)
        if 'Player 1' in winner_string:
            return 1
        elif 'Player 2' in winner_string:
            return 2
        else:
            return 0


    def _get_player_highest_league(self, replay):

        str_a = 'replay.initData.backup'
        str_b = 'user_initial_data'
        str_c = 'highest_league'

        return (
            replay.raw_data[str_a][str_b][0][str_c],
            replay.raw_data[str_a][str_b][1][str_c]
            )


    def _get_player_mmrs(self, replay):

        str_a = 'replay.initData.backup'
        str_b = 'user_initial_data'
        str_c = 'scaled_rating'

        return (
            replay.raw_data[str_a][str_b][0][str_c],
            replay.raw_data[str_a][str_b][1][str_c]
            )


    # get the player races
    def _get_player_races(self, replay):
        """
        _get_player_races
        Iterate through players in self.__Replay.players and extract player
        races as strings from the info

        Returns:
            tuple - Length 2 contain the races of both players
        """

        player_string = []

        for player in replay.players:
            # convert player to string
            player_string.append(str(player))


        return (
            self._get_race(player_string[0]),
            self._get_race(player_string[1])
            )


    def _get_race(self, player):
        """
        _get_race Extract race from string. String is assumed to be of the form:
        'Player x - Race'.

        Args:
            player (str): A string of form 'Player x - Race'

        Returns:
            str: Race of the player
        """

        RACE_LIST = [
            'Protoss',
            'Terran',
            'Zerg'
        ]

        # assert that player is a string
        assert isinstance(player, str), 'player should be a string'


        for race in RACE_LIST:

                if race.lower() in player.lower():

                    # delete race from the player1 string
                    race_string = ' ('+race+')'
                    # assert that race_string is in player
                    assert race_string in player, \
                        f'{player} does not adhere to to {race_string} formatting'
                    # use replace to delete the player race
                    player = player.replace(race_string, '')

                    # create regex to find 'Player 1 - ' leaving only actual name
                    reg_str = r'Player\s\d\s\-\s'

                    # assert that reg_str is in player string
                    assert re.search(reg_str, player), \
                        f'{player} does not adhere to to {reg_str} formatting'

                    return race

