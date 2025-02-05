# -*- coding: utf-8 -*-
"""Implement no limit texas holdem Round class"""
from enum import Enum

from rlcard.games.limitholdem import PlayerStatus

class Action(Enum):
    # DEFAULT ACTIONS
    FOLD = 0
    CHECK = 1
    CALL = 2
    ALL_IN = 3
    # BET ACTIONS
    BET_HALF_POT = 4
    BET_POT = 5
    BET_2POT = 6
    BET_5POT = 7
    # RAISE ACTIONS
    RAISE_2X = 8
    RAISE_3X = 9
    RAISE_5X = 10

default_actions_set = {
    Action.FOLD, 
    Action.CHECK, 
    Action.CALL, 
    Action.ALL_IN
}

bet_actions_set = {
    Action.BET_HALF_POT, 
    Action.BET_POT, 
    Action.BET_2POT,
    Action.BET_5POT
}

bet_actions_multipliers = {
    Action.BET_HALF_POT: 0.5, 
    Action.BET_POT: 1, 
    Action.BET_2POT: 2,
    Action.BET_5POT: 5
}

raise_actions_set = {
    Action.RAISE_2X,
    Action.RAISE_3X,
    Action.RAISE_5X
}

raise_actions_multipliers = {
    Action.RAISE_2X: 2,
    Action.RAISE_3X: 3,
    Action.RAISE_5X: 5
}

class NolimitholdemRound:
    """Round can call functions from other classes to keep the game running"""

    def __init__(self, num_players, init_raise_amount, dealer, np_random, disabled_actions=None):
        """
        Initialize the round class

        Args:
            num_players (int): The number of players
            init_raise_amount (int): The min raise amount when every round starts
        """
        self.np_random = np_random
        self.game_pointer = None
        self.num_players = num_players
        self.init_raise_amount = init_raise_amount

        # Exclude these actions from the set of legal actions
        self.disabled_actions = disabled_actions if disabled_actions is not None else set()

        self.dealer = dealer

        # Count the number without raise
        # If every player agree to not raise, the round is over
        self.not_raise_num = 0

        # Count players that are not playing anymore (folded or all-in)
        self.not_playing_num = 0

        # Raised amount for each player
        self.raised = [0 for _ in range(self.num_players)]

    def start_new_round(self, game_pointer, raised=None):
        """
        Start a new bidding round

        Args:
            game_pointer (int): The game_pointer that indicates the next player
            raised (list): Initialize the chips for each player

        Note: For the first round of the game, we need to setup the big/small blind
        """
        self.game_pointer = game_pointer
        self.not_raise_num = 0
        if raised:
            self.raised = raised
        else:
            self.raised = [0 for _ in range(self.num_players)]

    def proceed_round(self, players, action):
        """
        Call functions from other classes to keep one round running

        Args:
            players (list): The list of players that play the game
            action (str/int): An legal action taken by the player

        Returns:
            (int): The game_pointer that indicates the next player
        """
        #
        # Acting player
        player = players[self.game_pointer]
        
        #
        # Price for the player to continue to the next round
        diff = max(self.raised) - self.raised[self.game_pointer]
        
        #
        # Proceed depending on the given action type
        #
        # --> Base Actions
        # 
        # Action.FOLD = 0
        if action == Action.FOLD:
            player.status = PlayerStatus.FOLDED
        #
        # Action.CHECK = 1
        elif action == Action.CHECK:
            assert diff == 0, "Can't check when facing a bet"
            self.not_raise_num += 1
        #
        # Action.CALL = 2
        elif action == Action.CALL:
            assert diff != 0, "Can't call when not facing a bet"
            self.raised[self.game_pointer] = max(self.raised)
            player.bet(chips=diff)
            self.not_raise_num += 1
        #
        # Action.ALL_IN = 3
        elif action == Action.ALL_IN:
            all_in_quantity = player.remained_chips
            self.raised[self.game_pointer] += all_in_quantity
            player.bet(chips=all_in_quantity)
            self.not_raise_num = 1
        #
        # --> Bet Actions
        #
        elif action in bet_actions_set:
            assert diff == 0, "Can't bet when already facing a bet"
            bet_size = int(bet_actions_multipliers[action] * self.dealer.pot)
            self.raised[self.game_pointer] = bet_size
            player.bet(chips=bet_size)
            self.not_raise_num = 1
        #
        # --> Raise Actions
        #
        elif action in raise_actions_set:
            assert diff != 0, "Can't raise when not facing a bet"
            bet_size = int(raise_actions_multipliers[action] * max(self.raised)) - self.raised[self.game_pointer]
            self.raised[self.game_pointer] += bet_size
            player.bet(chips=bet_size)
            self.not_raise_num = 1
        #
        # Error condition
        else:
            raise ValueError(f'Action not recognized: {action}')

        #
        # Check player chips
        if player.remained_chips < 0:
            raise Exception("Player in negative stake")

        #
        # Update player's status if they went all-in
        if player.remained_chips == 0 and player.status != PlayerStatus.FOLDED:
            player.status = PlayerStatus.ALLIN

        #
        # Update counts based on player status
        if player.status == PlayerStatus.ALLIN:
            self.not_playing_num += 1
            self.not_raise_num -= 1  # Because already counted in not_playing_num
        elif player.status == PlayerStatus.FOLDED:
            self.not_playing_num += 1

        #
        # Advance game pointer to the next player, skipping folded players
        self.game_pointer = (self.game_pointer + 1) % self.num_players
        while players[self.game_pointer].status == PlayerStatus.FOLDED:
            self.game_pointer = (self.game_pointer + 1) % self.num_players

        return self.game_pointer

    def get_nolimit_legal_actions(self, players):
        """
        Obtain the legal actions for the current player

        Args:
            players (list): The players in the game

        Returns:
           (list):  A list of legal actions
        """
        #
        # Start with a list of all actions
        #
        legal_actions = set()

        #
        # Get the acting player's id
        #
        player = players[self.game_pointer]

        #
        # Get the bet size the player is facing
        #
        diff = max(self.raised) - self.raised[self.game_pointer]

        #
        # Case 1 - No bets have been made this round
        #
        if (#all(r == 0 for r in self.raised) or
            diff == 0 # Special case - the big blind can check their option to close the preflop round
        ):
            #
            # Add legal default actions
            legal_actions.add(Action.CHECK)
            legal_actions.add(Action.ALL_IN)
            #
            # Add legal bet actions
            for action, multiplier in bet_actions_multipliers.items():
                if (player.remained_chips > multiplier * self.dealer.pot): # Player has enough chips
                    legal_actions.add(action)
        #
        # Case 2 - Bets have been made this round
        #
        else:
            legal_actions.add(Action.FOLD)   # The player can always fold
            legal_actions.add(Action.CALL)   # The player can always call
            if player.remained_chips > diff: # The player can go all-in if they have enough chips
                legal_actions.add(Action.ALL_IN)
            #
            # Add legal raise actions
            for action, multiplier in raise_actions_multipliers.items():
                if (player.remained_chips > multiplier * diff): # Player has enough chips
                    legal_actions.add(action)

        #
        # Remove disabled actions
        #
        legal_actions -= self.disabled_actions

        return sorted(list(legal_actions), key=lambda a: a.value) # Legacy functions expect a list type
                                                                  # Sort for readability

    def is_over(self):
        """
        Check whether the round is over

        Returns:
            (boolean): True if the current round is over
        """
        if self.not_raise_num + self.not_playing_num >= self.num_players:
            return True
        return False
