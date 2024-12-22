from enum import Enum

import numpy as np
from copy import deepcopy
from collections.abc import Iterable

from rlcard.games.base import Card
from rlcard.games.limitholdem import Game
from rlcard.games.limitholdem import PlayerStatus

from rlcard.games.nolimitholdem import Dealer
from rlcard.games.nolimitholdem import Player
from rlcard.games.nolimitholdem import Judger
from rlcard.games.nolimitholdem import Round, Action


class Stage(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class NolimitholdemGame(Game):
    def __init__(self, allow_step_back=False, num_players=2, fixed_public_cards=[], fixed_player_cards={}, starting_stage=Stage.PREFLOP):
        """Initialize the class no limit holdem Game"""
        super().__init__(allow_step_back, num_players)
        self.np_random = np.random.RandomState()

        # small blind and big blind
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # config players
        self.init_chips = [100] * num_players

        # If None, the dealer will be randomly chosen
        self.dealer_id = None

        #
        # Deterministically deal cards
        #
        #
        # Set to insure the same card isn't dealt twice
        #
        observed_cards = set()
        
        #
        # Fix the cards that will be dealt on the flop, turn, and river
        # Cards are dealt in order (first 3 cards on the flop, etc.)
        #
        assert(isinstance(fixed_public_cards, Iterable))
        assert(len(fixed_public_cards) <= 5)
        for elem in fixed_public_cards:
            assert(isinstance(elem, Card))
            assert(not elem in observed_cards)
            observed_cards.add(elem)
        self.fixed_public_cards = fixed_public_cards

        #
        # Fix the cards that will be dealt to each player
        # fixed_player_cards: player_id -> [Card1, Card2]
        #
        assert(isinstance(fixed_player_cards, dict))
        assert(len(fixed_player_cards) < num_players)
        for key, value in fixed_player_cards.items():
            assert(isinstance(key, int))
            assert(isinstance(value, list))
            assert(key in range(num_players))
            assert(isinstance(value, Iterable))
            assert(len(value) == 2)
            assert(isinstance(value[0], Card) and isinstance(value[1], Card))
            assert(not (value[0] in observed_cards or value[1] in observed_cards))
            observed_cards.add(value[0])
            observed_cards.add(value[1])
        self.fixed_player_cards = fixed_player_cards

        #
        # Fix the starting stage of the game
        #
        if starting_stage is None:
            starting_stage = 'preflop'
        assert(starting_stage in ('preflop', 'flop', 'turn', 'river'))
        str_to_stage = {'preflop': Stage.PREFLOP, 'flop': Stage.FLOP, 'turn': Stage.TURN, 'river': Stage.RIVER}
        self.starting_stage = str_to_stage[starting_stage]

    def configure(self, game_config):
        """
        Specify some game specific parameters, such as number of players, initial chips, and dealer id.
        If dealer_id is None, he will be randomly chosen
        """
        self.num_players = game_config['game_num_players']
        # must have num_players length
        self.init_chips = [game_config['chips_for_each']] * game_config["game_num_players"]
        self.dealer_id = game_config['dealer_id']
    
    def deal_public_cards(self):
        if not self.dealer:
            raise ValueError("Dealer must be set before dealing public cards.")
        if not isinstance(self.public_cards, Iterable):
            print(f'Public cards: {self.public_cards}')
            raise TypeError("public_cards must be an iterable.")
        
        num_public_cards = {
            Stage.PREFLOP: 0,
            Stage.FLOP: 3,
            Stage.TURN: 4,
            Stage.RIVER: 5
        }
        
        if self.stage not in num_public_cards:
            raise ValueError(f"Invalid stage: {self.stage}")
        
        # Total cards needed for the current stage
        total = num_public_cards[self.stage]
        current_count = len(self.public_cards)
        num_cards_needed = total - current_count
        
        if num_cards_needed < 0:
            raise ValueError("public_cards already contains more cards than required for this stage.")
        
        # Add cards from fixed_public_cards if available
        cards_from_fixed = self.fixed_public_cards[current_count:(current_count + num_cards_needed)]
        self.public_cards += cards_from_fixed
        
        # Draw additional cards from the dealer if needed
        remaining_needed = total - len(self.public_cards)
        self.public_cards += [self.dealer.deal_card() for _ in range(remaining_needed)]

    def init_game(self):
        """
        Initialize the game of not limit holdem

        This version supports two-player no limit texas holdem

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        """
        if self.dealer_id is None:
            self.dealer_id = self.np_random.randint(0, self.num_players)

        # Initialize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)
        
        # Remove fixed cards from the deck
        for card in self.fixed_public_cards:
            print(f'Removing {card}')
            self.dealer.remove_card(card)
        for hand in self.fixed_player_cards.values():
            print(f'Removing {hand[0]}')
            print(f'Removing {hand[1]}')
            self.dealer.remove_card(hand[0])
            self.dealer.remove_card(hand[1])

        # Initialize players to play the game
        self.players = [Player(i, self.init_chips[i], self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        # Deal cards to each  player to prepare for the first round
        for pid in range(self.num_players):
            if pid in self.fixed_player_cards:
                self.players[pid].hand = self.fixed_player_cards[pid]
            else:
                self.players[pid].hand.append(self.dealer.deal_card())
                self.players[pid].hand.append(self.dealer.deal_card())

        # Initialize the starting stage of the game
        self.stage = self.starting_stage

        # Initialize public cards
        self.public_cards = []
        self.deal_public_cards()
        
        # Big blind and small blind
        s = (self.dealer_id + 1) % self.num_players
        b = (self.dealer_id + 2) % self.num_players
        self.players[b].bet(chips=self.big_blind)
        self.players[s].bet(chips=self.small_blind)

        # If the stage is PREFLOP, then
        # the player next to the big blind plays first
        if self.stage == Stage.PREFLOP:
            self.game_pointer = (b + 1) % self.num_players
        # Otherwise, the small blind starts
        else:
            self.game_pointer = s

        # Initialize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(self.num_players, self.big_blind, dealer=self.dealer, np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 4 rounds in each game.
        self.round_counter = int(self.stage.value)

        # Save the history for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_legal_actions(self):
        """
        Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        """
        return self.round.get_nolimit_legal_actions(players=self.players)

    def step(self, action):
        """
        Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next player id
        """

        # Verify the given action is legal
        if action not in self.get_legal_actions():
            print(action, self.get_legal_actions())
            print(self.get_state(self.game_pointer))
            raise Exception('Action not allowed')

        if self.allow_step_back:
            # First take a snapshot of the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            self.history.append((r, b, r_c, d, p, ps))

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        players_in_bypass = [1 if player.status in (PlayerStatus.FOLDED, PlayerStatus.ALLIN) else 0 for player in self.players]
        if self.num_players - sum(players_in_bypass) == 1:
            last_player = players_in_bypass.index(0)
            if self.round.raised[last_player] >= max(self.round.raised):
                # If the last player has put enough chips, he is also bypassed
                players_in_bypass[last_player] = 1

        # If a round is over, we deal more public cards
        if self.round.is_over():
            # Game pointer goes to the first player not in bypass after the dealer, if there is one
            self.game_pointer = (self.dealer_id + 1) % self.num_players
            if sum(players_in_bypass) < self.num_players:
                while players_in_bypass[self.game_pointer]:
                    self.game_pointer = (self.game_pointer + 1) % self.num_players

            # For the first round, we deal 3 cards
            if self.round_counter == 0:
                self.stage = Stage.FLOP
                self.deal_public_cards()
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            # For the following rounds, we deal only 1 card
            if self.round_counter == 1:
                self.stage = Stage.TURN
                self.deal_public_cards()
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            if self.round_counter == 2:
                self.stage = Stage.RIVER
                self.deal_public_cards()
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_state(self, player_id):
        """
        Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        """
        self.dealer.pot = np.sum([player.in_chips for player in self.players])

        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()
        state = self.players[player_id].get_state(self.public_cards, chips, legal_actions)
        state['stakes'] = [self.players[i].remained_chips for i in range(self.num_players)]
        state['current_player'] = self.game_pointer
        state['pot'] = self.dealer.pot
        state['stage'] = self.stage
        return state

    def step_back(self):
        """
        Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        """
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players = self.history.pop()
            self.stage = Stage(self.round_counter)
            return True
        return False

    def get_num_players(self):
        """
        Return the number of players in no limit texas holdem

        Returns:
            (int): The number of players in the game
        """
        return self.num_players

    def get_payoffs(self):
        """
        Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        """
        hands = [p.hand + self.public_cards if p.status in (PlayerStatus.ALIVE, PlayerStatus.ALLIN) else None for p in self.players]
        chips_payoffs = self.judger.judge_game(self.players, hands)
        return chips_payoffs

    @staticmethod
    def get_num_actions():
        """
        Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 6 actions (call, raise_half_pot, raise_pot, all_in, check and fold)
        """
        return len(Action)
