# ========================================================================= #
#                                                                           #
#  This file defines the three node types for the game tree:                #
#                                                                           #
#      * Terminal node - represents the end of the game                     #
#                      - determines the payoff for each player              #
#                                                                           #                        
#      * Chance node - represents chance actions in the game                #
#                    - for poker, this is the action of public cards        #
#                      being dealt on the board.                            #
#                                                                           #
#      * Decision node - represents a decision made by a player             #
#                      - each decision node is owned by one player          #
#                                                                           #
# ========================================================================= #

# External imports
from abc import ABC, abstractmethod
import numpy as np

# Internal imports
from rlcard.games.nolimitholdem.game import NolimitholdemGame, Stage

#
# Abstract base class for a node in the CFR public tree
#
#   Information that is universally neccessary across all node types:
#
#       - public_state - dictionary with public state information
#                        (public cards, pot, player chip stacks, etc.)
#
#       - active_players - boolean list of which player's are still active in the game.
#                          (if a player folds, then they're out of the game)
#                          NOTE - what if a player is all-in?
#
#   Two abstract functions:
#
#       - values() - using CFR, compute the players' regret values and the acting player's policy.
#
#       - grow() - add children nodes according to Growing-Tree CFR
#
#
class CFRNode(ABC):

    """
    #
    # Use global variables that are shared across nodes
    # to store private information that does not change across
    # game states. (ex. player's hands)
    #
    
    #
    # NOTE - this is potentially not thread safe when switching to multiparallelism
    #
    
    #
    # Player hands
    #
    # [(player1 card1, player1 card2), (player2 card1, player2 card2), ...]
    #
    # Neccessary for determining the winner of a showdown
    #
    for now, this is replaced by self.game.players[id].hand
    hands = None
    """

    #
    # NOTE - How to relate nodes to their corresponding game representations
    #
    #        Q: How much game information should be stored in each node?
    #
    #        A1: Current solution - store the entire game object
    #
    #        A2: Future solution- store only critical information, then use this 
    #                             info to create a NolimitHoldem game instance when we need to
    #                             (ex. when we need to determine payouts)
    #
    def __init__(self, game : NolimitholdemGame,
                       player_range : np.array, 
                       opponent_values : dict,   # Note - (i think) this should be moved to decision+chance classes
                       pid : int = -1,           # Note - this should be moved to decision+chance classes
                       actions : list = None):   # Note - this should be moved to decision+chance classes
        #
        # Game object associated with this node.
        #
        # Contains all game state info as well as game logic.
        #
        self.game = game

        #
        # Public state for this node
        #
        """ 
        for now, replaced by c.game.get_state()
        self.public_state = public_state
        """

        #
        # Player id for the player making the decision
        #
        # -1 if this is a decision or terminal node
        #
        """
        for now, this is replaced by self.game.game_pointer
        self.pid = pid
        """

        #
        # List of players that are still active in this hand
        #
        #  active_player[pid] = 
        #
        # There must always be one player that is active in a hand
        #
        # NOTE - I think this will only be used during showdowns 
        #
        """
        for now, this is replaced by game.players[pid].status
        self.player_statuses = player_statuses
        """

        #
        # Total number of players
        #
        """
        for now, this is replaced by self.game.num_players
        self.nplayers = len(active_players)
        """

        #
        # List of legal actions
        #
        #  - Decision node - 
        #  This is the set of legal actions that can be taken by the player making the decision.
        #
        #  - Chance node -
        #  This is the set of cards that can be dealt given the public state.
        #
        #  - Terminal node -
        #  None. This is the end of the game so no decisions need to be made.
        #
        #self.actions = self.public_state['raw_obs']['legal_actions']
        """
        for now, this is replaced by self.game.get_legal_actions()
        self.actions = actions
        """
        self.actions = self.game.get_legal_actions()

        #
        #  *IMPORTANT* - If an action is selected in the tree, then the game
        #                is deterministicly transitioned from this node to the 
        #                child node associated with that action.
        #
        
        #
        # Child states that result from taking an action in this state
        #
        # Equal to None if this is a terminal node. (get rid of this)
        #
        self.children = {a: None for a in self.actions} if self.actions else None
        
        #
        # Player's probability distribution over actions in this state
        # initalized to a random strategy.
        #
        random_strategy = np.random.rand(len(self.actions))
        random_strategy /= random_strategy.sum()
        self.strategy = dict(zip(self.actions, random_strategy))
        
        #
        # Regret values over possible player actions
        #
        self.regrets  = {a: 0. for a in self.actions}

        #
        # Regret values for the range gadget
        #
        self.gadget_regrets = {a: 0. for a in self.actions}
        
        #
        # CFR value of holding each possible hand according to the current strategy profile
        #
        # This is represented as a 52x52 upper triangular matrix with the diagonal entries
        # set to zero
        #
        # The input opponent values are taken as gadget values, this is to account
        # for the fact that the opponent can steer the game away from this public state
        # if she chooses.
        #
        # The values computed by cfr for both players are initialized to zero.
        #
        # Note: the goal of CFR is to compute the active player's values.
        #
        self.gadget_values = opponent_values
        self.cfr_values = {a: np.zeros((52, 52), dtype=np.float64) for a in self.actions}

        #
        # Probability of the player holding a hand in the public state
        # 
        # This is represented as a 52x52 upper triangular matrix with the diagonal entries
        # set to zero and the entire matrix is normalized.
        #
        # public_cards = list of indicies corresponding to publicly observed cards
        #
        def random_range():
            #
            # Initialize an upper triangular matrix of random values 
            # with zeros on the diagonal
            #
            rand_range = np.triu(np.random.rand(52, 52), k=1)
            #
            # Set the probability of holding public cards to zero
            #
            rand_range[self.public_cards, :] = 0.
            rand_range[:, self.public_cards] = 0.
            #
            # Normalize
            #
            rand_range /= rand_range.sum()

            return rand_range
        #
        # Note - state['raw_obs']['public_cards'] is a vector of Card objects
        #
        self.public_cards = [card.to_int() for card in self.game.public_cards]
        self.ranges = [
                        random_range() if i != pid else player_range 
                        for i in range(self.game.num_players)
                      ]
    """
    For now, we don't need this because the game object is stored in the node

    #
    # Helper function
    #
    # Takes the node's state information 
    # and returns a corresponding game instance.
    #
    # This is useful for leveraging rlcard's built-in functions
    # for nolimitholdem, such as get_payoffs()
    #
    def to_game(self) -> NolimitholdemGame:
        #
        # Initialize a NolimitholdemGame object using the information
        # stored in this node.
        #
        game = NolimitholdemGame(
                    num_players=self.nplayers,
                    fixed_public_cards=self.public_state['raw_obs']['public_cards'],

                )
        return
    """

    #
    # Compute the CFR values for this node
    #  
    @abstractmethod
    def update_values(self):
        pass

    #
    # Add a node to this node's subtree
    #
    @abstractmethod
    def grow(self):
        pass

    """
    For now this is commented out becuase the full game object
    is being stored in the node class

    #
    # Set player's hands for the class
    #
    @classmethod
    def set_hands(cls, player_hands : list):
        cls.hands = player_hands
    
    #
    # Get player's hands for the class
    #
    @classmethod
    def get_hands(cls):
        return cls.hands
    """        


#
# Terminal Node
#
# This node type represents an endpoint for the game.
#
class TerminalNode(CFRNode):

    #
    # NOTE 1 - the payoffs returned by update_values() depend on how this terminal
    #          node was reached
    #
    #          There are two ways a nolimitholdem game can end
    #
    #            1) A player folds
    #                --> the entire payout goes to the other player
    #
    #            2) A player checks or calls on the river
    #                --> the player's payouts are determined by a showdown
    #
    #           Currently, (2) is written into the code, but (1) is not.
    #
    #
    #  NOTE 2 - follow up to Note 1 -
    #
    #     I think the easiest way to handle this (and how I think
    #     rlcard handles it inside the nolimitholdem game class) is
    #     to treat a (1) as a special case of (2) where the last
    #     remaining player goes to a single player showdown.
    #     (trivially, this just means that player recieves the entire pot). 
    #
    
    #
    # Only store the information neccessary for computing player payoffs.
    #
    def __init__(self, game : NolimitholdemGame, 
                       player_range : np.array, 
                       opponent_values : dict):
        #
        # Use the inherited initialization function
        #
        # All other values are set to defaults
        #
        super.__init__(self, game, player_range, opponent_values)

    #
    # Given a state node in the public state tree, compute the updated cfr values, 
    # cfr regrets, and policies for each infoset in the public state.
    #
    # Base case for traversing the game tree.
    #
    # Note: Terminal nodes do not have children so we do
    #       not need to update regrets or strategies.
    #
    # The player's values at terminal nodes are their expected payoffs
    #
    #
    # NOTE - This section needs to be rewritten for more than 2 players
    #
    def update_values(self):
        #
        # For each player,
        # iterate over all possible hands in this infoset ...
        #
        for pid in range(self.nplayers):
            #
            # Using 'current player' to refer to the player whose values
            # we are computing at this iteration
            #
            # If the current player's pid is 0, then the opponent's pid is 1
            # And, vice versa.
            #
            opp_pid = (pid + 1) % 2
            #
            # Get the set of possible cards the opponent can have in their hand
            #
            # The player knows the opponent cant have a card that's on the board or
            # in their own hand.
            #
            card_mask = lambda x: not (x in self.public_cards or x in CFRNode.get_hands[pid])
            possible_cards = list(filter(card_mask, range(52)))
            #
            # For each possible hand the opponent could have...
            #
            for i, card1 in enumerate(possible_cards):
                for card2 in possible_cards[i+1:]:
                    #
                    # Hypothetical opponent hand
                    #
                    hypot_opp_hand = (card1, card2)
                    #
                    # Initialize a nolimitholdem game instance for this hypothetical
                    # setup to determine the payoff.
                    #
                    hypot_hands = [None, None]
                    hypot_hands[pid] = CFRNode.get_hands[pid]
                    hypot_hands[opp_pid] = hypot_opp_hand
                    hypot_game = self.to_game(hypot_opp_hand)
                    #
                    # Get utilities for this hypothetical hand combination
                    #
                    utils = hypot_game.get_payoffs()
                    #
                    # Player 1's value for this hypothetical opponent hand is equal
                    # to the returned payoff weighted by the probability of the
                    # oponent holding that hand.
                    #
                    self.cfr_values[pid][card1, card2] = self.ranges[opp_pid][card1, card2] * utils[pid]

        
