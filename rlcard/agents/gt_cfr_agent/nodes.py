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
#       - game - NolimitholdemGame object containing all information pertaining to the 
#                node's game state.
"""
#       - public_state - dictionary with public state information
#                        (public cards, pot, player chip stacks, etc.)
#
#       - active_players - boolean list of which player's are still active in the game.
#                          (if a player folds, then they're out of the game)
#                          NOTE - what if a player is all-in?
#
"""
#
#   Two abstract functions:
#
#       - update_values() - using CFR, compute the players' regret values and the acting player's policy.
#
#       - grow() - add a child node to this node's subtree according to the Growing-Tree CFR algo
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
    #                             Examples. 
    #                                 - to determine payouts
    #                                 - to use knowledge of the game dynamics
    #                                   when adding a new node to the tree
    #
    def __init__(self, game : NolimitholdemGame):
        #
        # Game object associated with this node.
        #
        # Contains all game state info as well as game logic.
        #
        # Note - It is the caller's responsibility to ensure that this is always
        #        a deep copy to prevent other nodes from modifying this node's game 
        #        information.
        #
        self.game = game

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
    # NOTE - right now, we store the entire game object
    #
    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # Use the inherited initialization function
        #
        # All other values are set to defaults
        #
        super.__init__(self, game)
        #
        # Poker games are terminated with showdowns
        #
        assert(self.game.stage == Stage.SHOWDOWN) # I think this is correct?
        #
        # Probability distribution over hands, for each player
        #
        # player_range[pid, card1, card2]
        #    = prob. player pid reaches this terminal node with hand (card1, card2)
        #
        # * IMPORTANT *
        # The np.array is stored by reference, and given to the terminal node
        # from its parent. Therefore, any change made to parent.player_ranges
        # is reflected here too.
        #
        self.player_ranges = player_ranges
        #
        # Value over hands, for each player
        #
        # values[pid, card1, card2] 
        #     = exp. payoff for player pid when holding hand (card1, card2)
        #
        self.cfr_values = np.zeros((self.game.num_players, 52, 52))

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
        for pid in range(self.game.num_players):
            #
            # Using 'current player' to refer to the player whose values
            # we are computing at this iteration
            #
            # If the current player's pid is 0, then the opponent's pid is 1
            # And, vice versa.
            #
            opp_pid = (pid + 1) % 2
            #
            # Remember the opponent's acutal hand
            #
            real_opp_hand = self.game.players[opp_pid]
            #
            # Get the set of possible cards the opponent can have in their hand
            #
            # The player knows the opponent cant have a card that's on the board or
            # in their own hand.
            #
            card_mask = lambda x: not (x in self.game.public_cards or x in self.game.players[pid].hand)
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
                    # Set the opponent's hand 
                    #
                    self.game.players[opp_pid].hand = hypot_opp_hand
                    #
                    # Get utilities for this hypothetical hand combination
                    #
                    utils = self.game.get_payoffs()
                    #
                    # Player 1's value for this hypothetical opponent hand is equal
                    # to the returned payoff weighted by the probability of the
                    # oponent holding that hand.
                    #
                    self.cfr_values[pid][card1, card2] = self.player_ranges[opp_pid] * utils[pid]
            #
            # Reset the opponent's hand to its actual value
            #
            self.game.players[opp_pid].hand = real_opp_hand



class DecisionNode(CFRNode):

    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # Start with the abstract class's initialization function
        #
        super().__init__(game)
        
        #
        # Player ranges are the probability that each player reaches this state,
        # under the current strategy profile, given a certain hand.
        #
        # For each player, this is expressed as a 52x52 upper triangular matrix,
        #
        # player_ranges[pid, card1, card2]
        #     = prob. player pid reaches this state, under the current strategy profile,
        #       given that they have the hand (card1, card2)
        #
        self.player_ranges = player_ranges
        
        #
        # List of legal actions
        #
        # This is the set of legal actions that can be taken by the player making the decision.
        #
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
        # The acting player's probability distribution over actions for all possible
        # hand combinations, given the public information.
        #
        # strategy[action, card1, card2] 
        #     = prob. of the acting player selecting the action 
        #       given she's holding the hand (card1, card2)
        #
        # Note - computing strategies for hands we don't have is neccessary for
        #        propagating the player's range down the game tree.
        #
        #
        # Start with an array of random values
        #
        self.strategy = np.random.rand(len(self.actions), 52, 52)

        #
        # Set duplicate card pairs to zero.
        #
        # i.e. we only need to track values for (card1, card2)
        #      and can set (card2, card1) to zero. 
        #
        for action in range(len(self.actions)):
            self.strategy[action, :, :] = np.triu(self.strategy[action, :, :], k=1)

        #
        # Set probabilities associated with hands containing public cards equal to zero
        #
        for card in self.game.public_cards:
            for action in range(len(self.actions)):
                self.strategy[action, card.to_int(), :] = 0.
                self.strategy[action, :, card.to_int()] = 0.

        #
        # Normalize the stategy
        #
        sum_values = self.strategy.sum(axis=0, keepdims=True)
        sum_values[sum_values == 0] = 1
        self.strategy /= sum_values

        #
        # Regret values over possible player actions for each hand in the infoset.
        #
        # Represented as an upper triangular matrix with zeros along the diagonal.
        #
        self.regrets  = np.zeros(len(self.actions), 52, 52)

        #
        # CFR value of holding each possible hand according to the current strategy profile
        #
        # This is represented as a 52x52 upper triangular matrix with the diagonal entries
        # set to zero
        #
        # values[pid, card1, card2] 
        #     = player pid's expected value given that they're in this state 
        #
        # All player's values are initialized to zero.
        #
        self.zero_values()

    #
    # Helper function - set values to zero.
    #
    def zero_values(self):
        self.values = np.zeros((self.game.num_players, 52, 52), dtype=np.float64)

    #
    # Perform a CFR value update
    #
    def update_values(self):
        #
        # Initalize player's values to zero
        #
        self.zero_values()

        #
        # For each child node...
        #
        for action in self.actions:

