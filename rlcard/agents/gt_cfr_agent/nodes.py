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
from itertools import permutations, combinations
import numpy as np
from sparse import COO

# Internal imports
from rlcard.games.nolimitholdem.game import NolimitholdemGame, Stage
from rlcard.utils.utils import init_standard_deck

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
    #        A2: Future solution  - store only critical information, then use this 
    #                               info to create a NolimitHoldem game instance when we need to
    #                             
    #                               Examples. 
    #                                   - to determine payouts
    #                                   - to use knowledge of the game dynamics
    #                                     when adding a new node to the tree
    #
    #
    # *IMPORTANT* - It's the caller's responsibility to ensure that this is always
    #               a deep copy to prevent other nodes from modifying this node's game 
    #               information.
    #
    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # Game object associated with this node.
        #
        # Contains all game state info as well as game logic.
        #
        self.game = game
        
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
    # Helper function - sets values to zero.
    #
    def zero_values(self):
        self.values = np.zeros((self.game.num_players, 52, 52), dtype=np.float64)

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
        super.__init__(self, game, player_ranges)
        #
        # Poker games are terminated with showdowns
        #
        assert(self.game.stage == Stage.SHOWDOWN) # I think this is correct?
        #
        # Compute the payoff matrix for this node.
        #
        # Store it in memory for fast computation update_values() function.
        #
        self.cache_payoffs()
        #
        # Value over hands, for each player
        #
        # values[pid, card1, card2]
        #
        #     = exp. payoff for player pid when holding hand (card1, card2)
        #
        #     = sum_{card3, card4} opp_range[card2, card3] * payoff[pid, card1, card2, card3, card4] 
        #
        self.values = np.zeros((self.game.num_players, 52, 52))

    #
    # Caches the payoff matrix for each hand combination
    #
    #  payoffs[pid, card1, card2, card3, card4, ...]
    #      = payoff for player pid under the hand configuration
    #        P1 hand = (card1, card2), P2 hand = (card3, card4), ...
    #
    #  Entries in the payoff matrix associated with public cards are set to zero.
    #
    #  Note - a priori
    #
    #         payoffs[pid, card1, card2, card3, card4] =/= payoffs[pid, card3, card4, card1, card2]
    #
    #         Ex - if one player folds, then it doesn't matter which cards they have, they
    #              will always lose to the other player
    #
    #
    # NOTE 1 - Should we identify when a player folded and save time evaluating payoffs
    #          since the folding player always loses the entire pot, independent of
    #          hand combinations.
    #
    #          This would also save a significant amount of memory since the (2, 52, 52, 52, 52)
    #          matrix could be replaced by a single value (equal to the pot size).
    #
    # NOTE 2 - In the 2 player case, Player 1's payoff is always equal to (pot - Player 2's payoff)
    #        
    #          Extrapolating to >2 players, 
    #          Player 1's payoff is always equal to (pot - sum of other players' payoffs)
    #
    #          We could take advantage of this to only store (num players - 1) payoffs
    #
    # NOTE 3 - The payoff matrix is relatively sparse.
    #
    #          Greater than half of the entries will always be zero.
    #
    #          Payoff matrix size
    #              = (2 * 52^4 approx. 14 mil entries) * 8 bytes = approx. 136 MB
    #
    #          Most GPUs have at least 4 GB of VRAM so it should fit comfortably.
    #          However, we will have many terminal nodes, all with their own payoff
    #          matrices. We will likely need to shuttle the range and payoff
    #          matrices on/off the GPU in order to perform the value update computations.
    #
    #          One possible remedy would be to store a spare representation of the
    #          payoff matrix. There are a few libraries with GPU support that could
    #          do this. One possible solution would be to use cupy and 
    #          cupyx.scipy.sparse.coo_matrix
    #
    # NOTE 4 - This solution only really works for 2 player games
    #
    #          This approach scales poorly with >2 players
    #
    #          In >2 player games, the payoffs will likely have to be computed
    #          manually with each value_update call. Rather than being cached here.
    #
    def cache_payoffs(self):
        #
        # Remember the actual hands for each player
        #
        # NOTE - do we care about storing their real hands in this node?
        #
        real_hands = [self.game.players[pid].hand for pid in range(self.game.num_players)]

        #
        # Initialize payoffs matrix to all zeros
        #
        #
        self.payoffs = COO([self.game.num_players] + [52, 52]*self.game.num_players, np.float64)
        
        #
        # Get the set of possible cards 
        # the players can have in their hands
        #
        possible_cards = [card for card in init_standard_deck() if card not in self.game.public_cards]

        #
        # For each possible hand combination...
        #
        # Note -
        #     combinations(possible_cards, 2) 
        #         = list of all possible 2 card hands
        #
        #     permulations(..., num_player) 
        #         = list of all possible hand assignments to each player
        #
        for hands in permutations(combinations(possible_cards, 2), self.game.num_players):
            #
            # Assign the hypothetical hands to each player in the game instance
            #
            for pid, hand in enumerate(hands):
                self.game.players[pid].hand = hand
            #
            # Compute the payoffs for this hand configuration 
            # in this node's game state
            #
            hand_payoffs = self.game.get_payoffs()
            #
            # Translate cards in the hand configuration to indexes
            #
            card_idxs = [idx for hand in hands for idx in [hand[0].to_int(), hand[1].to_int()]]
            #
            # Set each player's payoffs in the matrix
            #
            for pid in enumerate(hands):
                self.payoffs[[pid] + card_idxs] = hand_payoffs[pid]
        
        #
        # Restore the players' real hands in the game object
        #
        for pid, hand in enumerate(real_hands):
            self.game.players[pid].hand = hand



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
    # NOTE 1 - This section needs to be rewritten for more than 2 players
    #
    # NOTE 2 - We can cache the payouts in a matrix. We do not need to re-compute
    #          them every time a value update is made because the only things that
    #          change are the player's ranges.
    #
    # NOTE 3 - Does this need to be changed? Shouldn't the expected value be
    #
    #          values[pid, card1, card2] = sum_{card3, card4} payoff(pid=(card1, card2), opp=(card3, card4)) * range[opp, card3, card4]
    #
    #          i.e. we need the set of values for the player over all hands in the infoset.
    #
    def update_values(self):
        #
        # For each player,
        # iterate over all possible hands in this infoset ...
        #
        # values
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
                    # NOTE - see NOTE 3 above
                    #
                    self.values[pid, card1, card2] = self.player_ranges[opp_pid] * utils[pid]
            #
            # Reset the opponent's hand to its actual value
            #
            self.game.players[opp_pid].hand = real_opp_hand

class DecisionNode(CFRNode):

    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # Start with the abstract class's initialization function
        #
        super().__init__(game, player_ranges)
        
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
            #
            # Update the ranges for the acting player in the child
            # states, according to the 
            #
