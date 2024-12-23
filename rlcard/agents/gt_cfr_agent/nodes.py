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
import copy
from itertools import permutations, combinations
import numpy as np
from sparse import COO

# Internal imports
from rlcard.games.nolimitholdem.game import NolimitholdemGame, Stage
from rlcard.games.nolimitholdem.round import Action
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
    # The root node of the game tree is always a decision node
    # because the tree is constructed by a player in order to
    # make a decision.
    #
    # root_pid stores the player id for the player making the
    # decision at the root node
    #
    # This is used in the chance node because the player
    # who's constructing this game tree should not reason
    # about chance outcomes it knows can't happen.
    #
    # i.e. - the player knows it's impossible for a card in their
    #        hand to be dealt on the board.
    #
    root_pid = None

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
        # A player's range is the probability distribution over all possible hands,
        # given the sequence of actions that player has taken up to this game state.
        #
        # For each player, this is expressed as a 52x52 upper triangular matrix,
        #
        # player_ranges[pid, card1, card2]
        #     = prob. player pid reaches this state, under their current strategy,
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
        #     = the expected value for player pid holding hand=(card1, card2) 
        #
        # All player's values are initialized to zero.
        #
        self.zero_values()
    
    #
    # Helper function - sets values to zero.
    #
    def zero_values(self) -> None:
        self.values = np.zeros((self.game.num_players, 52, 52), dtype=np.float64)

    #
    # Store the player id for the acting player at the root node
    #
    @classmethod
    def set_root_pid(cls, pid : int) -> None:
        cls.root_pid = pid
    
    #
    # Get the player id for the acting player at the root node
    #
    @classmethod
    def get_root_pid(cls) -> int:
        assert(cls.root_pid)
        return cls.root_pid

    #
    # Compute the CFR values for this node
    #  
    @abstractmethod
    def update_values(self) -> None:
        pass

    #
    # Add a node to this node's subtree
    #
    @abstractmethod
    def grow(self) -> None:
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
    #     to treat (1) as a special case of (2) where the last
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
    # values[pid, card1, card2] = sum(payoff[pid, card1, card2, :, :] * range[opp, :, :])
    #
    #
    def update_values(self):
        #
        # For each player...
        #
        for pid in range(self.game.num_players):
            #
            # If pid is 0, then the opponent's pid is 1
            # And, vice versa.
            #
            # NOTE - eventually, this should be rewritten for >2 player games
            #
            opp_pid = (pid + 1) % 2
            #
            # Compute the expected value matrix for player pid
            #
            self.values[pid] = (self.payoffs[pid] * self.player_ranges[opp_pid][np.newaxis, np.newaxis, :, :]).sum(axis=(2,3))

    #
    # Add a node to this node's subtree
    #
    def grow(self):
        pass

#
# Decision node
#
# This node represents a decision made by a player in the game.
#
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
    # Updates the player's regrets and strategies 
    # according to the current values matrix.
    #
    def update_strategy(self) -> None:
        #
        # Get the id for the acting player
        #
        pid = self.game.game_pointer
        #
        # Use player values to the update the acting player's regrets
        #
        # Note 1: The regret for an action is defined as the value difference
        #         between always playing that action and playing our current strategy
        #
        # Note 2: Large positive regret = a good action, way better than our current strategy
        #         Large negative regret = a bad  action, way worse  than our current strategy
        #
        # Note 3: KEY IDEA - we want to shift our strategy toward actions that perform better
        #                    than our current strategy.
        #
        for action, child in enumerate(self.children):
            #
            # If the child is in the game tree
            #
            if child:
                self.regrets[action] += child.values[pid] - self.values[pid]
            #
            # Else, use the cfvn value
            #
            # NOTE - this value should be cached somewhere since we already had
            #        to compute this when we did the value update
            #
            else:
                pass
        #
        # Update the acting player's strategy according to the new regrets
        #
        # Using the regret matching formula
        #
        regret_pos = np.maximum(self.regrets, 0)
        regret_sum = np.sum(regret_pos, axis=0, keepdims=True) # sum along actions axis, (1, 52, 52) array
        self.strategy = regret_pos / regret_sum

    #
    # Perform a CFR value update
    #
    def update_values(self):
        #
        # Player id for the acting player
        #
        # Note - game_pointer holds the player id of the player making the decision
        #
        pid = self.game.game_pointer
        #
        # Initialize the player's values to zero
        #
        self.zero_values()
        #
        # For each child node...
        #
        for action, child in enumerate(self.children):
            #
            # If the child node is in the tree...
            #
            if child:
                #
                # Update the ranges for the acting player in the child
                # states, according to the acting player's strategy.
                #
                # For the acting player,
                # 
                # prob. of reaching child node 
                #     = prob. of reaching parent node * prob. of selecting action A
                # 
                # where action A transitions the game from the parent to the child node.
                #
                # 
                # For the non-acting player,
                #
                # range at child node = range at parent node
                #
                # NOTE - be careful to assign by value here and not by reference
                #
                child.player_ranges = np.copy(self.player_ranges)
                child.player_ranges[pid] = self.strategy[action] * self.player_ranges[pid]
                #
                # Compute the value of the child node
                #
                child.update_values()
                #
                # Use the child's values to update the parent's values
                #
                # For the acting player,
                #
                # the value contribution associated with selecting this action is
                # equal to the value of the child state weighted by the acting player's
                # probability of selecting the action
                #
                self.values[pid] += self.strategy[action] * child.values[pid]
                #
                # For the non-acting players,
                #
                # the player's value in the parent node is simply a sum of the 
                # player's values in the child nodes.
                #
                for opp_pid in range(self.game.num_players):
                    if opp_pid != pid:
                        self.values[opp_pid] += child.values[opp_pid]
            #
            # Else, 
            # 
            # The child is not in the tree,
            # we need to use the cfvn to estimate it. 
            #
            else:
                pass

        #
        # Now that the values have changed for this node,
        # the player's regrets and strategies need to be
        # updated to reflect this change in values.
        #
        self.update_strategy()
        
    #
    # Add the child node associated with the given action
    #
    def add_child(self, action : int) -> None:
        #
        # Validate the given action
        #
        assert 0 <= action < len(self.actions), "Invalid action index"
        assert self.children[action] is None, "Child already exists"
        
        #
        # Create a new game object for the new node
        # 
        # Initialize the object as a copy of the parent's game state
        #
        new_game = copy.deepcopy(self.game)
        
        #
        # Apply the given action to the (copy of) the parent's game state
        # to get the child's game state
        #
        action_str = new_game.get_legal_actions()[action]
        new_game.step(action_str)
        
        #
        # Build the child node
        #
        child_node = None
        
        #
        # Compute the child's player ranges
        #
        # NOTE - should this be made its own function, since the same
        #        operation is done in update_values()
        #
        pid = self.game.game_pointer
        child_ranges = np.copy(self.player_ranges)
        child_ranges[pid] = self.strategy[action] * self.player_ranges[pid]

        #
        # Case 1 - Child is a Chance Node 
        #
        # Check if the action caused a stage change
        #
        # NOTE: I'm not sure what the "END_HIDDEN" stage means?
        #
        if (self.game.stage != new_game.stage and 
            not new_game.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN)):
            #
            # Note: the chance node initializer expects the given game state
            #       to be the game state before cards are dealt
            #
            child_node = ChanceNode(copy.deepcopy(self.game), child_ranges)
        
        #
        # Case 2 - Child is a Terminal Node
        #
        elif new_game.stage == Stage.SHOWDOWN:
            child_node = TerminalNode(new_game, child_ranges)
        
        #
        # Case 3 - Child is a Decision Node
        #
        # NOTE - doing the stage check to exlude 'END_HIDDEN'
        #
        elif new_game.stage in (Stage.PREFLOP, Stage.FLOP, Stage.TURN, Stage.RIVER):
            child_node = DecisionNode(new_game, child_ranges)
        
        #
        # Case 4 - Unknown child
        #
        else:
            raise ValueError(f"Unrecognized game state in add_child(): {new_game}")
        
        #
        # Add the child to the parent node's child list
        #
        self.children[action] = child_node

    #
    # Add a node to this node's subtree
    #
    def grow(self, ):
        pass

#
# Chance node
#
# This node represents chance events in the game
#
# For poker, the only chance events that occur are when public cards
# are dealt on the board.
#
# Conceptually, chance nodes can be thought of as decision nodes where
# a 'chance' player takes a stochastic action.
#
class ChanceNode(CFRNode):

    #
    # Chance nodes transition poker states from one stage to the next.
    #
    # For instance, if a player selects an action that completes the preflop stage of
    # the game, then the chance node selects three cards to deal on the board, transitioning
    # the game to the flop stage.
    #
    # Stage.PREFLOP -> 3 cards dealt -> Stage.FLOP -> 1 card dealt -> Stage.TURN -> 1 card dealt -> Stage.RIVER
    #
    # In the above example, ChanceNode(..., ..., Stage.PREFLOP) selects a 3 card outcome
    # representing the 'chance' player selecting a random flop outcome.
    #
    # Note: stages can not be skipped, Stage.PREFLOP must transition to Stage.FLOP
    #       (assuming the game doesn't terminate)
    #

    #
    # Given the starting stage for the stage transition that this chance node
    # represents, return the number of cards that need to be dealt to complete
    # the transition
    #
    starting_stage_to_num_cards_dealt = {
        Stage.PREFLOP : 3, # Deal 3 cards to transition PREFLOP -> FLOP
        Stage.FLOP : 1,    # Deal 1 card to transition  FLOP -> TURN
        Stage.TURN : 1     # Deal 1 card to transition  TURN -> RIVER
    }

    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # Start with the base initialization function
        #
        super().__init__(game, player_ranges)

        #
        # Using the 'chance' player decision node analogy,
        #
        # Outcomes of chance events are the same as a 'chance'
        # player selecting an action that transitions deterministically
        # transitions the current game state to a new game state.
        #
        # So, the 'outcomes' list here is directly analgous to the 
        # 'actions' list in the decision node class.
        #
        # NOTE - would it be more or less confusing to just call the outcomes 'actions'?
        #
        # Outcomes is the set of possible card combinations that can be dealt for the
        # given transition.
        #
        # Note 1: For single card outcomes (FLOP->TURN, TURN->RIVER), the branching factor (=47, =46) 
        #         is not too bad. However, the 3 card outcome (PREFLOP->FLOP) branching factor is
        #         = (50 choose 3) = 19600 !!!
        #
        # Note 2: We don't consider outcomes that include cards that are in the root player's
        #         hand because she is the one constructing the tree and knows that these
        #         outcomes are impossible.
        #
        root_player = CFRNode.get_root_pid()
        valid_cards = [card for card in init_standard_deck() 
                            if (
                                (not card in self.game.public_cards) or
                                (not card in self.game.players[root_player].hand)    
                            )
                    ]
        self.outcomes = combinations(valid_cards,
                                     ChanceNode.starting_stage_to_num_cards_dealt[self.game.stage])
        
        #
        # Each outcome results in a unique child node
        #
        # Note: the game states for the child nodes are in the new game stage
        #
        # self.children[idx] = child node corresponding to the outcome at self.outcomes[idx]
        #
        self.children = {}

    
    #
    # Update values for all the players at this chance node
    #
    # The value for a player at this chance node is the sum of the
    # player's values at the child nodes, weighted by the probability
    # of the outcome.
    #
    def update_values(self):
        #
        # Initialize values to zero
        #
        self.zero_values()
        #
        # Probability of a given outcome occuring.
        #
        # Note: All cards are randomly drawn from the deck, so all outcomes are equally likely.
        #
        prob = 1 / len(self.outcomes)
        #
        # For each outcome of this chance node
        #
        # NOTE - this loop is a prime candidate for parallelism
        #
        for idx in range(len(self.outcomes)):
            #
            # If the child node is in the game tree...
            #
            if idx in self.outcomes:
                #
                # Get the child node associated with the outcome idx
                #
                child = self.children[idx]
                #
                # Update the player's ranges in the child node
                #
                # Note: here the 'chance' player is the acting player
                #       so the player's ranges at the child node are equal
                #       to their ranges at the parent node.
                #
                child.player_ranges = np.copy(self.player_ranges)

                #
                # Update the child node's player values
                #
                child.update_values()

                #
                # Update the player's values at this node according to
                # the child node's values.
                #
                self.values += prob * child.values

                #
                # Note: there is no strategy to update here because the
                #       'chance' player is purely stochastic
                #
            
            #
            # Else, the child node is not in the game tree
            # and we need the cfvn
            #
            else:
                pass

    #
    # Add the child node associated with the given outcome index
    #
    def add_child(self, idx : int) -> None:
        #
        # Validate the given outcome index
        #
        assert 0 <= idx < len(self.outcomes), "Invalid outcome index"
        assert not idx in self.children, "Child already exists"

        #
        # Create a new game object for the new node
        # 
        # Initialize the object as a copy of the parent's game state
        #
        new_game = copy.deepcopy(self.game)

        #
        # Perform the check/call action in the new game to cause the transition
        #
        # Only two actions cause the game to undergo a stage transition,
        # check and call.
        #
        # In RLCard, check and call is fused into a single action, Action.CHECK_CALL
        #
        # NOTE - For generality, should we just store the action that causes the transition
        #        at initialization time so that we don't have to assume Action.CHECK_CALL
        #        was the action that caused the stage transition?
        #
        new_game.step(Action.CHECK_CALL)

        #
        # During the step, the RLCard game will select a randomized outcome.
        #
        # We need to overwrite this outcome to the one we want.
        #
        outcome = self.outcomes[idx]
        assert len(new_game.public_cards) >= len(outcome), "Public cards is a different length than expected"
        new_game.public_cards[-1 * len(outcome):] = outcome

        #
        # Save the new child node in the parent's child list.
        #
        # Note 1 - All new poker stages start with a player decision.
        #
        # Note 2 - None of the player's made a decision at this node so their ranges
        #          at the child node remain unchanged. 
        #
        self.children[idx] = DecisionNode(new_game, np.copy(self.player_ranges))