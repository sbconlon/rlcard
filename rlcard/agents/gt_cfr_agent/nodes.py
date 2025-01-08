from __future__ import annotations  # Enables forward references

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
from rlcard.agents.gt_cfr_agent.cvfn import CounterfactualValueNetwork
from rlcard.games.limitholdem import PlayerStatus
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
#       - grow_tree() - add a child node to this node's subtree according to the Growing-Tree CFR algo
#
#
class CFRNode(ABC):
    #
    # Use global variables that are shared across nodes
    # to store private information that does not change across
    # game states. (ex. player's hands)
    #

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
    # CounterfactualValueNetowrk - this is set during tree initialization
    #                              and shared across all nodes in the tree.
    #
    # NOTE - would it make sense to use a different cfvn for each stage of the game?
    #
    cvfn = None

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
    #        A3: I think the minimal representation of a game state is its trajectory.
    #            If we have the trajectory then we can play forward the chance outcomes
    #            and the moves from the starting state until we reconstruct the given 
    #            game state.
    #
    #
    # *IMPORTANT* - It's the caller's responsibility to ensure that this is always
    #               a deep copy to prevent other nodes from modifying this node's game 
    #               information.
    #
    def __init__(self, game : NolimitholdemGame, player_ranges : np.array):
        #
        # If a node is active -
        #     then the game state is considered to be a full member of the game tree.
        #
        # If a non is inactive -
        #     then the game state is considered a pseudo-node of the game tree,
        #     these are placeholder leaf nodes that are used to store values for the
        #     game state, but have not been selected by the grow_tree() algorithm.
        #
        # All nodes are initialized as inactive.
        #
        # Note: This is a design decision to ensure that all nodes become active
        # through their own activation calls.
        #
        self.is_active = False

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
    # --> Helper functions
    # 
    # Sets values to zero.
    #
    def zero_values(self) -> None:
        self.values = np.zeros((self.game.num_players, 52, 52), dtype=np.float64)

    #
    # --> Class member setters
    #
    # Set the game tree's counterfactual value network
    #
    @classmethod
    def set_cvfn(cls, cvfn : CounterfactualValueNetwork) -> None:
        cls.cvfn = cvfn
    #
    # Get the game tree's counterfactual value network
    #
    @classmethod
    def get_cvfn(cls) -> CounterfactualValueNetwork:
        return cls.cvfn
    #
    # Set the player id for the acting player at the root node
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
    # --> Abstract base class methods
    #
    # Compute the CFR values for this node,
    # and return a list of queries made to the cvfn in the node's subtree
    #
    @abstractmethod
    def update_values(self) -> None:
        pass
    #
    # Add a node to this node's subtree,
    # return True if successful, False otherwise
    #
    @abstractmethod
    def grow_tree(self, hands : list[list[int]]) -> bool:
        pass
    #
    # Search this node's subtree for the node with the given trajectory.
    # If the node is found, return it.
    # If the node is not found, return its closest ancestor in the tree.
    #
    @abstractmethod
    def search(self, trajectory: list[int]) -> DecisionNode:
        pass

#
# Terminal Node
#
# This node type represents an endpoint for the game.
#
# NOTE - This class should probably be refactored for readability.
#
class TerminalNode(CFRNode):

    #
    # Payouts in showdown settings are only determined by the public cards.
    # Therefore, the Terminal nodes can share a cache of computed payoff
    # matrices.
    #
    # In practice, this is a great speedup because the game tree encounters
    # many of the same showdowns when considering river decisions.
    #
    # NOTE - Be careful, as of now, this is not thread safe!!!
    #
    payoffs_cache = {} # str representation of public cards -> payoff matrix

    #
    # Hash function for the payoffs cache
    #
    # Convert the 5 board cards to a unique string
    #
    @classmethod
    def five_cards_to_str(cls, board: list[Card]) -> str:
        assert len(board) == 5, "Only compute payoffs for river showdowns"
        output = ''
        for card in sorted(board, key=lambda c: c.to_int()):
            output += str(card)
        return output
    
    #
    # Check if the board for the game is in the payoffs cache
    #
    @classmethod
    def is_in_cache(cls, game: NolimitholdemGame) -> bool:
        board_id = TerminalNode.five_cards_to_str(game.public_cards)
        return board_id in TerminalNode.payoffs_cache

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
    @classmethod
    def cache_payoffs(cls, game: NolimitholdemGame):
        #
        # Check that the payoff matrix for this board hasn't already been computed
        #
        board = TerminalNode.five_cards_to_str(game.public_cards)
        assert board not in TerminalNode.payoffs_cache

        #
        # Remember the actual hands for each player
        #
        # NOTE 1 - Do we care about storing their real hands in this node?
        #          for the time being, we will insist that each node stores an accurate,
        #          full game state representation, even if this is not neccessary for this node.
        #
        real_hands = [game.players[pid].hand for pid in range(game.num_players)]

        #
        # Initialize payoffs matrix to all zeros
        #
        shape = [game.num_players] + [52, 52]*game.num_players
        payoffs = np.zeros(shape, dtype=np.float64)
        #self.payoffs = COO(
        #                   coords=np.empty((0, len(shape)), dtype=np.int64),  # No coordinates
        #                   data=np.empty(0, dtype=np.float64),                # No values
        #                   shape=shape
        #)

        #
        # Get the set of possible cards 
        # the players can have in their hands
        #
        possible_cards = [card for card in init_standard_deck() if card not in game.public_cards]

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
        for hands in permutations(combinations(possible_cards, 2), game.num_players):
            #
            # Filter out hand combinations that share cards
            #
            # i.e. It is impossible for Player 1 and Player 2 to both
            #      be holding the ace of spades.
            #
            if not set(hands[0]).isdisjoint(set(hands[1])):
                continue
            #
            # Assign the hypothetical hands to each player in the game instance
            #
            for pid, hand in enumerate(hands):
                game.players[pid].hand = list(hand)
            #
            # Compute the payoffs for this hand configuration 
            # in this node's game state
            #
            hand_payoffs = game.get_payoffs()
            #
            # Translate cards in the hand configuration to indexes
            #
            card_idxs = [idx for hand in hands for idx in [hand[0].to_int(), hand[1].to_int()]]
            #
            # Set each player's payoffs in the matrix
            #
            # Note: Scale each payout by the pot size.
            #       This way, showdowns with different pot sizes can share the same payout matrix
            #
            for pid in range(len(hands)):
                payoffs[pid, *card_idxs] = hand_payoffs[pid] / game.dealer.pot
        
        #
        # Restore the players' real hands in the game object
        #
        for pid, hand in enumerate(real_hands):
            game.players[pid].hand = hand
        #
        # Save the computed payout matrix in the payout cache
        #
        TerminalNode.payoffs_cache[board] = payoffs


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
        # Note: all terminal nodes are activated by default
        #
        # NOTE - for now, I don't see a reason why terminal nodes
        #        should be considered 'deactivated', but this might
        #        change depending on the overhead for computing
        #        payoff matrices.
        #
        super().__init__(game, player_ranges)
        self.is_active = True
        #
        # TerminalNodes only represent completed game states
        #
        assert self.game.is_over()
        #
        # If this is a showdown, we need to compute the full
        # payoff matrix.
        #
        # NOTE - Should we split this into two child classes?
        #
        self.num_active = sum([p.status != PlayerStatus.FOLDED for p in self.game.players])
        if self.num_active > 1:
            #
            # Compute the payoff matrix for this node.
            #
            # Store it in memory for fast computation update_values() function.
            #
            if not TerminalNode.is_in_cache(self.game):
                TerminalNode.cache_payoffs(self.game)
        #
        # Else, all the player's folded except for one.
        # In this case, the payouts are independent of the hands
        # the players are holding.
        #
        elif self.num_active == 1:
            self.payoffs = self.game.get_payoffs()
        else:
            raise ValueError("Terminal node reached with zero active players")
        #
        # Value of each hand, for each player
        #
        # values[pid, card1, card2]
        #
        #     = exp. payoff for player pid when holding hand (card1, card2)
        #
        #     = sum_{card3, card4} opp_range[card2, card3] * payoff[pid, card1, card2, card3, card4] 
        #
        self.values = np.zeros((self.game.num_players, 52, 52))

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
    # Returns an empty list because no querries to the cfvn were needed.
    #
    def update_values(self) -> list:
        #
        # Look up the payoffs matrix in the cache, if this is a river showdown
        #
        if self.num_active > 1:
            assert TerminalNode.is_in_cache(self.game), "Cant update values if the node's payoffs havent been computed"
            key = TerminalNode.five_cards_to_str(self.game.public_cards)
            payoffs = TerminalNode.payoffs_cache[key]

        #
        # For each player...
        #
        for pid in range(self.game.num_players):
            #
            # If pid is 0, then the opponent's pid is 1
            # And, vice versa.
            #
            # NOTE - Eventually, this should be rewritten for >2 player games
            #
            opp_pid = (pid + 1) % 2
            #
            # Compute the expected value matrix for player pid
            #
            # Note: payoffs from the cache have to be scaled by the pot size
            #
            if self.num_active > 1:
                self.values[pid] = self.game.dealer.pot * (payoffs[pid] * self.player_ranges[opp_pid][np.newaxis, np.newaxis, :, :]).sum(axis=(2,3))
            else:
                self.values[pid] = np.triu(np.ones((52, 52)) * self.payoffs[pid], k=1) * self.player_ranges[opp_pid]
        #
        # No querries were made to the cfvn
        #
        return []

    #
    # Just return false because we can't add a child to a terminal node
    #
    def grow_tree(self, hands : list[list[int]]) -> bool:
        return False
    
    #
    # Terminal nodes are active by default,
    # so nothing to do here.
    #
    def activate(self) -> None:
        pass
    
    #
    # If we hit a terminal node while following the game's trajectory, then
    # the trajectory is inavlid.
    #
    def search(self, trajectory: list[int]) -> DecisionNode:
        raise AssertionError("Search should not be run on a terminal node")

#
# Decision node
#
# This node represents a decision made by a player in the game.
#
class DecisionNode(CFRNode):
    #
    # Initialize a new, non-active, decision node
    #
    def __init__(self, game : NolimitholdemGame, player_ranges : np.ndarray):
        #
        # Start with the abstract class's initialization function
        #
        # Note: this initialized the node as non-active
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
            self.strategy[:, card.to_int(), :] = 0.
            self.strategy[:, :, card.to_int()] = 0.

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
        self.regrets  = np.zeros((len(self.actions), 52, 52))

    #
    # Updates the player's regrets and strategies 
    # according to the current values matrix.
    #
    def update_strategy(self) -> None:
        #
        # Verify the node is active
        #
        assert self.is_active, "Only active nodes in the game tree perform strategy updates"
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
        for action_idx, child in enumerate(self.children.values()):
            #
            # Children of active nodes should be initialized already
            #
            assert child is not None, "Active node should not have a null child"
            #
            # Perform the regret update for this node
            #
            self.regrets[action_idx] += child.values[pid] - self.values[pid]
        #
        # Update the acting player's strategy according to the new regrets
        #
        # Using the regret matching formula
        #
        regret_pos = np.maximum(self.regrets, 0)
        regret_sum = np.sum(regret_pos, axis=0, keepdims=True) # sum along actions axis, (1, 52, 52) array
        regret_sum[regret_sum == 0] = 1 # avoid dividing by zero
        self.strategy = regret_pos / regret_sum
        if np.isnan(self.strategy).any():
            import ipdb; ipdb.set_trace()
    
    #
    # Estimate this node's values using the cfvn
    #
    def update_values_w_cfvn(self):
        #
        # The cfvn should only be used on inactive nodes
        #
        assert not self.is_active, "CFVN should not be used on active nodes"
        #
        # Convert the node's information into an input vector for the cvfn
        #
        input = DecisionNode.get_cvfn().to_vect(self.game, self.player_ranges, False)
        #
        # Query the network
        #
        self.strategy, self.values = DecisionNode.get_cvfn().query(input)

    #
    # Perform a CFR value and strategy update
    #
    # Returns a list of queries made to the cvfn in the node's subtree
    #
    def update_values(self) -> list[tuple[np.ndarry]]:
        #
        # If this node is not active, 
        #
        assert self.is_active, "Value update attempted on an inactive node in the game tree"
        #
        # Else, the node is active in the game tree and we
        # need to perform a full value update with recursion.
        #
        # Player id for the acting player
        #
        # Note: game_pointer holds the player id of the player making the decision
        #
        pid = self.game.game_pointer
        #
        # Store copies of the opponent values and player range before the start
        # of the value update.
        #
        # This is needed for constructing query information.
        #
        # NOTE - For now, this assumes a 2 player game
        #
        opponent_values = np.copy(self.values[(pid + 1) % 2])
        player_range = np.copy(self.player_ranges[pid])
        #
        # Initialize the player's values to zero
        #
        self.zero_values()
        #
        # List to store cvfn queries made by the node's children
        #
        querries = []
        #
        # For each child node...
        #
        for action, child in self.children.items():
            #
            # Verify the children are not null
            #
            assert child is not None, "Active nodes cant have null children"
            #
            # Update the ranges for the players in the child
            # state, according to the acting player's strategy.
            #
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
            #
            # NOTE - be careful to assign by value here and not by reference
            #
            action_idx = self.actions.index(action)
            child.player_ranges = np.copy(self.player_ranges)
            child.player_ranges[pid] = self.strategy[action_idx] * self.player_ranges[pid]
            #
            # Compute the value of the child node
            #
            if child.is_active:
                #
                # Recurse down the game tree to compute the child's values
                #
                querries.append(child.update_values())
            else:
                #
                # Use the CFVN to estimate the child's values
                #
                child.update_values_w_cfvn()
                #
                # Cache the information needed for potentially solving this query
                #
                # A query takes the form:
                #   tuple(game state, opponent values, player range, trajectory seed)
                #
                # NOTE - For now, store the entire game object.
                #        
                #        This is a waste of memory.
                #
                #        In the future, we should only store the information
                #        needed to reconstruct the game state.
                #
                querries.append((self.game, opponent_values, player_range, [action.value]))
            #
            # Use the child's values to update the parent's values
            #
            # For the acting player,
            #
            # the value contribution associated with selecting this action is
            # equal to the value of the child state weighted by the acting player's
            # probability of selecting the action
            #
            self.values[pid] += self.strategy[action_idx] * child.values[pid]
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
        # Now that the values have changed for this node,
        # the player's regrets and strategies need to be
        # updated to reflect this change in values.
        #
        self.update_strategy()
        #
        # Return the querries made in this subtree
        # during the value update
        #
        return querries
        
    #
    # Add the child node associated with the given action.
    #
    # Note: Terminal nodes are always activated.
    #
    def add_child(self, action : Action) -> None:
        #
        # Validate the given action
        #
        assert action in self.actions, "Invalid action index"
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
        new_game.step(action)
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
        action_idx = self.actions.index(action) # convert enum -> int
        pid = self.game.game_pointer
        child_ranges = np.copy(self.player_ranges)
        child_ranges[pid] = self.strategy[action_idx] * self.player_ranges[pid]
        #
        # Case 1 - Child is a Terminal Node
        #
        if new_game.is_over():
            child_node = TerminalNode(new_game, child_ranges)
        #
        # Case 2 - Child is a Chance Node 
        #
        # Check if the action caused a stage change
        #
        # NOTE: I'm not sure what the "END_HIDDEN" stage means?
        #
        elif (self.game.stage != new_game.stage and 
            not new_game.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN)):
            #
            # Note: the chance node initializer expects the given game state
            #       to be the game state before cards are dealt
            #
            child_node = ChanceNode(copy.deepcopy(self.game), child_ranges)
        #
        # Case 4 - Child is a Decision Node
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
    # Activate the node
    #
    # Only leaf nodes can be activated, 
    # children can only be added to active nodes.
    # 
    # When activating a node, add its children as non-active nodes.
    # This alows for more efficient value updates on the active node.
    #
    def activate(self) -> None:
        #
        # Validate that this node satisfies the conditions
        # of a non-active node before activating it.
        #
        assert not self.is_active, "This node is already active"
        assert all(child is None for child in self.children.values()), "Non-active nodes can't have children"
        #
        # Set the activation flag to true
        #
        self.is_active = True
        #
        # Add the child nodes, as non-active nodes
        #
        for action in self.actions:
            self.add_child(action)

    #
    # Add a node to this node's subtree
    #
    # Return true if successful, false otherwise
    #
    # NOTE - we need to use a mixed strategy here using PUCT statistics.
    #        See SOG paper, page 15.
    #
    def grow_tree(self, hands : list[list[int]]) -> bool:
        #
        # If this node is not active, then activate it.
        #
        if not self.is_active:
            self.activate()
            return True
        #
        # Else, we need to continue sampling a trajectory through the game tree.
        #
        # Get the acting player's strategy for the provided hand.
        #
        pid = self.game.game_pointer
        strat = self.strategy[:, hands[pid][0], hands[pid][1]] # np.array, (num_actions,)
        #
        # Sample an action using the acting player's strategy
        #
        strat = strat / np.sum(strat) # renormalize to avoid floating point error weirdness
        try:
            action = np.random.choice(self.actions, p=strat)
        except:
            import ipdb; ipdb.post_mortem()
        #
        # Check that this child is not None
        #
        assert self.children[action] is not None, "Active nodes cant have null children"
        #
        # Recurse to the child
        #
        return self.children[action].grow_tree(hands)

    #
    # Search for the DecisionNode in the node's subtree corresponding
    # to the given trajectory.
    #
    # If the game state is not found, then return the nearest
    # node in the game tree.
    #
    # pid - the acting player's at at the target game state.
    #
    # trajectory - sequence of actions and chance outcomes that define
    #              the game state we are searching for.
    #
    # Note: Game states are uniquely defined by their trajectories
    #
    #       If two states have the same trajectores,
    #       then they are identical.
    #
    def search(self, trajectory: list[int]) -> DecisionNode:
        #
        # Search should only be conducted on active nodes.
        #
        assert self.is_active, "Search attempt on a non-active node"
        #
        # Get the depth of this node in the full game tree
        #
        n = len(self.game.trajectory) # Number of moves from the start of the game 
                                      # to reach this node's game state
        #
        # Check that the given search trajectory corresponds to a game state
        # in the game tree. 
        #
        # i.e. the root state's trajectory must be a subset of the search trajectory
        #
        if trajectory[:n] != self.root.game.trajectory:
            raise ValueError("The given search trajectory is not in the game tree")
        #
        # Initialize search variables
        #
        node = self
        prev_node = None
        depth = n
        #
        # Traverse down the game tree as far as possible
        #
        while node.is_active and isinstance(node, (DecisionNode, ChanceNode)):
            #
            # Found case - We found a node with the given trajectory
            #
            if depth == len(trajectory):
                assert trajectory == node.game.trajectory, "Node game state and trajectory mismatch"
                assert isinstance(node, DecisionNode), "Given trajectories must correspond to decision nodes"
                assert node.game.game_pointer == pid, "Given trajectories must correspond to decision nodes with the expected acting player"
                return node
            #
            # Continue searching
            #
            prev_node = node
            if isinstance(node, DecisionNode):
                node = node.children[trajectory[depth]]
            else:
                node = node.outcomes[trajectory[depth]]
            depth += 1
        #
        # Not found case - The search path hit the bottom of the game tree without finding
        #                  the node, traverse back up the search path looking for a
        #                  DecisionNode owned by the player pid.
        #
        if isinstance(prev_node, DecisionNode) or isinstance(prev_node, ChanceNode):
            return prev_node
        #
        # Error case - We could not find an exact match or an ancestor match.
        #
        # NOTE - Throw an error for now.
        #        In practice, this should not happen for poker.
        #
        raise ValueError("Given trajectory or ancestor not found")

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
        self.children = None

    #
    # Update values for all the players at this chance node
    #
    # The value for a player at this chance node is the sum of the
    # player's values at the child nodes, weighted by the probability
    # of the outcome.
    #
    # Returns a list of queries made to the cvfn in the node's subtree
    #
    def update_values(self) -> list[tuple[np.ndarray]]:
        #
        # If this node is not active, 
        # then estimate its values using the cfvn. 
        #
        if not self.is_active:
            #
            # All inactive nodes should be childless
            #
            assert not self.children, "Inactive nodes should not have children"
            #
            # Convert the nodes information into an input vector for the cvfn
            #
            input = DecisionNode.get_cvfn().to_vect(self.game, self.player_ranges, True)
            #
            # Query the network
            #
            self.values, self.strategy = DecisionNode.get_cvfn().query(input)
            #
            # Package the input/output of the network into a query tuple
            #
            return [(input, np.copy(self.values), np.copy(self.strategy))]
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
        # List to store querries to the cvfn made in the subtree
        #
        querries = []
        #
        # For each outcome of this chance node
        #
        # NOTE - this loop is a prime candidate for parallelism
        #
        for child in self.children:
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
            querries.append(child.update_values())
            #
            # Update the player's values at this node according to
            # the child node's values.
            #
            self.values += prob * child.values
            #
            # Note: there is no strategy to update here because the
            #       'chance' player is purely stochastic
            #
        return querries

    #
    # Add the child node associated with the given outcome index
    #
    def add_child(self, idx : int) -> None:
        #
        # Validate the given outcome index
        #
        assert self.is_active, "Only active nodes can have children"
        assert 0 <= idx < len(self.outcomes), "Invalid outcome index"
        assert self.children is not None, "Child list for an active node is none"
        assert self.children[idx] is None, "Child already exists"

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
        # NOTE - is there a better way to do this?
        #
        desired_outcome = self.outcomes[idx]
        assert (len(new_game.public_cards) >= len(desired_outcome), 
                "Public cards is a different length than expected")
        random_outcome = new_game.public_cards[-1 * len(desired_outcome):] 
        new_game.public_cards[-1 * len(desired_outcome):] = desired_outcome
        
        #
        # Return the cards that were dealt on the random outcome back to the deck.
        #
        assert (all(card not in self.game.dealer.deck for card in random_outcome), 
                "Cards that were dealt are still in the deck")
        self.game.dealer.deck.append(random_outcome)

        #
        # Save the new child node in the parent's child list.
        #
        # Note 1 - All new poker stages start with a player decision.
        #
        # Note 2 - None of the player's made a decision at this node so their ranges
        #          at the child node remain unchanged. 
        #
        self.children[idx] = DecisionNode(new_game, np.copy(self.player_ranges))
    
    #
    # Make the node an active node in the game tree
    #
    def activate(self) -> None:
        #
        # Verify that the node is well-formatted
        #
        assert not self.is_active, "Cant activate an already active node"
        assert self.children is None, "Non-active nodes should not have children"
        #
        # Set the activation flag
        #
        self.is_active = True
        #
        # Add the children as non-active nodes
        #
        # Note: this is a costly operation.
        #
        # NOTE - refactor and parallelize this
        #
        self.children = [None] * len(self.outcomes)
        for idx in range(len(self.outcomes)):
            self.add_child(idx)
    
    #
    # Add a child node to this subtree
    #
    def grow_tree(self, hands : list[list[int]]) -> bool:
        #
        # If this node isn't active, then activate it.
        #
        if not self.is_active:
            self.activate()
            return True
        #
        # Else, we need to continue sampling a trajectory.
        #
        # Randomly sample an outcome from the outcomes list
        #
        # If the sampled outcome contains a card in the player's hands,
        # then sample another outcome.
        #
        # Repeat until a valid outcome is found.
        #
        # Note 1: This saves time versus scanning the entire outcomes list.
        #
        #         We have an approx. 80% chance of sampling a valid action.
        #
        #         The expected number of retries follows the geometric distribution:
        #             E[retries] = 1/p = 1/.8 = 1.25 retries
        # 
        #         This is significantly faster in expectation than scanning the entire
        #         outcomes list.
        #
        # Note 2: We are always guaranteed that a valid outcome exists.
        #
        used_cards = {card for hand in hands for card in hand}
        while True:
            idx = np.random.choice(len(self.outcomes))
            if all(card.to_int() not in used_cards for card in self.outcomes[idx]):
                break
        #
        # If the child node associated with this outcome is not in the game tree,
        # then add it.
        #
        if idx not in self.children:
            self.add_child(idx)
            return True
        #
        # Else, the child is not in the game tree and we can recurse.
        #
        return self.children[idx].grow_tree(hands)
