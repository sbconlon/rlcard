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
import tensorflow as tf
import time
import treys

# Internal imports
from rlcard.games.base import Card
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.utils import random_strategy, starting_hand_values, invalid_hands, get_2d_coors, get_1d_coor
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.nolimitholdem.game import NolimitholdemGame, Stage
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck

# Card-removal helpers for fold terminals: hand h = (HAND_CARD1[h], HAND_CARD2[h]),
# in the same hand order as every other 1326-vector (np.triu_indices(52, k=1)).
INVALID_HANDS_F32 = tf.cast(invalid_hands, tf.float32)   # (52, 1326): 1 where hand j contains card c
_hand_i, _hand_j = np.triu_indices(52, k=1)
HAND_CARD1 = tf.constant(_hand_i, dtype=tf.int32)         # (1326,)
HAND_CARD2 = tf.constant(_hand_j, dtype=tf.int32)         # (1326,)


class CFRTree:
    #
    # Initializes an empty game tree
    #
    # Inputs:
    #
    #    - cfvn = neural network used to approx. strategies and values
    #  
    #    - n_players = num. of players in the game
    #
    #
    def __init__(self, cfvn: CounterfactualValueNetwork, n_players: int=2, use_gadget: bool=True):
        #
        # Store the counterfactual value network
        #
        self.cfvn = cfvn
        self.cfvn_enabled = tf.constant(bool(self.cfvn), dtype=tf.bool)
        #
        # Re-solving (gadget on): the opponent's root range is produced by the gadget game.
        # Fixed-belief solve (gadget off): both root ranges are given and stay fixed; used
        # for query solving (paper: 'a subgame with fixed beliefs for both players').
        #
        self.use_gadget = bool(use_gadget)
        #
        # Store the number of players and actions
        #
        self.n_players = n_players
        self.all_actions = []
        #
        # Counts the number of nodes in the tree
        #
        self.n_nodes = tf.constant(0)
        #
        # Game tree
        #
        #   - Adjacency matrix
        #
        #   - tree[i, j] = action index if node j is a descendant of node i otherwise -1
        #
        #   - Example
        #
        #            D
        #      T     T    T
        #
        #      tree = [ -1  0  1  2 ]
        #             [ -1 -1 -1 -1 ]
        #             [ -1 -1 -1 -1 ]
        #             [ -1 -1 -1 -1 ]
        #
        #      Nodes 1, 2, 3 are terminal nodes and descendants of Node 0.
        #
        self.tree = None
        #
        # Node types
        #
        #   - Vector of length = # of nodes
        #
        #   - node_types[i] = 0 if node i is a decision node
        #                     1 if node i is a non-showdown terminal node
        #                     2 if node i is a showdown terminal node
        #
        #   - Example
        #
        #           D
        #     T     T    T
        #
        #     node_types = [0, 1, 1, 1]
        #
        #     Nodes 1, 2, 3 are terminal nodes and descendants of Node 0.
        #
        self.node_types = None
        #
        # Game states
        #
        #   - Vector of length = # of nodes
        #
        #   - game_states[i] = game state object for node i
        #
        self.game_states: list[NolimitholdemGame]
        self.game_states = []
        #
        # Legal actions
        #
        #   - Matrix of size (# of nodes, # of actions)
        #
        #   - legal_actions[i, j] = whether or not action j can be taken at node i.
        #
        self.legal_actions = None
        #
        # Players
        #
        #    - Vector of length = # of nodes
        #
        #    - players[i] = player id of the acting player if node i is a decision node
        #                   -1 if node i is a terminal node
        #
        self.players = None
        #
        # Activate nodes
        #
        #    - Vector of length = # of nodes
        #
        #    - active_nodes[i] = boolean whether node i is active
        #
        self.active_nodes = None
        #
        # Range map
        #
        #   - Matrix of size (# nodes, # of players)
        #
        #   - range_map[i, j] = idx for the range vector associated with 
        #                       player j at node i.
        #
        self.range_map = None
        #
        # Ranges
        #
        #   - Matrix of size (# of decision nodes + # of players, # of hands)
        #
        #   - NOTE: 1326 is the number of distinct hands
        #
        #   - ranges[range_map[i, j], k] = prob. of player j reaching node i with hand k
        #
        self.ranges = None
        #
        # Values
        #
        #   - Matrix of size (# of nodes, # of player, # of hands)
        #
        #   - values[i, j, k] = expected value of player j having hand k in node i
        #
        self.values = None
        #
        # Strategy indexes
        #
        #    - Matrix of size = (# of nodes, total # of actions)
        #
        #    - strat_idxs[i, j] = 
        #       
        #        If action j is a valid action at node i,
        #        then it's the row id into the strategy matrix
        #
        #        Otherwise, -1.
        #
        self.strat_idxs = None
        #
        # Strategies
        #
        #    - Matrix size = (Approx. # of decision nodes * # of actions, # of hands)
        #
        #    - strategies[strat_idxs[i, j], k] 
        #         = prob. of player at node i selecting action j with hand k
        #
        self.strategies = None
        #
        # Cummulative Strategies
        #
        #    - Matrix size = (Approx. # of decision nodes * # of actions, # of hands)
        #
        #    - cum_strategies[strat_idxs[i, j], k]
        #         = cummulative strategy over all updates for node i, action j, and hand k
        #
        self.cum_strategies = None
        #
        # Regrets
        #
        #    - Matrix size = (Approx. # of decision nodes * # of actions, # of hands)
        # 
        #    - regrets[strat_idxs[i, j], k]
        #          = regret of player at node i selecting action j with hand k 
        #
        self.regrets = None
        #
        # Vector indexes
        #
        #    - Vector of length = # of nodes
        #
        #    - vect_idxs[i] = 
        #
        #        If node i is an inactive decision node, idx into vect_idxs
        #
        #        Otherwise, -1.
        #
        self.vect_idxs = None
        #
        # Vectorized decision node prefixs (contain all non-range elements)
        #
        #    - Matrix of size = (Approx. # of non-active decision nodes, # of prefix-features)
        #
        #    - feat_vect_prefixs[vector_idxs[i]] = feature vector prefix for node i
        #
        self.feat_vect_prefixs = None
        #
        # Non-showdown winner
        #
        #   - Vector of size = # of nodes
        #
        #   - winner[i] =
        #
        #       If node i is a non-showdown t
        #
        self.non_showdown_winner = None
        #
        # Put in pot
        #
        #   - Matrix of size = (# of nodes, # of players)
        #
        #   - pots[i, j] = chips player j has put into the pot at node i
        #
        self.put_in_pot = None
        #
        # Board string to payoff idx
        #
        #   - Dictionary that maps board strings to idxs in the payoff matrix 
        #
        #   - board_to_idx[board_str] = row id into the payoff matrix
        #
        #   - If board_str is not in the dictionary, then the payoff has not
        #     been cached yet.
        #
        self.board_to_idx = {}
        #
        # Payoff indexes
        #
        #   - Vector of size = # of nodes
        #
        #   - playoffs_idxs[i] =
        #
        #       If node i is a terminal node with a showdown,
        #       then node i row id into the payoff matrix.
        #
        #       -1 otherwise.
        #
        self.payoffs_idxs = None
        #
        # Payoffs
        #
        #   - Matrix size = (# of distinct payoffs, # of hands, # of hands)
        #
        #   - payoffs[payoffs_idxs[i], j, k]
        #       = payoff to the player at node i holding hand j against hand k
        #
        self.payoffs = None
        #
        # Visits
        #
        #   - Vector of size = # of nodes
        #
        #   - visits[i] = # of times node i has been visited during growth
        # 
        self.visits = None
        #
        # Bind the compiled CFR functions
        #
        self._bind_compiled()



    #####################################
    #                                   #
    #            Getters                #
    #                                   #
    #####################################

    def get_strategy(self, node: int) -> np.ndarray:
        res = tf.gather(self.strategies, self.strat_idxs[node, :]).numpy()
        return res[self.legal_actions[node], :]
    
    #
    # Average strategy over the solve so far (linearly weighted). Each hand's column is
    # normalized on read, which also handles nodes that joined the search late.
    #
    def get_cum_strategy(self, node: int) -> np.ndarray:
        res = tf.gather(self.cum_strategies, self.strat_idxs[node, :]).numpy()[self.legal_actions[node], :]
        sums = res.sum(axis=0, keepdims=True)
        return np.divide(res, sums, out=np.zeros_like(res), where=sums > 0)
    
    #
    # Ids of the frontier: inactive decision nodes, the ones the CFVN is asked about.
    #
    def frontier_nodes(self) -> np.ndarray:
        return np.where((self.node_types.numpy() == 0) & (~self.active_nodes.numpy()))[0]

    #
    # A logged CFVN query: the public state at a frontier node and both players' ranges
    # there, exactly as the network saw them. Deep-copied so later CFR iterations and
    # pickling to a worker process cannot touch the tree's own objects.
    #
    def frontier_query(self, node: int) -> tuple[NolimitholdemGame, np.ndarray]:
        game = copy.deepcopy(self.game_states[node])
        ranges = tf.gather(self.ranges, self.range_map[node]).numpy().copy()
        return game, ranges


    def get_values(self, node: int) -> np.ndarray:
        return self.values[node, :, :]



    #####################################
    #                                   #
    #      Build Tree Functions         #
    #                                   #
    ##################################### 

    #
    # Initialize the gadget game
    #
    # A key idea of re-solving for imperfect information games
    #
    # For any player making a decision at a game state, 
    # they must consider that their opponent can choose to steer the game 
    # toward that game state or not.
    #
    # The example used in the literature is rock-paper-scissors.
    #
    #     - Player 1 selects an action (r-p-s) that's hidden from Player 2
    #
    #     - At Player 2's decision node, they must reason about how frequently
    #       Player 1 chooses to play toward each state.
    #
    #       i.e. the frequency at which Player 1 chooses each action
    #
    #     - This reasoning about their opponent's strategy in an ancestor node,
    #       then informs their strategy in the current node.
    #
    #       e.x. If Player 2 knows that Player 1 always chooses to play toward the paper node, then
    #            their optimal strategy at that node would be to always play Scissors.
    #
    #
    # This reasoning is modeled here by the "regret gadget" 
    # 
    # The oppponent has two (fictitious) actions:
    #
    #     - "Follow"    (F) - choose to play toward the root state
    #
    #     - "Terminate" (T) - reject the root state, select actions to play away from the root state
    # 
    # The values for the two actions:
    #
    #     - value for following   (f_values) - the opponent's values at the 
    #                                          root state of the cfr tree.
    #                                          = self.root.player_values[opponent player id]
    #
    #     - value for terminating (t_values) - this an input for re-solving
    #                                          and is either heuristically derived
    #                                          or taken from the solution of a previous CFR run
    #                                          = self.terminate_values
    #
    # Using these values, the opponent can compute an associated "gadget regret" and
    # "gadget strategy" for the follow-or-terminate gadget decision.
    #
    # This strategy can then be taken as the opponent's range in the root node
    # of the CFR for the next CFR value update iteration.
    # 
    # To initialize the gadget game, we need the opponent's values for 
    # choosing the Terminate gadget action.
    #
    #     Case 1: This is the start of the game. We need to use a heuristic
    #     to estimate the opponent player's values. This can be done by precomputing 
    #     the winning percentage of each hand if the game checked to showdown.
    #
    #     Case 2: This is not the start of the game. Use the opponent players' values
    #     from the previous CFR run.
    #
    # Note: The naming conventions here are a tad confusing. 'terminate_values' is
    #       the payoff matrix the opponent player recieves in the gadget game
    #       when she selects the Terminate action.
    #
    #       This is not to be confused with the 'gadget_values' which is the opponent's
    #       expected value in the gadget game according to their current gadget strategy
    #       and the current cfr values at the root node.
    #
    #       So, the word 'values' here is used twice to mean two different things.
    #
    #       This arises because the gadget game is a meta-game. In the gadget game,
    #       the opponent is seeking to maximize her expected value from payoffs, but 
    #       those payoffs are themselves values for the real game.
    #
    # NOTE - How does the gadget game change in the >2 player game setting. 
    #
    def init_gadget_game(self, input_opponents_values: np.ndarray =None):
        root_game = self.game_states[0]
        if input_opponents_values is not None:
            self.terminate_values = tf.constant(np.asarray(input_opponents_values, dtype=np.float32)) # same type as the heuristic branch
        else:
            self.terminate_values = tf.constant(starting_hand_values(root_game) * root_game.dealer.pot, dtype=tf.float32) # = t_values = v_2 in the literature
        self.gadget_regrets = tf.Variable(tf.zeros((2, 1326), dtype=tf.float32), trainable=False) # 2 gadget actions, (Follow, Terminate)
        self.gadget_values = tf.Variable(tf.zeros((1326,), dtype=tf.float32), trainable=False)
    
    #
    # Creates the root node
    #
    def add_root_node(self, game_state: NolimitholdemGame, 
                            player_ranges: np.ndarray,
                            opponent_values: np.ndarray =None) -> None:
        #
        # Sanity checks
        #
        assert self.n_nodes == 0, f"Trying to add a root node to a tree with {self.n_nodes} nodes."
        assert player_ranges.shape == (self.n_players, 1326), f"Expected {(self.n_players, 1326)}, Got {player_ranges.shape}"
        #
        # Set all the actions
        #
        self.all_actions = [action.value for action in game_state.get_all_actions()]
        #
        # Set the idxs of the public cards
        #
        # public_card_mask - vector of size # of hands
        #
        #     public_card_mask[i] = whether hand i contains a public card
        #
        root_pub_cards = tf.constant([card.to_int() for card in game_state.public_cards], dtype=tf.int32)
        self.public_card_mask = tf.reduce_any(tf.gather(invalid_hands, root_pub_cards), axis=0)
        #
        # Set internal tensors
        #
        self.tree = tf.constant([[-1]], dtype=tf.int32) # nodes are never descendants of themselves
        self.node_types = tf.constant([0], dtype=tf.int8) # all root nodes are decision nodes
        self.players = tf.constant([game_state.game_pointer], dtype=tf.int32)
        self.game_states.append(game_state)
        legal_actions = [action.value for action in game_state.get_legal_actions()]
        self.legal_actions = tf.constant([[action in legal_actions for action in self.all_actions]], dtype=tf.bool)
        self.active_nodes = tf.constant([False], dtype=tf.bool)
        self.range_map = tf.constant([np.arange(self.n_players)], dtype=tf.int32)
        self.ranges = tf.constant(player_ranges, dtype=tf.float32)
        self.values = tf.constant(tf.zeros((1, self.n_players, 1326), dtype=tf.float32))
        legal_actions = [self.all_actions.index(action.value) for action in game_state.get_legal_actions()]
        self.strategies = tf.Variable(random_strategy(len(legal_actions), game_state.public_cards), dtype=tf.float32, trainable=False)
        self.regrets = tf.Variable(np.zeros((len(legal_actions), 1326)), dtype=tf.float32, trainable=False)
        self.cum_strategies = tf.Variable(tf.zeros_like(self.strategies), trainable=False)
        self.cfr_iters = tf.Variable(0.0, trainable=False) # CFR iterations run on this tree so far (the linear-averaging weight)
        self.vect_idxs = tf.constant([-1], dtype=tf.int32) # Root node never inactive
        self.put_in_pot = tf.constant([p.in_chips for p in game_state.players], dtype=tf.float32)[None, :]
        self.non_showdown_winner = tf.constant([-1], dtype=tf.int32)
        root_strat_idxs = [-1] * len(self.all_actions)
        for idx, action in enumerate(legal_actions):
            root_strat_idxs[action] = idx
        self.strat_idxs = tf.constant([root_strat_idxs], dtype=tf.int32)
        self.payoffs_idxs = tf.constant([-1], dtype=np.int32)
        self.visits = tf.constant([0], dtype=tf.int32)
        #
        # Increment node count
        #
        self.n_nodes += 1
        #
        # Initialize gadget game
        #
        if self.use_gadget:
            self.init_gadget_game(opponent_values)

    #
    # Adds a decision node to the game tree
    #
    # Inputs
    #
    #    - parent (int) = node id of the parent of the new node.
    #
    def add_child(self, parent_id: int, action_id: int) -> None:
        #
        # Get the child game state
        #
        assert 0 <= parent_id < self.n_nodes, f"Invalid parent id: {parent_id}"
        parent_game = self.game_states[parent_id]
        action = Action(action_id)
        assert action in parent_game.get_legal_actions()
        action_idx = self.all_actions.index(action_id)
        child_game = copy.deepcopy(parent_game)
        child_game.step(action)
        #
        # Determine the node type for the child
        #
        num_active = lambda game: sum([p.status != PlayerStatus.FOLDED for p in game.players])
        child_type = None
        if child_game.is_over(): # Terminal node
            if num_active(child_game) == 1:
                child_type = 1 # Non-showdown
            else:
                child_type = 2 # Showdown
        elif (parent_game.stage != parent_game.stage and 
            not child_game.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN)):
            child_type = 3 # Chance node
        elif child_game.stage in (Stage.PREFLOP, Stage.FLOP, Stage.TURN, Stage.RIVER):
            child_type = 0 # Decision node
        #
        # Add child to the tree matrix
        #
        self.tree = tf.pad(self.tree, paddings=[[0, 1], [0, 1]], constant_values=-1)
        new_col = self.tree.shape[1] - 1
        self.tree = tf.tensor_scatter_nd_update(
            self.tree,
            indices=[[parent_id, new_col]],
            updates=[action_idx],
        )
        #
        # Add child to node types
        #
        self.node_types = tf.concat([self.node_types, tf.constant([child_type], dtype=tf.int8)], axis=0)
        self.players = tf.concat([self.players, tf.constant([child_game.game_pointer], dtype=tf.int32)], axis=0)
        #
        # Add child to game states
        #
        self.game_states.append(child_game)
        #
        # Add child to legal actions
        #
        legal_actions = [action.value for action in child_game.get_legal_actions()]
        child_legal_actions = tf.constant([[action in legal_actions for action in self.all_actions]], dtype=tf.bool)
        self.legal_actions = tf.concat([self.legal_actions, child_legal_actions], axis=0)
        #
        # Add child to active nodes
        #
        # Note - terminal nodes are always considered active
        #
        self.active_nodes = tf.concat([self.active_nodes, tf.constant([child_type in (1, 2)], dtype=tf.bool)], axis=0) # terminals (fold and showdown) are always active
        #
        # Add child range
        #
        child_id = self.n_nodes
        parent_player_id = self.players[parent_id]
        self.range_map = tf.concat([self.range_map, self.range_map[parent_id, None]], axis=0)
        child_range = self.strategies[self.strat_idxs[parent_id, action_idx]] * self.ranges[self.range_map[parent_id, parent_player_id]]
        self.ranges = tf.Variable(tf.concat([self.ranges, child_range[None, :]], axis=0))
        self.range_map = tf.tensor_scatter_nd_update(self.range_map,
                                                     indices=[[child_id, parent_player_id]],
                                                     updates=[self.ranges.shape[0] - 1])
        #
        # Add child values
        #
        self.values = tf.Variable(
            tf.concat([self.values, tf.zeros((1, self.n_players, 1326), dtype=tf.float32)], axis=0)
        )
        #
        # Add strategy and feature vector
        #
        self.strat_idxs = tf.concat([self.strat_idxs, tf.fill((len(self.all_actions),), -1)[None, :]], axis=0)
        if child_type == 0: # Decision node
            legal_actions = [self.all_actions.index(action.value) for action in child_game.get_legal_actions()]
            child_strategy = tf.constant(random_strategy(len(legal_actions), child_game.public_cards), dtype=tf.float32)
            base = self.strategies.shape[0]
            self.strategies = tf.Variable(tf.concat([self.strategies, child_strategy], axis=0), trainable=False)
            self.regrets = tf.Variable(
                tf.concat([self.regrets, tf.zeros((len(legal_actions), 1326), dtype=tf.float32)], axis=0),
                trainable=False
            )
            self.cum_strategies = tf.Variable( # extend, keeping every existing node's accumulated average
                tf.concat([self.cum_strategies, tf.zeros((len(legal_actions), 1326), dtype=tf.float32)], axis=0),
                trainable=False
            )
            indices = tf.stack( # [[child_id, action1], [child_id, action2], ...] NOTE - I think we can replace this
                [tf.fill((len(legal_actions),), child_id), legal_actions],
                axis=1
            )
            vals = base + tf.range(len(legal_actions))
            self.strat_idxs = tf.tensor_scatter_nd_update(self.strat_idxs, indices, vals)
        #
        # Add feature-prefix index: one entry per node, like every other per-node tensor.
        # feat_vect_prefixs holds one 57-float row per decision node only;
        # vect_idxs[node] is that node's row, or -1 when it has none.
        #
        if child_type == 0 and self.cfvn:
            prefix = self.cfvn.get_prefix(child_game)[None, :]          # (1, 57)
            if self.feat_vect_prefixs is None:
                prefix_row = 0
                self.feat_vect_prefixs = prefix
            else:
                prefix_row = self.feat_vect_prefixs.shape[0]
                self.feat_vect_prefixs = tf.concat([self.feat_vect_prefixs, prefix], axis=0)
        else:
            prefix_row = -1
        self.vect_idxs = tf.concat([self.vect_idxs, tf.constant([prefix_row], dtype=tf.int32)], axis=0)
        #
        # Add pot size
        #
        self.put_in_pot = tf.concat([self.put_in_pot, tf.constant([[p.in_chips for p in child_game.players]], dtype=tf.float32)], axis=0)
        #
        # Add payoffs
        #
        if child_type == 1: # Non-showdown terminal
            self.payoffs_idxs = tf.concat([self.payoffs_idxs, tf.constant([-1], dtype=tf.int32)], axis=0)
            active_players = [p.status != PlayerStatus.FOLDED for p in child_game.players]
            winner = active_players.index(True)
            self.non_showdown_winner = tf.concat([self.non_showdown_winner, tf.constant([winner], dtype=tf.int32)], axis=0)
        elif child_type == 2: # Showdown terminal
            self.non_showdown_winner = tf.concat([self.non_showdown_winner, tf.constant([-1], dtype=tf.int32)], axis=0)
            board_str = five_cards_to_str(child_game.public_cards)
            if board_str in self.board_to_idx: # Cache hit
                self.payoffs_idxs = tf.concat([self.payoffs_idxs, [self.board_to_idx[board_str]]], axis=0)
            else: # Cache miss
                payoff_matrix = tf.cast(tf.expand_dims(compute_payoff_matrix(child_game), axis=0), tf.float32) # TODO: compute payoffs w/ tensorflow
                if self.payoffs:
                    self.payoffs = tf.concat([self.payoffs, payoff_matrix], axis=0)
                else:
                    self.payoffs = payoff_matrix
                self.payoffs_idxs = tf.concat([self.payoffs_idxs, tf.constant([self.payoffs.shape[0]-1])], axis=0)
                self.board_to_idx[board_str] = self.payoffs_idxs[child_id]
        else: # Non-terminal node
            self.non_showdown_winner = tf.concat([self.non_showdown_winner, tf.constant([-1], dtype=tf.int32)], axis=0)
            self.payoffs_idxs = tf.concat([self.payoffs_idxs, tf.constant([-1], dtype=tf.int32)], axis=0)
        #
        # Add visits
        #
        self.visits = tf.concat([self.visits, tf.constant([0], dtype=tf.int32)], axis=0)
        #
        # Update node count
        #
        self.n_nodes += 1
        #
        # The tensors were rebuilt: discard the compiled traces
        #
        self._bind_compiled()
    
    #
    # Active nodes are considered part of the tree.
    #
    # All their children are added to the tree as non-active nodes.
    #
    def activate(self, node_id: int):
        # Validate input
        assert 0 <= node_id < self.n_nodes, f"Invalid node id {node_id}"
        assert self.node_types[node_id] == 0, "Only decision nodes can be activated"
        assert not self.active_nodes[node_id], "Node is already active"
        # Activate the node
        self.active_nodes = tf.tensor_scatter_nd_update(self.active_nodes, indices=[[node_id]], updates=[True])
        # Add the node's children to the tree
        game = self.game_states[node_id]
        legal_action_ids = [action.value for action in game.get_legal_actions()]
        for action_id in legal_action_ids:
            self.add_child(node_id, action_id)

    #
    # Activate the children of a given node
    #
    def activate_children(self, node_id: int):
        # Validate input
        assert 0 <= node_id < self.n_nodes, f"Invalid node id {node_id}"
        assert self.node_types[node_id] == 0, "Only decision nodes can be activated"
        assert self.active_nodes[node_id], "Node is not already active"
        # Activate the children
        for child_id in np.where(self.tree[node_id, :] >= 0)[0]:
            if self.node_types[child_id] == 0 and not self.active_nodes[child_id]:
                self.activate(child_id)
    
    #
    # Fully expand the tree and activate all decision nodes.
    #
    def activate_full_tree(self):
        assert self.n_nodes > 0, "Must initialize the root node"
        node = 0
        while node < self.n_nodes:
            if self.node_types[node] == 0 and not self.active_nodes[node]:
                self.activate(node)
            node += 1



    #####################################
    #                                   #
    #          CFR Functions            #
    #                                   #
    ##################################### 

    #
    # (Re)bind the compiled CFR functions.
    #
    # tf.function captures the tensor and Variable OBJECTS it sees when it traces, and
    # add_child rebuilds those objects on every new node. A trace made before a node was
    # added therefore reads and writes stale copies. Rebinding discards the traces, so the
    # next call re-traces (and XLA recompiles) against the current tree. Called from
    # __init__ and at the end of add_child.
    #
    def _bind_compiled(self):
        self.update_gadget_regrets = tf.function(self._update_gadget_regrets_impl, jit_compile=True)
        self.update_ranges         = tf.function(self._update_ranges_impl, jit_compile=True)
        self.update_values_w_cfvn  = tf.function(self._update_values_w_cfvn_impl, jit_compile=True)
        self.update_values         = tf.function(self._update_values_impl, jit_compile=True)
        self.cfr_update_w_cfvn     = tf.function(self._cfr_update_w_cfvn_impl, jit_compile=True)
        self.cfr_update_full_tree  = tf.function(self._cfr_update_full_tree_impl, jit_compile=True)

    #
    # Update the regrets in the gadget game
    #
    # NOTE - implement this function for >2 player games, shouldn't be too hard
    #
    def _update_gadget_regrets_impl(self):
        #
        # Compute the gadget strategy using the gadget regrets
        #
        # Note 1: Because there are only two actions for this decision (T, F),
        #         we only need to compute the prob. of selecting one action.
        #         
        #         Here we choose to compute the follow prob. because it is
        #         used by the cfr root node.
        #
        # Note 2: Let, self.gadget_regret[0] be the Follow    action regrets
        #         and, self.gadget_regret[1] be the Terminate action regrets
        #
        pos0 = tf.maximum(self.gadget_regrets[0], 0.0)
        pos1 = tf.maximum(self.gadget_regrets[1], 0.0)
        denom = pos0 + pos1
        probs = tf.math.divide_no_nan(pos0, denom)
        follow_prob = tf.where(denom == 0.0, 0.5, probs) # per-hand P(Follow); stays unnormalized for the gadget's own value and regrets
        #
        # In the above line, we assign 50-50 probability to hands with zero in the denominator.
        # This works for valid hands, but has the side effect of giving invalid hands non-zero
        # reach probabilities.
        #
        # Mask out invalid hands
        #
        follow_prob = tf.where(self.public_card_mask, 0.0, follow_prob)
        opp_range = tf.math.divide_no_nan(follow_prob, tf.reduce_sum(follow_prob)) # normalized copy: a distribution over hands, used ONLY as the range
        #
        # Set the opponent's range in the cfr root node to the gadget's follow strategy
        #
        # Reasoning:
        #     Gadget follow strat = probability of choosing to play toward the cfr root state
        #     Opp. root range = prob. of the opp. reaching this state given her strategy
        #     Therefore, gadget follow strat = opponent's range at the root node.
        #
        # Two different quantities come out of the same vector, and only one is normalized:
        #
        #     follow_prob[h] - per-hand probability that the opponent holding h chooses Follow.
        #                      Used for the gadget's expected value and regrets below. NEVER
        #                      normalized across hands: the gadget is a separate two-action
        #                      game for each hand.
        #
        #     opp_range[h]   - follow_prob normalized to sum to one over all hands, used as the
        #                      opponent's range at the root (a distribution over hands).
        #
        # Note 1 - normalizing the range is not in DeepStack's pseudocode, but it fits the
        #          reasoning of the gadget game: the start of the subgame can be thought of as
        #          a game where the players are dealt cards from a weighted deck.
        #
        # Note 2 - without normalization the actor's values are ordinary counterfactual values
        #          (sums over opponent hands), which are large but not wrong. With it they are
        #          expected values conditional on the opponent's hand distribution. This is a
        #          convention choice; every solve must use the same one so that opt-out values
        #          carried between re-solves stay on the same scale as root values.
        #
        # Note 3 - multiplying all of the opponent's reach probabilities by a constant scales the
        #          actor's counterfactual values uniformly and does not change the actor's strategy
        #          (regret matching is scale invariant).
        #
        # Note 4 - Bayes' perspective:
        #          Un-normalized = prob. the opponent reaches the root state given they hold hand (i, j)
        #          Normalized    = prob. the opponent holds hand (i, j) given that they reach the root state
        #
        opp_pid = tf.cast((self.players[0] + 1) % 2, tf.int32)
        range_id = tf.cast(self.range_map[0, opp_pid], tf.int32)
        self.ranges.scatter_nd_update(
            indices=tf.reshape(range_id, [1, 1]), 
            updates=tf.reshape(opp_range, [1, -1])
        )
        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        opp_vals = self.values[0, opp_pid, :]
        new_gadget_values = (follow_prob * opp_vals + 
                             (1.0 - follow_prob) * self.terminate_values)
        #
        # Update the gadget regrets
        #
        # Subtle point:
        #     - If the opponent chooses to Follow    then they recieve the gadget value at iteration t
        #     - If the opponent chooses to Terminate then they recieve the gadget value at iteration t - 1
        #
        # Why?
        #     - I think because, conceptually, if the opponent is choosing to terminate at time t, then
        #       they don't get to observe the payoffs that would've happened to the CFR tree at time t
        #
        # Note:
        #    - Always selecting Follow    yields a fixed payoff equal to the opp. cfr values at the root node
        #    - Always selecting Terminate yields a fixed payoff equal to the terminate values
        #
        follow_regrets = tf.maximum(self.gadget_regrets[0] + opp_vals - new_gadget_values, 0.0)
        term_regrets = tf.maximum(self.gadget_regrets[1] + self.terminate_values - self.gadget_values, 0.0)
        self.gadget_regrets[0].assign(follow_regrets)
        self.gadget_regrets[1].assign(term_regrets)
        #
        # Update the gadget values to the new values
        #
        self.gadget_values.assign(new_gadget_values) # in place: a plain rebind is not carried across compiled iterations

    # NOTE - We can further improve this function by applying the range updates on 
    #        a per level basis in parallel.
    def _update_ranges_impl(self):
        parents = tf.where(self.active_nodes & (self.node_types == 0))[:, 0] # Active decision nodes
        parents = tf.cast(parents, tf.int32)
        def body(i):
            parent = parents[i]
            player = self.players[parent]
            branch = tf.gather(self.tree, parent)
            child_mask = branch >= 0
            actions = tf.boolean_mask(branch, child_mask)
            children = tf.where(child_mask)[:, 0]
            child_range_idxs = tf.gather(self.range_map[:, player], children)
            strat_idxs = tf.gather(self.strat_idxs[parent, :], actions)
            new_rows =  tf.gather(self.ranges, self.range_map[parent, player]) * tf.gather(self.strategies, strat_idxs)
            self.ranges.scatter_nd_update(tf.expand_dims(child_range_idxs, axis=1), new_rows)
            return (i+1,)
        i, n = tf.constant(0, dtype=tf.int32), tf.shape(parents)[0]
        cond = lambda i: i < n
        tf.while_loop(cond, body, (i,))

    def _update_values_w_cfvn_impl(self):
        #
        # No decision child has ever been created (e.g. a re-solve rooted at a spot whose
        # children are all terminal): there is no frontier and no prefix table. Python-level
        # check, resolved at trace time.
        #
        if self.feat_vect_prefixs is None:
            return
        # Get the batch of featurized game states
        inactive_nodes = tf.cast(tf.where((self.node_types == 0) & (~self.active_nodes))[:, 0], tf.int32) # non-active decision nodes
        N = tf.shape(inactive_nodes)[0]
        #
        # Nothing on the frontier right now (every decision node is active): skip the query.
        # Tensor-level check, becomes a tf.cond.
        #
        if N > 0:
            rm = tf.gather(self.range_map, inactive_nodes)
            ranges_block = tf.gather(self.ranges, rm)
            ranges_flat  = tf.reshape(ranges_block, [N, self.n_players * 1326])
            vects = tf.concat(
                [
                    tf.gather(self.feat_vect_prefixs, tf.gather(self.vect_idxs, inactive_nodes)),
                    ranges_flat
                ],
                axis=1
            )
            # Query the cfvn
            strats, vals = self.cfvn.query(vects)
            # Keep only this node's legal actions and renormalize each hand's column over them;
            # a hand whose legal mass is zero falls back to uniform over the legal actions.
            legal_mask = tf.gather(self.legal_actions, inactive_nodes)                        # (N, A) bool
            legal_f = tf.cast(legal_mask, strats.dtype)[:, :, None]                            # (N, A, 1)
            strats = strats * legal_f
            legal_sum = tf.reduce_sum(strats, axis=1, keepdims=True)                           # (N, 1, 1326)
            n_legal = tf.reduce_sum(legal_f, axis=1, keepdims=True)                            # (N, 1, 1)
            strats = tf.where(legal_sum > 0.0, tf.math.divide_no_nan(strats, legal_sum), legal_f / n_legal)
            strats = strats * tf.cast(~self.public_card_mask, strats.dtype)[None, None, :]     # blocked hands stay zero
            # Update strategies
            na_pairs = tf.cast(tf.where(legal_mask), tf.int32)
            nodes = tf.gather(inactive_nodes, na_pairs[:, 0])
            strat_row_ids = tf.gather_nd(
                self.strat_idxs,
                tf.stack([nodes, na_pairs[:, 1]], axis=1)
            )
            self.strategies.scatter_nd_update(
                indices=tf.expand_dims(strat_row_ids, axis=1),
                updates=tf.gather_nd(strats, na_pairs)
            )
            # Update values
            self.values.scatter_nd_update(
                indices=tf.expand_dims(inactive_nodes, 1),
                updates=vals
            )

    def _update_values_impl(self):
        for node in tf.range(self.n_nodes-1, -1, -1):
            #
            # Active decision node
            #
            if self.node_types[node] == 0 and self.active_nodes[node]: 
                player = self.players[node]
                row = self.tree[node]
                children = tf.cast(tf.where(row != -1)[:, 0], dtype=tf.int32)
                actions = tf.gather(row, children)
                # Update player values
                node_strat_idxs = tf.gather(self.strat_idxs[node, :], actions)
                strats = tf.gather(self.strategies, node_strat_idxs)
                child_values = tf.gather(self.values[:, player, :], children)
                my_values = tf.einsum('ai,ai->i', strats, child_values)
                # Update opponent values
                opponent = (player + 1) % 2
                opp_values = tf.reduce_sum(
                    tf.gather(self.values[:, opponent, :], children),
                    axis=0
                )
                self.values.scatter_nd_update(
                    indices=tf.stack([tf.stack([node, player]), tf.stack([node, opponent])]),
                    updates=tf.stack([my_values, opp_values], axis=0)
                )
                # Update regrets
                old_regrets = tf.gather(self.regrets, node_strat_idxs)
                new_regrets = tf.maximum(old_regrets + child_values - my_values, 0.0)
                self.regrets.scatter_nd_update(tf.expand_dims(node_strat_idxs, 1), new_regrets)
                # Update strategy
                regret_sum = tf.reduce_sum(new_regrets, axis=0, keepdims=True)
                probs = tf.math.divide_no_nan(new_regrets, regret_sum)
                N_ACTIONS = tf.cast(tf.shape(actions)[0], tf.float32)
                zero_cols = tf.equal(regret_sum, 0.0)
                uniform = tf.fill(tf.shape(new_regrets), 1.0 / N_ACTIONS)
                updated_strats = tf.where(zero_cols, uniform, probs)
                updated_strats = tf.where(self.public_card_mask, 0., updated_strats)
                self.strategies.scatter_nd_update(tf.expand_dims(node_strat_idxs, 1), updated_strats)
                # TODO - Add cummulative strategy update here
            #
            # Non-showdown terminal node
            #
            elif self.node_types[node] == 1:
                winner = self.non_showdown_winner[node]
                loser = (winner + 1) % 2
                profit = self.put_in_pot[node, loser]
                # Other player's range as seen by [winner, loser]: row 0 is the loser's, row 1 the winner's
                other_ranges = tf.gather(self.ranges, tf.gather(self.range_map[node], [loser, winner]))   # (2, 1326)
                # Card removal: subtract the mass of the other player's hands that share a card with h
                card_mass  = tf.matmul(other_ranges, INVALID_HANDS_F32, transpose_b=True)                  # (2, 52)
                blocked    = (tf.gather(card_mass, HAND_CARD1, axis=1)
                              + tf.gather(card_mass, HAND_CARD2, axis=1) - other_ranges)                    # (2, 1326)
                compatible = tf.reduce_sum(other_ranges, axis=1, keepdims=True) - blocked                   # (2, 1326)
                keep_cols = tf.cast(~self.public_card_mask, dtype=tf.float32)
                signs = tf.constant([1.0, -1.0], dtype=self.values.dtype)
                updates = signs[:, None] * profit * compatible * keep_cols[None, :]
                indices = tf.stack([tf.fill([2], node), tf.stack([winner, loser], axis=0)], axis=1)
                self.values.scatter_nd_update(
                    indices,
                    updates
                )
            #
            # Showdown terminal node
            #
            elif self.node_types[node] == 2:
                #
                # self.payoffs[node, :, :]
                #    = (1326, 1326) matrix
                #
                # self.ranges[node, 1, :][np.newaxis, :]
                #    = (1, 1326) vector
                #
                # We multiply each row of the payoff matrix by the range vector,
                # then sum each row to get a (1326,) result
                #
                # NOTE - We divide the values by 1/2 because (for 2-players) each player puts
                #        in an equal amount of money. So the winning player profits 1/2 the pot
                #        and the losing player loses 1/2 the pot.
                #
                pot = tf.reduce_sum(self.put_in_pot[node])
                P = self.payoffs[self.payoffs_idxs[node], :, :]      # (1326, 1326)
                R = tf.gather(self.ranges, self.range_map[node, :])  # (2, 1326)
                vals = 0.5 * pot * tf.einsum('ij,pj->pi', P, R)      # (I, J) * (P, J) -> (P, I) = (2, 1326)
                self.values.scatter_nd_update(indices=[[node]], updates=[tf.gather(vals, [1, 0])])

    # Wrapper function
    def cfr_update(self, n_iters: int):
        # Select the correct update function
        if self.cfvn_enabled:
            self.cfr_update_w_cfvn(n_iters)
        else:
            self.cfr_update_full_tree(n_iters)

    #
    # Apply CFR updates
    #
    # Assumption - child nodes are always below their parents in the tree matrix
    #              i.e. child row id > parent row id
    #
    # TODO - Implement returning querries
    #
    def _cfr_update_w_cfvn_impl(self, n_iters: int):
        acc = tf.zeros_like(self.strategies)
        for i in tf.range(n_iters):
            # Downward pass - propagate range probabilities
            self.update_ranges()
            # Update non-active decision node values with the cfvn
            self.update_values_w_cfvn()
            # Upward pass - bubble up expected values
            self.update_values()
            # Update gadget game regrets (re-solving only; a fixed-belief solve keeps its root ranges)
            if self.use_gadget:
                self.update_gadget_regrets()
            # Accumulate strategies, linearly weighted by the iteration number over the whole solve
            weight = self.cfr_iters + tf.cast(i, tf.float32) + 1.0
            acc = acc + weight * self.strategies
        self.cum_strategies.assign_add(acc)
        self.cfr_iters.assign_add(tf.cast(n_iters, tf.float32))
        # Return a list of querries that were made to the cfvn
        return # NOTE - Not implemented
    
    def _cfr_update_full_tree_impl(self, n_iters: int):
        acc = tf.zeros_like(self.strategies)
        for i in tf.range(n_iters):
            # Downward pass - propagate range probabilities
            self.update_ranges()
            # Upward pass - bubble up expected values
            self.update_values()
            # Update gadget game regrets (re-solving only; a fixed-belief solve keeps its root ranges)
            if self.use_gadget:
                self.update_gadget_regrets()
            # Accumulate strategies, linearly weighted by the iteration number over the whole solve
            weight = self.cfr_iters + tf.cast(i, tf.float32) + 1.0
            acc = acc + weight * self.strategies
        self.cum_strategies.assign_add(acc)
        self.cfr_iters.assign_add(tf.cast(n_iters, tf.float32))



    #####################################
    #                                   #
    #          Grow Functions           #
    #                                   #
    ##################################### 

    #
    # Add a node to the game tree
    #
    # Which node to add is determined by sampling a hand configuration
    # weighted by the player's ranges at the root node, then actions are 
    # sampled down the tree until an action for which the resulting node 
    # is not activated. Then that node is activated.
    #
    # Returns whether the the attempt was successful or not.
    #     full tree = unsuccessful attempt
    #
    def grow(self, actor_hand: int = None) -> bool:
        #
        # --> STEP 1 - Assign hands to the players
        #
        #   * The acting player recieves their true hand assignemnt. 
        #
        #   * All other player's hand assignments are sampled at random, weighted 
        #     by their ranges at the decision point.
        #
        #   Edge case - If a given decision node is so bad for an opponent that
        #               their follow strategy is all zeros, then we can just
        #               sample a hand at random.
        #
        player_hands = np.zeros((self.n_players,), dtype=np.int32)
        valid_hands = tf.cast(~self.public_card_mask, dtype=tf.float32)
        # Give the acting player their true hand
        player = int(self.players[0]) # TODO - This should be the decision point (not neccisarily the root)
        if actor_hand is not None:
            #
            # Self-play: the acting player holds a known hand (a deliberate deviation from the
            # paper, which samples every hand from the root beliefs; see decision A12).
            #
            player_hands[player] = actor_hand
            actor_cards = list(get_2d_coors(actor_hand))
            hand_mask = tf.cast(~tf.reduce_any(tf.gather(invalid_hands, actor_cards), axis=0), dtype=tf.float32)
            valid_hands *= hand_mask
        #
        # Every player without an assigned hand (the opponent; the actor too in a fixed-belief
        # query solve) gets one sampled from their range at the root, as in the paper's simulations.
        #
        for opponent in range(self.n_players):
            # Skip players whose hand is already assigned
            if opponent == player and actor_hand is not None:
                continue
            # Opponent's range at the root, restricted to hands compatible with the cards already dealt
            probs = self.ranges[self.range_map[0, opponent], :] * valid_hands # TODO - Use decision point
            # Degenerate range (no mass on any compatible hand): fall back to uniform over compatible hands
            if tf.reduce_sum(probs) == 0.:
                probs = valid_hands / tf.reduce_sum(valid_hands)
            # tf.random.categorical takes LOG-probabilities; give zero-mass hands an effectively -inf logit
            very_neg = tf.constant(-1e9, tf.float32)
            masked_logits = tf.where(probs > 0.0, tf.math.log(tf.maximum(probs, 1e-30)), very_neg)
            # Sample
            hand = tf.random.categorical(masked_logits[tf.newaxis, :], num_samples=1)[0, 0]
            player_hands[opponent] = hand
            # DEBUG
            if not valid_hands[hand]:
                print("Invalid hand was sampled")
                import ipdb; ipdb.set_trace()
            # Update valid hands
            hand_mask = tf.cast(~tf.reduce_any(tf.gather(invalid_hands, list(get_2d_coors(int(hand)))), axis=0), dtype=tf.float32) # hands sharing a card with the sampled hand
            valid_hands *= hand_mask
        #
        # --> STEP 2 - Activate a node
        #
        #   * Traverse the tree according to PUCT statistics.
        #     See SOG paper, page 15.
        #
        #   * Activate the first inactive node incountered.
        #
        path = [0] # TODO - Use the decision point
        order_cache = {}
        attempted = set()
        while path:
            node = path[-1]
            player = self.players[node]
            hand = player_hands[player]
            row = self.tree[node]
            children = tf.reshape(tf.where(row != -1), -1)
            actions = tf.gather(row, children)
            # Compute a strategy weighted by PUCT scores
            if node in order_cache:
                action_order = order_cache[node]
            else:
                total_visits = tf.cast(self.visits[node], tf.float32)
                epsilon = tf.constant(1e-5, dtype=tf.float32)
                c_puct = tf.constant(1.0, dtype=tf.float32) # PUCT exploration parameter
                child_values = tf.gather(self.values[:, player, hand], children)
                child_visits = tf.cast(tf.gather(self.visits, children), tf.float32)
                cfr_strat = tf.gather(self.cum_strategies, tf.gather(self.strat_idxs[node, :], actions))[:, hand]
                cfr_strat = cfr_strat / tf.maximum(tf.reduce_sum(cfr_strat), 1e-12) # stored rows are weighted sums
                puct_scores = child_values + c_puct * (tf.math.sqrt(epsilon+total_visits) / (1 + child_visits)) * cfr_strat
                puct_strat = tf.nn.softmax(puct_scores)
                strat = 0.5 * (cfr_strat + puct_strat)
                # Determine the order of actions to attempt
                probs = strat.numpy()
                probs = probs / probs.sum()
                action_order = np.random.choice(
                    actions.numpy(), 
                    size=tf.shape(actions)[0].numpy(),
                    p=probs,
                    replace=False
                )
                order_cache[node] = action_order
            # Select the next node
            exhausted = True
            for action in action_order:
                try:
                    next_node = tf.where(row == action)[0, 0].numpy()
                except:
                    import ipdb; ipdb.set_trace()
                # Skip if we've already attempted this node
                if next_node in attempted:
                    continue
                # Skip if this node isn't a decision node
                if self.node_types[next_node] != 0:
                    continue
                # SUCCESS - Activate this node if it isn't active
                if not self.active_nodes[next_node]:
                    path.append(next_node)
                    idxs = tf.constant(path, dtype=tf.int32)
                    updates = tf.gather(self.visits, path) + tf.ones(len(path), dtype=tf.int32)
                    self.visits = tf.tensor_scatter_nd_update(self.visits, idxs[:, tf.newaxis], updates) # functional op: keep the result
                    self.activate(next_node)
                    return True
                # Else, select this node
                path.append(next_node)
                exhausted = False
                break
            # If this node's children are exhausted, then backtrack
            if exhausted:
                path.pop()
                attempted.add(node)
            continue
        # FAILURE - Tree is full
        return False



    #####################################
    #                                   #
    #       Debugging Functions         #
    #                                   #
    ##################################### 

    #
    # Helper function - Check that the values and strategies corresponding to hands
    #                   containing public cards are zero.
    #
    def check_public_cards(self):
        hands = tf.reshape(tf.where(self.public_card_mask), -1)
        # Check strategy
        if not tf.math.reduce_all(tf.gather(self.strategies, hands, axis=1) == 0.):
            invalid_hand_strats = tf.gather(self.strategies, hands, axis=1)
            print('NON-ZERO STRATEGY FOR AN INVALID HAND')
            print()
            print(invalid_hand_strats)
            print()
            print(tf.where(invalid_hand_strats != 0))
            import ipdb; ipdb.set_trace()
        # Check values
        if not tf.math.reduce_all(tf.gather(self.values, hands, axis=2) == 0.):
            invalid_hand_values = tf.gather(self.values, hands, axis=2)
            print('NON-ZERO VALUES FOR AN INVALID HAND')
            print()
            print(invalid_hand_values)
            print()
            print(tf.where(invalid_hand_values != 0))
            import ipdb; ipdb.set_trace()

    #
    # Helper function - Print the game tree, showing the internal state for a
    #                   selected hand.
    #
    # NOTE - We need to make this function more configurable.
    #        (i.e. able to designate both player's hands or neither's hand)
    #
    def print_tree(self, player: int, hand: int) -> None:
        # Validate the inputs
        root_game = self.game_states[0]
        card1_idx, card2_idx = get_2d_coors(hand)
        card1, card2 = Card(card_id=card1_idx), Card(card_id=card2_idx)
        assert not self.public_card_mask[hand], f'Given hand ({card1}, {card2}) contains a public card.'
        assert player < root_game.num_players, f'Invalid player: {player}'
        # Assign the given player the given hand
        root_game.players[player].hand = [card1, card2]
        # Print the header
        to_str = lambda cards: [str(card) for card in cards]
        print()
        print('--------------------------------------------------')
        print()
        print(f'Public cards: {to_str(root_game.public_cards)}')
        print()
        print(f'Pot: {root_game.dealer.pot} chips')
        print()
        print('Player 0')
        print(f'   Stack: {root_game.players[0].remained_chips} chips')
        print(f'   Hand: {to_str(root_game.players[0].hand)}')
        print()
        print('Player 1')
        print(f'   Stack: {root_game.players[1].remained_chips} chips')
        print(f'   Hand: {to_str(root_game.players[1].hand)}')
        print()
        print('--------------------------------------------------')
        # Construct the game tree graph
        import anytree
        root = anytree.Node(name='0', value=0) # node.value = node id
        q: list[anytree.Node]
        q = [root]
        while q:
            parent_node = q.pop()
            if self.node_types[parent_node.value] == 0: # Decision node
                row = self.tree[parent_node.value]
                children = tf.reshape(tf.where(row != -1), -1)
                actions = tf.gather(row, children)
                for i in range(tf.shape(children)[0]):
                    child_node = anytree.Node(name=actions[i].numpy(), value=children[i].numpy(), parent=parent_node)
                    if self.active_nodes[children[i]]:
                        q.append(child_node)
        # Print the game tree
        hand_to_int = lambda hand: get_1d_coor(*sorted([hand[0].to_int(), hand[1].to_int()]))
        p0_hand = hand_to_int(root_game.players[0].hand)
        p1_hand = hand_to_int(root_game.players[1].hand)
        for pre, fill, node in anytree.RenderTree(root):
            my_pid = self.players[node.value]
            
            if node is root:
                print(f"{fill}")
                print(f"--> P{my_pid} Decision")
            else:
                parent = node.parent
                
                action_idx = self.tree[parent.value, node.value]
                action = Action(self.all_actions[action_idx])
                
                parent_pid = self.players[parent.value]
                
                if parent_pid == 0:
                    strat = self.strategies[self.strat_idxs[parent.value, action_idx], p0_hand]
                    regret = self.regrets[self.strat_idxs[parent.value, action_idx], p0_hand]
                else:
                   strat = self.strategies[self.strat_idxs[parent.value, action_idx], p1_hand]
                   regret = self.regrets[self.strat_idxs[parent.value, action_idx], p1_hand]
                
                print(f"{pre} ({action} : strat = {strat}, regret = {regret})")
                print(f"{fill}")
                
                if self.node_types[node.value] == 0:
                     print(f"{fill} --> ({node.value}) P{my_pid} Decision")
                elif self.node_types[node.value] in (1, 2):
                    print(f"{fill} --> ({node.value}) Terminal Node")
                else:
                    raise ValueError("Unrecognized node type")
            
            p0_reach_prob = self.ranges[self.range_map[node.value, 0], p0_hand]
            p1_reach_prob = self.ranges[self.range_map[node.value, 1], p1_hand]

            p0_value = self.values[node.value, 0, p0_hand]
            p1_value = self.values[node.value, 1, p1_hand]

            print(f"{fill}")
            print(f"{fill} P0 reach prob = {p0_reach_prob}")
            print(f"{fill} P1 reach prob = {p1_reach_prob}")
            print(f"{fill}")
            print(f"{fill} P0 value = {p0_value}")
            print(f"{fill} P1 value = {p1_value}")
            print(f"{fill}") 
        import ipdb; ipdb.set_trace()    





#####################################
#                                   #
#     Payoff Helper Functions       #
#                                   #
##################################### 

#
# Helper function - Returns a unique string for the given
#                   5 card list, independent of card order.
#
def five_cards_to_str(board: list[Card]) -> str:
    assert len(board) == 5, "Only compute payoffs for river showdowns"
    output = ''
    for card in sorted(board, key=lambda c: c.to_int()):
        output += str(card)
    return output
                    
#
# Helper function - Computes the payoff matrix for a given
#                   terminal showdown game.
#
def compute_payoff_matrix(game: NolimitholdemGame) -> np.ndarray:
    #
    # Deck of trey cards
    #
    deck = sorted(init_standard_deck(), key=lambda x: x.to_int()) 
    trey_deck = [card.to_treys() for card in deck]
    community_cards = [card.to_treys() for card in game.public_cards]
    community_idxs = [card.to_int() for card in game.public_cards]

    #
    # Useful for converting from 1d coor to 2d corrs
    #
    # upper_i[hand_idx] = card1 idx
    # upper_j[hand_idx] = card2 idx
    #
    upper_i, upper_j = np.triu_indices(52, k=1)

    #
    # Get a vector of hand ranks
    #
    evaluator = treys.Evaluator()
    hand_evals = np.ones(1326) * np.inf
    for hand in range(1326):
        x, y = upper_i[hand], upper_j[hand]
        if x in community_idxs or y in community_idxs:
            continue
        hand_evals[hand] = evaluator.evaluate(community_cards, [trey_deck[x], trey_deck[y]])

    # Vectorized version of get_hand_payoff
    def get_hand_payoff(hand1, hand2):
        """
        Compute the payoff of a poker hand matchup in a vectorized manner.
        """
        # Get the 2d coors for the hand idxs
        h1_card1, h1_card2 = upper_i[hand1], upper_j[hand1]
        h2_card1, h2_card2 = upper_i[hand2], upper_j[hand2]

        # Check 1: Card overlap between hands
        has_overlap = (
            (h1_card1 == h2_card1) |
            (h1_card1 == h2_card2) |
            (h1_card2 == h2_card1) |
            (h1_card2 == h2_card2)
        )

        # Check 2: Community cards
        has_community_cards = (
            np.isin(h1_card1, community_idxs) |
            np.isin(h1_card2, community_idxs) |
            np.isin(h2_card1, community_idxs) |
            np.isin(h2_card2, community_idxs)
        )

        # Valid hands pass both checks
        valid_hands = ~(has_overlap | has_community_cards)


        # Compare hand strengths using precomputed `hand_evals`
        # NOTE - not using equality here, in the case of draws we want the payoff to remain zero.
        player_wins = hand_evals[hand1] < hand_evals[hand2]
        player_loses = hand_evals[hand1] > hand_evals[hand2]

        # Initialize payoff vector
        payoffs = np.zeros_like(hand1, dtype=np.float64)  # Default all payoffs to 0

        # Assign the winning hands 1s and losing hands -1s
        payoffs = np.where(
            player_wins & valid_hands, 
            1,
            payoffs
        )

        payoffs = np.where(
            player_loses & valid_hands, 
            -1,
            payoffs
        )

        return payoffs

    #
    # Define shape
    #
    shape = [1326] * game.num_players

    #
    # Apply vectorized function
    #
    # Note - np.fromfunction applies the function to the indices of the result array.
    #
    #        So, payoffs[i, j, k, l] = get_hand_payoff(i, j, k, l)
    #
    #        where payoffs.shape = shape (as defined above)
    #
    payoffs = np.fromfunction(lambda hand1, hand2: get_hand_payoff(hand1.astype(int),
                                                                   hand2.astype(int)), 
                                                                   shape, dtype=np.float64)
    
    return payoffs