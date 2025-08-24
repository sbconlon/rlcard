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
    def __init__(self, cfvn: CounterfactualValueNetwork, n_players: int=2):
        #
        # Store the counterfactual value network
        #
        self.cfvn = cfvn
        self.cfvn_enabled = tf.constant(bool(self.cfvn), dtype=tf.bool)
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
        self.vect_idxs = tf.constant([], dtype=tf.int32, shape=(0,))
        #
        # Vectorized decision node prefixs (contain all non-range elements)
        #
        #    - Matrix of size = (Approx. # of non-active decision nodes, # of prefix-features)
        #
        #    - feat_vect_prefixs[vector_idxs[i]] = feature vector prefix for node i
        #
        self.feat_vect_prefixs = tf.constant([], dtype=tf.float32, shape=(0, 56))
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



    #####################################
    #                                   #
    #            Getters                #
    #                                   #
    #####################################

    def get_strategy(self, node: int) -> np.ndarray:
        res = tf.gather(self.strategies, self.strat_idxs[node, :]).numpy()
        return res[self.legal_actions[node], :]
    
    def get_cum_strategy(self, node: int) -> np.ndarray:
        res = tf.gather(self.cum_strategies, self.strat_idxs[node, :]).numpy()
        return res[self.legal_actions[node], :]
    
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
            self.terminate_values = input_opponents_values
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
        self.vect_idxs = tf.constant([-1], dtype=tf.int32) # Root node never inactive
        self.put_in_pot = tf.constant([p.in_chips for p in game_state.players], dtype=tf.float32)[None, :]
        self.non_showdown_winner = tf.constant([-1], dtype=tf.int32)
        root_strat_idxs = [-1] * len(self.all_actions)
        for idx, action in enumerate(legal_actions):
            root_strat_idxs[action] = idx
        self.strat_idxs = tf.constant([root_strat_idxs], dtype=tf.int32)
        self.payoffs_idxs = tf.constant([-1], dtype=np.int32)
        #
        # Increment node count
        #
        self.n_nodes += 1
        #
        # Initialize gadget game
        #
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
        self.active_nodes = tf.concat([self.active_nodes, tf.constant([child_type == 1], dtype=tf.bool)], axis=0)
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
            indices = tf.stack( # [[child_id, action1], [child_id, action2], ...] NOTE - I think we can replace this
                [tf.fill((len(legal_actions),), child_id), legal_actions],
                axis=1
            )
            vals = base + tf.range(len(legal_actions))
            self.strat_idxs = tf.tensor_scatter_nd_update(self.strat_idxs, indices, vals)
            # Feature vector
            if self.cfvn:
                if self.feat_vect_prefixs is not None:
                    self.feat_vect_prefixs: tf.Tensor
                    self.vect_idxs = tf.concat([self.vect_idxs, [self.feat_vect_prefixs.shape[0]]], axis=0)
                    self.feat_vect_prefixs = tf.concat(
                        [self.feat_vect_prefixs, [self.cfvn.get_prefix(child_game)]],
                        axis=0
                    )
                else:
                    self.vect_idxs = tf.constant([0], dtype=tf.int32)
                    self.feat_vect_prefixs = tf.expand_dims(self.cfvn.get_prefix(child_game), axis=0)
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
        # Update node count
        #
        self.n_nodes += 1
    
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
    # Update the regrets in the gadget game
    #
    # NOTE - implement this function for >2 player games, shouldn't be too hard
    #
    @tf.function(jit_compile=True)
    def update_gadget_regrets(self):
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
        gadget_follow_strat = tf.where(denom == 0.0, 0.5, probs)
        #
        # In the above line, we assign 50-50 probability to hands with zero in the denominator.
        # This works for valid hands, but has the side effect of giving invalid hands non-zero
        # reach probabilities.
        #
        # Mask out invalid hands
        #
        gadget_follow_strat = tf.where(self.public_card_mask, 0.0, gadget_follow_strat)
        gadget_follow_strat = tf.math.divide_no_nan(gadget_follow_strat, tf.reduce_sum(gadget_follow_strat))
        #
        # Set the opponent's range in the cfr root node to the gadget's follow strategy 
        #
        # Reasoning:
        #     Gadget follow strat = probability of choosing to play toward the cfr root state
        #     Opp. root range = prob. of the opp. reaching this state given her strategy
        #     Therefore, gadget follow strat = opponent's range at the root node.
        #
        # Normalize the opponent's reach probibility.
        #
        # Note 1 - while this step is not included in deepstack's pseudocode,
        #          I think it fits the reasoning of the gadget game - the initial
        #          state of the sub game can be thought of as the start of a game
        #          where the two player's are dealt cards from a weighted deck.
        #          Using this reasoning, the follow strat should be normalized.
        #
        # Note 2 - I came to this conclusion after noticing the expected values
        #          blow up because we are giving a 1.0 reach probability for the
        #          opponent for good hands and 0.0 reach probability for bad hands.
        #          When this is propagated down to terminal nodes, it leads to
        #          expected values greater than the total number of chips in the game.
        #
        # Note 3 - I don't think multiplying all reach probabilities by a constant
        #          will affect the output strategy.
        #
        # Note 4 - Bayes' perspective:
        #          Un-normalized = prob opp. player reaches the root state given they have the hand (i, j)
        #          Normalized    = prob opp. player reaches the root state and has the hand (i, j)
        #
        opp_pid = tf.cast((self.players[0] + 1) % 2, tf.int32)
        range_id = tf.cast(self.range_map[0, opp_pid], tf.int32)
        self.ranges.scatter_nd_update(
            indices=tf.reshape(range_id, [1, 1]), 
            updates=tf.reshape(gadget_follow_strat, [1, -1])
        )
        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        opp_vals = self.values[0, opp_pid, :]
        new_gadget_values = (gadget_follow_strat * opp_vals + 
                             (1.0 - gadget_follow_strat) * self.terminate_values)
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
        self.gadget_values = new_gadget_values

    # NOTE - We can further improve this function by applying the range updates on 
    #        a per level basis in parallel.
    @tf.function(jit_compile=True)
    def update_ranges(self):
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

    @tf.function(jit_compile=True)
    def update_values_w_cfvn(self):
        # Get the batch of featurized game states
        inactive_nodes = tf.cast(tf.where((self.node_types == 0) & (~self.active_nodes))[:, 0], tf.int32) # non-active decision nodes
        N = tf.shape(inactive_nodes)[0]
        rm = tf.gather(self.range_map, inactive_nodes)
        ranges_block = tf.gather(self.ranges, rm)
        ranges_flat  = tf.reshape(ranges_block, [N, -1])
        vects = tf.concat(
            [
                tf.gather(self.feat_vect_prefixs, tf.gather(self.vect_idxs, inactive_nodes)),
                ranges_flat
            ], 
            axis=1
        )
        # Query the cfvn
        strats, vals = self.cfvn.query(vects)
        # Update strategies
        legal_mask = tf.gather(self.legal_actions, inactive_nodes)
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

    @tf.function(jit_compile=True)
    def update_values(self):
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
                range_sum = tf.reduce_sum(tf.gather(self.ranges, tf.gather(self.range_map[node], [loser, winner])), axis=1)
                keep_cols = tf.cast(~self.public_card_mask, dtype=tf.float32)
                signs = tf.constant([1.0, -1.0], dtype=self.values.dtype)
                updates = signs[:, None] * profit * range_sum[:, None] * tf.stack([keep_cols, keep_cols], axis=0)
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
        # Initialize the cummulative strategies
        self.cum_strategies = tf.Variable(tf.zeros(tf.shape(self.strategies), dtype=self.strategies.dtype))
        # Select the correct update function
        if self.cfvn_enabled:
            self.cfr_update_w_cfvn(n_iters)
        else:
            self.cfr_update_full_tree(n_iters)
        # Finish the cummulative strategies
        self.cum_strategies.assign(tf.divide(self.cum_strategies, n_iters))

    #
    # Apply CFR updates
    #
    # Assumption - child nodes are always below their parents in the tree matrix
    #              i.e. child row id > parent row id
    #
    # TODO - Implement returning querries
    #
    @tf.function(jit_compile=True)
    def cfr_update_w_cfvn(self, n_iters: int):
        for _ in tf.range(n_iters):
            #
            # Downward pass - propagate range probabilities
            #
            #strt = time.time()
            self.update_ranges()
            #print(f'update_ranges {time.time() - strt} s')
            #
            # Update non-active decision node values with the cfvn
            #
            #strt = time.time()
            self.update_values_w_cfvn()
            #print(f'update_values_w_cfvn {time.time() - strt} s')
            #
            # Upward pass - bubble up expected values
            #
            #strt = time.time()
            self.update_values()
            #print(f'update_values {time.time() - strt} s')
            # Update gadget game regrets
            #strt = time.time()
            self.update_gadget_regrets()
            #print(f'update_gadget_regrets {time.time() - strt} s')
        # Return a list of querries that were made to the cfvn
        return # NOTE - Not implemented
    
    @tf.function(jit_compile=True)
    def cfr_update_full_tree(self, n_iters: int):
        acc = tf.zeros_like(self.strategies)
        for _ in tf.range(n_iters):
            #self.print_tree(player=0, hand=863) # 863 = 9H QH
            # Downward pass - propagate range probabilities
            #self.check_public_cards()
            self.update_ranges()
            # Upward pass - bubble up expected values
            #self.check_public_cards()
            self.update_values()
            # Update gadget game regrets
            #self.check_public_cards()
            self.update_gadget_regrets()
            acc = acc + self.strategies
        self.cum_strategies.assign_add(acc)



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