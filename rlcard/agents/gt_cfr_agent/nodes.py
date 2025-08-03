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
import cupy as cp
import treys

# Internal imports
from rlcard.games.base import Card
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.utils import random_strategy, get_1d_coor, get_card_coors, starting_hand_values
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
        #
        # Store the number of players and actions
        #
        self.n_players = n_players
        self.all_actions = []
        #
        # Counts the number of nodes in the tree
        #
        self.n_nodes = 0
        #
        # Game tree
        #
        #   - Adjacency matrix
        #
        #   - tree[i, j] = action id if node j is a descendant of node i otherwise -1
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
        self.tree = cp.array([], dtype=np.int8)
        #
        # Node types
        #
        #   - Vector of length = # of nodes
        #
        #   - node_types[i] = 0 if node i is a decision node
        #                     1 if node i is a terminal node
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
        self.node_types = cp.array([], dtype=np.int8)
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
        # Players
        #
        #    - Vector of length = # of nodes
        #
        #    - players[i] = player id of the acting player if node i is a decision node
        #                   -1 if node i is a terminal node
        #
        self.players = cp.array([], dtype=np.int8)
        #
        # Activate nodes
        #
        #    - Vector of length = # of nodes
        #
        #    - active_nodes[i] = boolean whether node i is active
        #
        self.active_nodes = cp.array([], dtype=np.bool_)
        #
        # Range map
        #
        #   - Matrix of size (# nodes, # of players)
        #
        #   - range_map[i, j] = idx for the range vector associated with 
        #                       player j at node i.
        #
        self.range_map = cp.array([], dtype=np.int8)
        #
        # Ranges
        #
        #   - Matrix of size (# of decision nodes + # of players, # of hands)
        #
        #   - NOTE: 1326 is the number of distinct hands
        #
        #   - ranges[range_map[i, j], k] = prob. of player j reaching node i with hand k
        #
        self.ranges = cp.array([], dtype=np.float32)
        #
        # Values
        #
        #   - Matrix of size (# of nodes, # of player, # of hands)
        #
        #   - values[i, j, k] = expected value of player j having hand k in node i
        #
        self.values = cp.array([], dtype=np.float32)
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
        self.strat_idxs = cp.array([[]], dtype=np.int8)
        #
        # Strategies
        #
        #    - Matrix size = (Approx. # of decision nodes * # of actions, # of hands)
        #
        #    - strategies[strat_idxs[i], j, k] 
        #         = prob. of player at node i selecting action j with hand k
        #
        self.strategies = cp.array([[]], dtype=np.float32)
        #
        # Regrets
        #
        #    - Matrix size = (Approx. # of decision nodes * # of actions, # of hands)
        # 
        #    - regrets[strat_idxs[i], j, k]
        #          = regret of player at node i selecting action j with hand k 
        #
        self.regrets = cp.array([[]], dtype=np.float32)
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
        self.payoffs_idxs = cp.array([], dtype=np.int8)
        #
        # Payoffs
        #
        #   - Matrix size = (# of distinct payoffs, # of players, # of hands, # of hands)
        #
        #   - payoffs[payoffs_idxs[x], i, j, k]
        #       = payoff to player i at node x when holding hand j against hand k
        #
        self.payoffs = cp.array([], dtype=np.float32)

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
            self.terminate_values = cp.array(input_opponents_values)
        else:
            self.terminate_values = cp.array(starting_hand_values(root_game)) * root_game.dealer.pot # = t_values = v_2 in the literature
        self.gadget_regrets = cp.zeros((2, 1326)) # 2 gadget actions, (Follow, Terminate)
        self.gadget_values = cp.zeros((1326,))
    
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
        # Set internal matrices
        #
        self.tree = cp.array([[-1]], dtype=cp.int8) # nodes are never descendants of themselves
        self.node_types = cp.array([0], dtype=cp.int8) # all root nodes are decision nodes
        self.players = cp.array([game_state.game_pointer], dtype=cp.int8)
        self.game_states.append(game_state)
        self.active_nodes = cp.array([False], dtype=cp.bool_)
        self.range_map = cp.array([cp.arange(self.n_players)], dtype=cp.int8)
        self.ranges = cp.array(player_ranges)
        self.values = cp.zeros((1, self.n_players, 1326))
        legal_actions = [self.all_actions.index(action.value) for action in game_state.get_legal_actions()]
        self.strategies = cp.array(random_strategy(len(legal_actions), game_state.public_cards))
        self.regrets = cp.zeros((len(legal_actions), 1326))
        root_strat_idxs = [-1] * len(self.all_actions)
        for idx, action in enumerate(legal_actions):
            root_strat_idxs[action] = idx
        self.strat_idxs = cp.array([root_strat_idxs], dtype=np.int8)
        self.payoffs_idxs = cp.array([-1], dtype=np.int8)
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
        child_game = copy.deepcopy(parent_game)
        child_game.step(action)
        #
        # Determine the node type for the child
        #
        child_type = None
        if child_game.is_over():
            child_type = 1 # Terminal node
        elif (parent_game.stage != parent_game.stage and 
            not child_game.stage in (Stage.END_HIDDEN, Stage.SHOWDOWN)):
            child_type = 2 # Chance node
        elif child_game.stage in (Stage.PREFLOP, Stage.FLOP, Stage.TURN, Stage.RIVER):
            child_type = 0 # Decision node
        #
        # Add child to the tree matrix
        #
        new_column = cp.ones((self.n_nodes, 1), dtype=cp.int8) * -1
        self.tree = cp.hstack([self.tree, new_column])
        new_row = cp.ones(self.n_nodes+1, dtype=np.int8) * -1
        self.tree = cp.vstack([self.tree, new_row])
        self.tree[parent_id, self.n_nodes] = self.all_actions.index(action_id)
        #
        # Add child to node types and players
        #
        self.node_types = cp.append(self.node_types, child_type)
        if child_type == 0:
            self.players = cp.append(self.players, child_game.game_pointer)
        else:
            self.players = cp.append(self.players, -1)
        #
        # Add child to game states
        #
        self.game_states.append(child_game)
        #
        # Add child to active nodes
        #
        # Note - terminal nodes are always considered active
        #
        self.active_nodes = cp.append(self.active_nodes, child_type == 1)
        #
        # Add child range
        #
        child_id = self.n_nodes
        player_id = self.players[child_id]
        action_idx = self.all_actions.index(action_id)
        child_range_idxs = self.range_map[parent_id]
        self.range_map = cp.vstack([self.range_map, child_range_idxs])
        parent_range_idx = self.range_map[parent_id, player_id]
        parent_strat_idx = self.strat_idxs[parent_id, action_idx]
        child_range = self.strategies[parent_strat_idx] * self.ranges[parent_range_idx]
        self.ranges = cp.vstack([self.ranges, child_range])
        self.range_map[child_id, player_id] = self.ranges.shape[0] - 1
        #
        # Add child values
        #
        child_values = cp.zeros((1, self.n_players, 1326))
        self.values = cp.concatenate([self.values, child_values], axis=0)
        #
        # Add strategy
        #
        self.strat_idxs = cp.vstack([self.strat_idxs, [-1]*len(self.all_actions)])
        if child_type == 0: # Decision node
            legal_actions = [self.all_actions.index(action.value) for action in child_game.get_legal_actions()]
            child_strategy = cp.array(random_strategy(len(legal_actions), child_game.public_cards))
            base = self.strategies.shape[0]
            self.strategies = cp.concatenate([self.strategies, child_strategy], axis=0)
            self.regrets = cp.concatenate([self.regrets, cp.zeros((len(legal_actions), 1326))], axis=0)
            for offset, action in enumerate(legal_actions):
                self.strat_idxs[child_id, action] = base + offset
        #
        # Add payoffs
        #
        if child_type == 1: # Terminal node
            num_active = sum([p.status != PlayerStatus.FOLDED for p in child_game.players])
            if num_active > 1: # Showdown
                board_str = five_cards_to_str(child_game.public_cards)
                if board_str in self.board_to_idx: # Cache hit
                    self.payoffs_idxs = cp.append(self.payoffs_idxs, self.board_to_idx[board_str])
                else: # Cache miss
                    payoff_matrix = compute_payoff_matrix(child_game)
                    payoff_matrix = cp.expand_dims(payoff_matrix, axis=0)
                    if self.payoffs:
                        self.payoffs = cp.concatenate([self.payoffs, payoff_matrix], axis=0)
                    else:
                        self.payoffs = payoff_matrix
                    self.payoffs_idxs = cp.append(self.payoffs_idxs, self.payoffs.shape[0]-1)
                    self.board_to_idx[board_str] = self.payoffs_idxs[child_id]
            else: # No showdown
                self.payoffs_idxs = cp.append(self.payoffs_idxs, -1)
        else: # None terminal node
            self.payoffs_idxs = cp.append(self.payoffs_idxs, -1)
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
        self.active_nodes[node_id] = True
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
        for child_id in cp.where(self.tree[node_id, :] >= 0)[0]:
            if self.node_types[child_id] == 0 and not self.active_nodes[child_id]:
                self.activate(int(child_id))

    #
    # Update the regrets in the gadget game
    #
    # NOTE - implement this function for >2 player games, shouldn't be too hard
    #
    def update_gadget_regrets(self):
        
        """
        # DEBUG
        print('---')
        opp_pid = (self.root.game.game_pointer + 1) % 2
        card1, card2 = sorted(card.to_int() for card in self.root.game.players[opp_pid].hand)
        #print([str(card) for card in self.root.game.players[opp_pid].hand])
        """
        
        # Game state for the root
        root_game = self.game_states[0]

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
        gadget_regrets_positives = cp.maximum(self.gadget_regrets, 0) # Should already be non-negative

        """
        # DEBUG
        print()
        print(f'Regrets - Follow: {gadget_regrets_positives[0, card1, card2]}  Terminate: {gadget_regrets_positives[1, card1, card2]}')
        print(f'Value  -  {self.gadget_values[card1, card2]}')
        """

        denom = gadget_regrets_positives[0] + gadget_regrets_positives[1]
        safe_denom = cp.where(denom == 0, 1, denom) # remove zeros from denom to avoid dividing by zero
        gadget_follow_strat = cp.where(denom == 0, 0.5, gadget_regrets_positives[0] / safe_denom)

        """
        # DEBUG
        print(f'Follow strat - {gadget_follow_strat[card1, card2]}')
        """

        #
        # In the above line, we assign 50-50 probability to hands with zero in the denominator.
        # This works for valid hands, but has the side effect of giving invalid hands non-zero
        # reach probabilities.
        #
        # Mask out invalid hands
        #
        for card in root_game.public_cards:
            gadget_follow_strat[get_card_coors(card.to_int())] = 0

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
        opp_pid = (self.players[0] + 1) % 2
        if cp.sum(gadget_follow_strat) != 0:
            self.ranges[self.range_map[0, opp_pid], :] = gadget_follow_strat / cp.sum(gadget_follow_strat)
        else:
            self.ranges[self.range_map[0, opp_pid], :] = gadget_follow_strat # ALL ZEROS

        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        new_gadget_values = (gadget_follow_strat * self.values[0, opp_pid, :] + 
                             (1 - gadget_follow_strat) * self.terminate_values)

        """
        # DEBUG
        print()
        print(f'New value - follow strat * root value + (1 - follow strat) * terminate value = {gadget_follow_strat[card1, card2]} * {self.root.values[opp_pid, card1, card2]} + {1 - gadget_follow_strat[card1, card2]} * {self.terminate_values[card1, card2]} = {new_gadget_values[card1, card2]}')

        # DEBUG
        print()
        print(f'Follow regret update = max(regret + node value - new gadget value, 0) = max({self.gadget_regrets[0, card1, card2]} + {self.root.values[opp_pid, card1, card2]} - {new_gadget_values[card1, card2]}, 0) = {np.maximum(self.gadget_regrets[0]  + self.root.values[opp_pid] - new_gadget_values, 0)[card1, card2]}')
        print(f'Terminate regret update = max(regret + terminate value - gadget value, 0) = max({self.gadget_regrets[1, card1, card2]} + {self.terminate_values[card1, card2]} - {self.gadget_values[card1, card2]}, 0) = {np.maximum(self.gadget_regrets[1] + self.terminate_values     - self.gadget_values, 0)[card1, card2]}')
        """

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
        self.gadget_regrets[0] = cp.maximum(self.gadget_regrets[0] + self.values[0, opp_pid, :] - new_gadget_values, 0) # gadget value @ t
        self.gadget_regrets[1] = cp.maximum(self.gadget_regrets[1] + self.terminate_values - self.gadget_values, 0)    # gadget value @ t + 1

        #
        # Update the gadget values to the new values
        #
        self.gadget_values = new_gadget_values
    
    def update_ranges(self):
        for parent in range(self.n_nodes):
            player = self.players[parent]
            for child in cp.where(self.tree[parent, :] >= 0)[0]:
                action = self.tree[parent, child]
                self.ranges[self.range_map[child, player], :] = self.ranges[self.range_map[parent, player], :] * self.strategies[self.strat_idxs[parent, action], :]
    
    def update_values_w_cfvn(self):
        inactive_nodes = cp.where((self.node_types == 0) & (~self.active_nodes))[0] # non-active decision nodes
        batch_vects = cp.zeros((inactive_nodes.shape[0], 2709)) # (num. of inactive nodes, cfvn input vector size)
        batch_actions = []
        # TODO - vectorize this loop
        for batch_idx, node in enumerate(inactive_nodes):
            batch_vects[batch_idx, :] = self.cfvn.to_vect(
                        self.game_states[int(node)], 
                        self.ranges[self.range_map[node, :], :]
            )
            # NOTE - I hate this
            # TODO - Rework how we handle actions
            batch_actions.append([self.all_actions.index(a.value)
                                  for a in self.game_states[int(node)].get_legal_actions()])

        strats, vals = self.cfvn.query(batch_vects)
        strats, vals = cp.array(strats), cp.array(vals)
        # TODO - vectorize this loop
        for batch_idx, node in enumerate(inactive_nodes):
            self.strategies[self.strat_idxs[node, batch_actions[batch_idx]], :] = strats[batch_idx, batch_actions[batch_idx], :]
        self.values[inactive_nodes, :, :] = vals

    def update_values(self):
        for node in range(self.n_nodes-1, -1, -1):
            # Active decision node
            if self.node_types[node] == 0 and self.active_nodes[node]: 
                player = self.players[node]
                actions = self.tree[node, cp.where(self.tree[node] != -1)][0]
                # Update player
                strat = self.strategies[self.strat_idxs[player, actions], :] # shape = (num. of actions, 1326)
                children = cp.where(self.tree[node] != -1)
                child_values = self.values[children, player, :][0] # shape = (num. of actions, 1326)
                self.values[node, player, :] = cp.sum(strat * child_values, axis=0)
                # Update opponent
                opponent = (player + 1) % 2
                self.values[node, opponent, :] = cp.sum(self.values[children, opponent, :], axis=1)
                # Update regrets
                self.regrets[self.strat_idxs[player, actions], :] = cp.maximum(
                    self.regrets[self.strat_idxs[player, actions], :] + child_values - self.values[node, opponent, :],
                    0
                )
                # Update strategy
                regret_sum = cp.sum(self.regrets[self.strat_idxs[player, actions], :], axis=0, keepdims=True)
                denom = cp.where(regret_sum == 0, 1, regret_sum)
                self.strategies[self.strat_idxs[player, actions], :] = cp.where(
                    regret_sum == 0,
                    1/actions.shape[0],
                    self.regrets[self.strat_idxs[player, actions], :]/denom
                ) # NOTE - This might be wrong?
                # TODO - Add cummulative strategy update here
            # Terminal node
            elif self.node_types[node] == 1:
                num_active = sum([p.status != PlayerStatus.FOLDED 
                                  for p in self.game_states[node].players]) # NOTE - This is bad
                # Non-showdown
                if num_active != 1:
                    payout = cp.array(self.game_states[node].get_payoffs()) # NOTE - This is bad
                    self.values[node, :, :] = payout[:, None] * cp.sum(self.ranges[self.range_map[node, :], :], axis=1)[:, None]
                # Showdown
                else:
                    #
                    # self.payoffs[node, 0, :, :]
                    #    = (1326, 1326) matrix
                    #
                    # self.ranges[node, 1, :][np.newaxis, :]
                    #    = (1, 1326) matrix
                    #
                    # We multiply each row of the payoff matrix by the range vector,
                    # then sum each row to get (1326,) result
                    #
                    pot = self.game_states[node].dealer.pot
                    self.values[node, 0, :] = 0.5 * pot * cp.sum(
                        self.payoffs[self.payoffs_idxs[node], 0, :, :] *
                        self.ranges[self.range_map[node, 1], :][np.newaxis, :],
                        axis=1
                    )
                    self.values[node, 1, :] = 0.5 * pot * cp.sum(
                        self.payoffs[self.payoffs_idxs[node], 1, :, :] *
                        self.ranges[self.range_map[node, 0], :][cp.newaxis, :],
                        axis=1
                    )



    #
    # One iteration of CFR
    #
    # Assumption - child nodes are always below their parents in the tree matrix
    #              i.e. child row id > parent row id
    #
    # TODO - Implement returning querries
    #
    def cfr_update(self) -> list[np.ndarray]:
        # Downward pass - propagate range probabilities
        #start = cp.cuda.Event(); end = cp.cuda.Event()
        #start.record()
        self.update_ranges()
        #end.record(); end.synchronize()
        #print("update_ranges took", cp.cuda.get_elapsed_time(start, end), "ms")
        # Update non-active decision node values with the cfvn
        #start = cp.cuda.Event(); end = cp.cuda.Event()
        #start.record()
        self.update_values_w_cfvn()
        #end.record(); end.synchronize()
        #print("update_values_w_cfvn took", cp.cuda.get_elapsed_time(start, end), "ms")
        # Upward pass - bubble up expected values
        #start = cp.cuda.Event(); end = cp.cuda.Event()
        #start.record()
        self.update_values()
        #end.record(); end.synchronize()
        #print("update_values took", cp.cuda.get_elapsed_time(start, end), "ms")
        # Update gadget game regrets
        #start = cp.cuda.Event(); end = cp.cuda.Event()
        #start.record()
        self.update_gadget_regrets()
        #end.record(); end.synchronize()
        #print("update_gadget_regrets took", cp.cuda.get_elapsed_time(start, end), "ms")
        #print(f'update_gadget_regrets {time.time() - strt} s')
        # Return a list of querries that were made to the cfvn
        return [] # NOTE - Not implemented
            
                    

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
    community_idxs = cp.array([card.to_int() for card in game.public_cards])

    #
    # Useful for converting from 1d coor to 2d corrs
    #
    # upper_i[hand_idx] = card1 idx
    # upper_j[hand_idx] = card2 idx
    #
    upper_i, upper_j = cp.triu_indices(52, k=1)

    #
    # Get a vector of hand ranks
    #
    evaluator = treys.Evaluator()
    hand_evals = cp.ones(1326) * cp.inf
    for hand in range(1326):
        x, y = upper_i[hand], upper_j[hand]
        if x in community_idxs or y in community_idxs:
            continue
        hand_evals[hand] = evaluator.evaluate(community_cards, [trey_deck[int(x)], trey_deck[int(y)]])

    # Vectorized version of get_hand_payoff
    def get_hand_payoff(pid, hand1, hand2):
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
            cp.isin(h1_card1, community_idxs) |
            cp.isin(h1_card2, community_idxs) |
            cp.isin(h2_card1, community_idxs) |
            cp.isin(h2_card2, community_idxs)
        )

        # Valid hands pass both checks
        valid_hands = ~(has_overlap | has_community_cards)


        # Compare hand strengths using precomputed `hand_evals`
        # NOTE - not using equality here, in the case of draws we want the payoff to remain zero.
        player1_wins = hand_evals[hand1] < hand_evals[hand2]
        player2_wins = hand_evals[hand1] > hand_evals[hand2]

        # Initialize payoff vector
        payoffs = cp.zeros_like(hand1, dtype=cp.float64)  # Default all payoffs to 0

        # Assign the winning hands 1s and losing hands -1s
        payoffs = cp.where(
            player1_wins & valid_hands, 
            cp.where(pid==0, 1, -1),
            payoffs
        )

        payoffs = cp.where(
            player2_wins & valid_hands, 
            cp.where(pid==1, 1, -1), 
            payoffs
        )

        return payoffs

    #
    # Define shape
    #
    shape = [game.num_players] + [1326] * game.num_players

    #
    # Apply vectorized function
    #
    # Note - np.fromfunction applies the function to the indices of the result array.
    #
    #        So, payoffs[i, j, k, l] = get_hand_payoff(i, j, k, l)
    #
    #        where payoffs.shape = shape (as defined above)
    #
    payoffs = cp.fromfunction(lambda pid, hand1, hand2: get_hand_payoff(cp.array(pid, dtype=cp.int8), 
                                                                        cp.array(hand1, dtype=cp.int8),
                                                                        cp.array(hand2, dtype=cp.int8)), 
                                                                        shape, dtype=cp.float64)
    
    return payoffs





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
    cfvn = None

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
        
        # DEBUG
        #self.check_matrix(self.player_ranges)
        
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
    # --> Debug functions
    #
    # Verify that the player's values have zero entries
    # corresponding to invalid hands.
    #
    def check_matrix(self, matrix):
        assert matrix.shape[1:] == (52, 52) # Only check hand matrices
        try:
            is_error = False
            for i in range(matrix.shape[0]):
                assert(np.all(matrix[i][np.tril_indices(52)] == 0.))
                #assert not np.all(matrix[i] == 0.), "Input matrix has all zero values"
                card_idxs = [card.to_int() for card in self.game.public_cards]
                for cid in card_idxs:
                    if np.any(matrix[i, cid, :]) or np.any(matrix[i, :, cid]):
                        is_error = True
                        break
            if is_error:
                raise ValueError("Input matrix has non-zero values for invalid hands")
        except:
            import traceback
            print(traceback.format_exc())
            assert(False)
    #
    # Activate the entire game tree
    # (ABC method)
    #
    @classmethod
    def activate_full_tree(self) -> None:
        pass
    #
    # --> Class member setters
    #
    # Set the game tree's counterfactual value network
    #
    @classmethod
    def set_cfvn(cls, cfvn : CounterfactualValueNetwork) -> None:
        cls.cfvn = cfvn
    #
    # Get the game tree's counterfactual value network
    #
    @classmethod
    def get_cfvn(cls) -> CounterfactualValueNetwork:
        return cls.cfvn
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
    # and return a list of queries made to the cfvn in the node's subtree
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
        # Deck of trey cards
        #
        deck = sorted(init_standard_deck(), key=lambda x: x.to_int()) 
        trey_deck = [card.to_treys() for card in deck]
        community_cards = [card.to_treys() for card in game.public_cards]
        community_idxs = [card.to_int() for card in game.public_cards]

        #
        # Get a vector of hand ranks
        #
        """
        idx_to_2d = np.triu_indices(52, k=1)
        evaluator = treys.Evaluator()
        
        # Modified eval_hand function
        def eval_hand(idx):
            idx = np.atleast_1d(idx).astype(int)  # Ensure idx is a numpy array
            
            # Convert flat idx to 2D indices
            card1 = idx_to_2d[0][idx]
            card2 = idx_to_2d[1][idx]
            
            # Vectorized filtering of invalid hands
            mask = np.isin(card1, community_idxs) | np.isin(card2, community_idxs)
            
            # Compute hand rankings (assign np.inf to invalid hands)
            hand_ranks = np.full(idx.shape, np.inf)  # Initialize all as np.inf
            valid_idx = ~mask  # Boolean mask for valid hands
            
            if np.any(valid_idx):  # Only evaluate valid hands
                hand_ranks[valid_idx] = [
                    evaluator.evaluate([trey_deck[c1], trey_deck[c2]], community_cards)
                    for c1, c2 in zip(card1[valid_idx], card2[valid_idx])
                ]
            
            return hand_ranks
        

        # Vectorized evaluation
        eval_func = np.vectorize(eval_hand, otypes=[float])  # Vectorize the function
        hand_evals = eval_func(np.arange(1326))  # Apply over all indices

        import ipdb; ipdb.set_trace()
        """

        evaluator = treys.Evaluator()
        hand_evals = np.ones((52, 52)) * np.inf
        for x in range(52):
            if x in community_idxs:
                continue
            for y in range(x+1, 52):
                if y in community_idxs:
                    continue
                hand_evals[x, y] = evaluator.evaluate(community_cards, [trey_deck[x], trey_deck[y]])

        # Vectorized version of get_hand_payoff
        def get_hand_payoff(pid, card1, card2, card3, card4, evals):
            """
            Compute the payoff of a poker hand matchup in a vectorized manner.
            """
            # Ensure values are integers
            card1, card2, card3, card4 = map(np.asarray, (card1, card2, card3, card4))

            # Condition 1: Invalid hands (wrong ordering)
            invalid_hands = (card1 >= card2) | (card3 >= card4)

            # Condition 2: Duplicate or community card in hand
            unique_cards = (card1 != card3) & (card1 != card4) & (card2 != card3) & (card2 != card4)
            no_community_cards = (~np.isin(card1, community_idxs)) & (~np.isin(card2, community_idxs)) \
                                & (~np.isin(card3, community_idxs)) & (~np.isin(card4, community_idxs))
            valid_hands = unique_cards & no_community_cards & ~invalid_hands

            # Compare hand strengths using precomputed `hand_evals`
            # NOTE - not using equality here, in the case of draws we want the payoff to remain zero.
            player1_wins = evals[card1, card2] < evals[card3, card4]
            player2_wins = evals[card1, card2] > evals[card3, card4]

            # Initialize payoff vector
            payoffs = np.zeros_like(card1, dtype=np.float64)  # Default all payoffs to 0

            # Assign the winning hands 1s and losing hands -1s
            payoffs = np.where(
                player1_wins & valid_hands, 
                np.where(pid==0, 1, -1),
                payoffs
            )

            payoffs = np.where(
                player2_wins & valid_hands, 
                np.where(pid==1, 1, -1), 
                payoffs
            )

            return payoffs

        # Define shape
        shape = [game.num_players] + [52, 52] * game.num_players

        # Apply vectorized function
        payoffs = np.fromfunction(lambda pid, c1, c2, c3, c4: get_hand_payoff(pid.astype(int), 
                                                                              c1.astype(int), 
                                                                              c2.astype(int), 
                                                                              c3.astype(int), 
                                                                              c4.astype(int),
                                                                              hand_evals), 
                                                                              shape, dtype=np.float64)

        # Save the payoffs matrix
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
        # DEBUG
        #self.check_matrix(self.player_ranges)
        """
        print()
        history = self.game.trajectory[8:]
        pstr = '--> Value update on '
        for aid in history[:-1]:
            pstr += str(Action(aid)) + ' -> '
        pstr += str(Action(history[-1]))
        pstr += ' Terminal Node'
        print(pstr)
        print([str(card) for card in self.game.public_cards], f' Pot: {self.game.dealer.pot}')
        print(f'Player {pid} {tuple(str(card) for card in self.game.players[pid].hand)}')
        
        card1, card2 = self.game.players[pid].hand[0].to_int(), self.game.players[pid].hand[1].to_int()
        print(f'Player {pid} reach prob. = {self.player_ranges[pid, card1, card2]}')
        """
        #
        # Compute the expected value matrix for player pid
        #
        # Note: payoffs from the cache have to be scaled by the pot size
        #
        if self.num_active > 1:
            #
            # Look up the payoffs matrix in the cache, if this is a river showdown
            #
            assert TerminalNode.is_in_cache(self.game), "Cant update values if the node's payoffs havent been computed"
            key = TerminalNode.five_cards_to_str(self.game.public_cards)
            payoffs = TerminalNode.payoffs_cache[key]
            #
            # self.player_ranges[opp_pid] -> (52, 52), prob of the opponent holding cards (k, k)
            #
            # self.player_ranges[opp_pid][np.newaxis, np.newaxis, :, :] -> (1, 1, 52, 52), adds two ficticous dimensions
            # 
            # payoffs[pid] -> (52, 52, 52, 52), payoff for pid holding cards (i, j) and opponent holding (k, l)
            #
            # payoffs[pid] * self.player_ranges[opp_pid][np.newaxis, np.newaxis, :, :] = payoffs[pid][i, j, k, l] * self.player_ranges[opp_pid][k, l] -> (52, 52, 52, 52)
            #
            # (payoffs[pid] * self.player_ranges[opp_pid][np.newaxis, np.newaxis, :, :]).sum(2, 3) = sum_{k, l} payoffs[pid][i, j, k, l] * self.player_ranges[opp_pid][k, l] -> (52, 52)
            #     = self.values[pid]
            #
            # self.values[pid][i, j] = sum_{k,l} payoffs[pid][i, j, k, l] * self.player_ranges[k, l]
            #                        = sum_{all opponent hands} [payoff of pid holding hand (i, j) vs opp_pid holding hand (k, l)] * prob opp_pid is holding hand (k, l) 
            #
            self.values[0] = 0.5 * self.game.dealer.pot * (payoffs[0] * self.player_ranges[1][np.newaxis, np.newaxis, :, :]).sum(axis=(2, 3))
            self.values[1] = 0.5 * self.game.dealer.pot * (payoffs[1] * self.player_ranges[0][:, :, np.newaxis, np.newaxis]).sum(axis=(0, 1))
        else:
            self.values[0] = np.triu(np.ones((52, 52)) * self.payoffs[0] * np.sum(self.player_ranges[1]), k=1)
            self.values[1] = np.triu(np.ones((52, 52)) * self.payoffs[1] * np.sum(self.player_ranges[0]), k=1)
        """
        # DEBUG
        if self.num_active > 1:
            print('validating...')
            game = copy.deepcopy(self.game)
            # DEBUG - Hard compute the values
            deck = sorted(init_standard_deck(), key=lambda x: x.to_int())
            public_cards = set([card.to_int() for card in self.game.public_cards])
            debug_values = np.zeros((2, 52, 52))
            for i in range(52):
                for j in range(i+1, 52):
                    for k in range(52):
                        for l in range(k+1, 52):
                            card_set = set([i, j, k, l])
                            if len(card_set) != 4 or not card_set.isdisjoint(public_cards):
                                assert(payoffs[0, i, j, k, l] == 0 and payoffs[1, i, j, k, l] == 0)
                                continue
                            game.players[0].hand = [deck[i], deck[j]]
                            game.players[1].hand = [deck[k], deck[l]]
                            payoff = game.get_payoffs()
                            assert(payoff[0] == 0.5 * self.game.dealer.pot * payoffs[0, i, j, k, l])
                            assert(payoff[1] == 0.5 * self.game.dealer.pot * payoffs[1, i, j, k, l])
                            debug_values[0, i, j] += payoff[0] * self.player_ranges[1, k, l]
                            debug_values[1, k, l] += payoff[1] * self.player_ranges[0, i, j]
            try:
                self.check_matrix(debug_values)
                assert(np.allclose(debug_values, self.values))
            except Exception as e:
                import ipdb; ipdb.post_mortem()
                assert(False)
            self.values = debug_values
            print('done')
        """
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
    # Debug function
    # 
    # Terminal nodes are active leaf nodes by default,
    # so nothing to do here.
    #
    def activate_full_tree(self):
        pass

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
        self.accumulate_strategy = np.random.rand(len(self.actions), 52, 52) # sum of all strategies
        self.n_strategy_updates = 0 # number of strategy updates

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
        # Used to construct PUCT strategy during tree growth
        #
        # Counts child node visits during growth
        #
        # self.visits : action -> visit count
        #
        self.visits = dict.fromkeys(self.actions, 0)
    
    #
    # Return the average strategy across strategy updates
    #
    def cummulative_strategy(self):
        if self.n_strategy_updates == 0:
            raise ValueError('Computing cummulative strategy before any strategy updates have been performed.')
        cum_strategy = self.accumulate_strategy / self.n_strategy_updates
        # Renormalize
        denom = cum_strategy.sum(axis=0, keepdims=True)
        denom[denom == 0] = 1
        return cum_strategy / denom

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
        """
        print()
        history = self.game.trajectory[8:]
        if history == []:
            print(f'--> Strategy update on root')
        else:
            pstr = '--> Strategy update on '
            for aid in history[:-1]:
                pstr += str(Action(aid)) + ' -> '
            pstr += str(Action(history[-1]))
            pstr += ' Decision Node'
            print(pstr)
        """
        card1, card2 = self.game.players[pid].hand[0].to_int(), self.game.players[pid].hand[1].to_int()
        """
        print([str(card) for card in self.game.public_cards], f' Pot: {self.game.dealer.pot}')
        print(f'Player {pid} {tuple(str(card) for card in self.game.players[pid].hand)}')
        """
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
            self.regrets[action_idx] = np.maximum(self.regrets[action_idx] + child.values[pid] - self.values[pid], 0)
        #
        # Update the acting player's strategy according to the new regrets
        #
        # Using the regret matching formula
        #
        # Note: if all regrets are non-positive, then default to a uniform policy.
        #
        regret_sum = np.sum(self.regrets, axis=0, keepdims=True) # sum along actions axis, (1, 52, 52) array
        denom = np.where(regret_sum == 0, 1, regret_sum) # remove zeros from denom to avoid dividing by zero
        num_actions = len(self.actions)
        self.strategy = np.where(regret_sum == 0, 1/num_actions, self.regrets/denom)
        #
        # Invalid hands are assigned a uniform policy above.
        # Mask them out.
        #
        tril_idxs = np.tril_indices(52)
        self.strategy[:, tril_idxs[0], tril_idxs[1]] = 0.
        public_card_idxs = [card.to_int() for card in self.game.public_cards]
        for cid in public_card_idxs:
            self.strategy[:, cid, :] = 0.
            self.strategy[:, :, cid] = 0.
        #
        # Update cummulative strategy parameters
        #
        self.n_strategy_updates += 1
        self.accumulate_strategy += self.strategy

        """
        print(f'Value = {self.values[pid, card1, card2]}')
        print(f'Child values = {str({self.actions[i]: self.values[pid, card1, card2] for i in range(len(self.actions))})}')
        print(f'Regrets = {str({self.actions[i]: self.regrets[i, card1, card2] for i in range(len(self.actions))})}')
        print(f'Strategy = {str({self.actions[i]: self.strategy[i, card1, card2] for i in range(len(self.actions))})}')
        #if history == []:
        #    import ipdb; ipdb.set_trace()
        """
        
    #
    # Estimate this node's values using the cfvn
    #
    def update_values_w_cfvn(self):
        #
        # The cfvn should only be used on inactive nodes
        #
        assert not self.is_active, "CFVN should not be used on active nodes"
        #
        # Convert the node's information into an input vector for the cfvn
        #
        input = DecisionNode.get_cfvn().to_vect(self.game, self.player_ranges, False)
        #
        # Query the network
        #
        all_actions = self.game.round.get_all_actions()
        valid_action_idxs = [all_actions.index(action) for action in self.actions]
        self.strategy, self.values = DecisionNode.get_cfvn().query(input, valid_action_idxs)
        
        pid = self.game.game_pointer
        """
        opp_pid = (pid + 1) % 2
        print()
        history = self.game.trajectory[8:]
        pstr = '--> CFVN update on '
        for aid in history[:-1]:
            pstr += str(Action(aid)) + ' -> '
        pstr += str(Action(history[-1]))
        pstr += ' Decision Node'
        print(pstr)
        card1, card2 = self.game.players[pid].hand[0].to_int(), self.game.players[pid].hand[1].to_int()
        card3, card4 = self.game.players[opp_pid].hand[0].to_int(), self.game.players[opp_pid].hand[1].to_int()
        print([str(card) for card in self.game.public_cards], f' Pot: {self.game.dealer.pot}')
        print(f'Player {pid} {tuple(str(card) for card in self.game.players[pid].hand)}')
        print(f'value = {self.values[pid, card1, card2]}')
        print()
        print(f'Player {opp_pid} {tuple(str(card) for card in self.game.players[opp_pid].hand)}')
        print(f'value = {self.values[opp_pid, card3, card4]}')
        import ipdb; ipdb.set_trace()
        """

    #
    # Perform a CFR value and strategy update
    #
    # Returns a list of queries made to the cfvn in the node's subtree
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
        # List to store cfvn queries made by the node's children
        #
        querries = []
        """
        print('================')
        print()
        history = self.game.trajectory[8:]
        if history == []:
            print(f'--> Root')
        else:
            pstr = '--> '
            for aid in history[:-1]:
                pstr += str(Action(aid)) + ' -> '
            pstr += str(Action(history[-1]))
            pstr += ' Decision Node'
            print(pstr)
        """
        
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
            
            #self.check_matrix(child.player_ranges) # DEBUG

            """
            opp_pid = (pid + 1) % 2
            card1, card2 = self.game.players[pid].hand[0].to_int(), self.game.players[pid].hand[1].to_int()
            card3, card4 = self.game.players[opp_pid].hand[0].to_int(), self.game.players[opp_pid].hand[1].to_int()
            print(f'Pid reach prob. = {self.strategy[action_idx, card1, card2]} * {self.player_ranges[pid, card1, card2]} = {child.player_ranges[pid, card1, card2]}')
            print(f'Opp reach prob. = {child.player_ranges[opp_pid, card3, card4]}')
            """

            #
            # Compute the value of the child node
            #
            if child.is_active:
                #
                # Recurse down the game tree to compute the child's values
                #
                querries += child.update_values()
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
                #querries.append((self.game, opponent_values, player_range, [action.value]))
                #
                # NOTE - Testing giving the network the child game directly
                #
                child_pid = child.game.game_pointer
                child_opp = (child_pid + 1) % 2
                querries.append((child.game, 
                                 child.values[child_opp],
                                 child.player_ranges[child_pid],
                                 []))
            #
            # Use the child's values to update the parent's values
            #
            # For the acting player,
            #
            # the value contribution associated with selecting this action is
            # equal to the value of the child state weighted by the acting player's
            # probability of selecting the action
            #
            """
            print()
            history = self.game.trajectory[8:]
            if history == []:
                print(f'--> Value update on root')
            else:
                pstr = '--> Value update on '
                for aid in history[:-1]:
                    pstr += str(Action(aid)) + ' -> '
                pstr += str(Action(history[-1]))
                pstr += ' Decision Node'
                print(pstr)
            card1, card2 = self.game.players[pid].hand[0].to_int(), self.game.players[pid].hand[1].to_int()
            print([str(card) for card in self.game.public_cards], f' Pot: {self.game.dealer.pot}')
            print(f'Player {pid} {tuple(str(card) for card in self.game.players[pid].hand)}')
            """
            self.values[pid] += self.strategy[action_idx] * child.values[pid]
            """
            print(f'value_[{self.actions[action_idx]}] = strat prob * child value = {self.strategy[action_idx, card1, card2]} * {child.values[pid, card1, card2]} -> {self.values[pid, card1, card2]}')
            if history == []:
                import ipdb; ipdb.set_trace()
            """
            
            #
            # For the non-acting players,
            #
            # the player's value in the parent node is simply a sum of the 
            # player's values in the child nodes.
            #
            for opp_pid in range(self.game.num_players):
                if opp_pid != pid:
                    self.values[opp_pid] += child.values[opp_pid]
        
        #print('================')
        

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
        
        #self.check_matrix(child_ranges) # DEBUG
        
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
    # Note - We use a mixed strategy here using PUCT statistics.
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
        card1, card2 = hands[pid][0], hands[pid][1]
        cfr_strat = self.strategy[:, card1, card2] # np.array (num_actions,)
        assert(np.isclose(np.sum(cfr_strat), 1)) # Check that the CFR strategy is normalized
        #
        # Compute a strategy weighted by PUCT scores
        #
        total_visits = sum(self.visits.values())
        puct_scores = np.zeros(len(self.actions)) # np.array (num_actions,)
        c_puct = 1.0 # PUCT exploration parameter
        for i, a in enumerate(self.actions):
            puct_scores[i] = self.children[a].values[pid, card1, card2]
            puct_scores[i] += c_puct * cfr_strat[i] * np.sqrt(1e-5 + total_visits) / (1 + self.visits[a])
        epsilon = 1e-5 # Add a small value to ensure we dont assign a zero probability to any action
        puct_strat = (puct_scores - np.min(puct_scores, 0) + epsilon) / np.sum(puct_scores - np.min(puct_scores, 0)  + epsilon) # Shift and normalize PUCT scores
        #
        # Mix the CFR and PUCT strategies
        #
        strat = 0.5 * (cfr_strat + puct_strat)
        #
        # Attempt to add a node in this node's subtree
        #
        action_order = np.random.choice(self.actions, size=len(self.actions), p=strat, replace=False) # Order in which to try actions
        for action in action_order:
            #
            # Check that this child is not None
            #
            assert self.children[action] is not None, "Active nodes cant have null children"
            #
            # Recurse to the child
            #
            if self.children[action].grow_tree(hands):
                return True
        return False

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
        if trajectory[:n] != self.game.trajectory:
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
                assert isinstance(node, DecisionNode), "The given trajectory must correspond to a decision node"
                return node
            #
            # Continue searching
            #
            prev_node = node
            if isinstance(node, DecisionNode):
                node = node.children[Action(trajectory[depth])]
            else:
                node = node.children[trajectory[depth]]
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
    # Debug function
    # 
    # Activate all nodes using DFS traversal
    #
    def activate_full_tree(self):
        self.activate()
        for child in self.children.values():
            child.activate_full_tree()
    #
    # Debug function
    #
    def print_tree(self):
        import anytree
        
        root = anytree.Node(name="root", value=self)
        
        q = [root]
        while q:
            parent_node = q.pop()
            if isinstance(parent_node.value, DecisionNode):
                for action in parent_node.value.actions:
                    child = parent_node.value.children[action]
                    node = anytree.Node(name=action.value, value=child, parent=parent_node)
                    if child.is_active:
                        q.append(node)
        
        
        card1 = self.game.players[0].hand[0].to_int()
        card2 = self.game.players[0].hand[1].to_int()
        card3 = self.game.players[1].hand[0].to_int()
        card4 = self.game.players[1].hand[1].to_int()
        
        print()
        print("Public Cards:", [str(card) for card in self.game.public_cards])
        print("P0 Hand =", [str(card) for card in self.game.players[0].hand])
        print("P1 Hand =", [str(card) for card in self.game.players[1].hand])
        print()

        for pre, fill, node in anytree.RenderTree(root):
            
            my_pid = node.value.game.game_pointer
            
            if node is root:
                print(f"{pre} {node.name}")
                print(f"{fill}")
                print(f"--> P{my_pid} Decision")
            else:
                parent = node.parent
                
                action = parent.value.actions.index(Action(node.name))
                
                parent_pid = parent.value.game.game_pointer
                
                if parent_pid == 0:
                    strat = parent.value.strategy[action, card1, card2]
                    regret = parent.value.regrets[action, card1, card2]
                else:
                    strat = parent.value.strategy[action, card3, card4]
                    regret = parent.value.regrets[action, card3, card4]
                
                print(f"{pre} ({Action(node.name)} : strat = {strat}, regret = {regret})")
                print(f"{fill}")
                
                if isinstance(node.value, DecisionNode):
                     print(f"{fill} --> P{my_pid} Decision")
                elif isinstance(node.value, TerminalNode):
                    print(f"{fill} --> Terminal Node")
                else:
                    raise ValueError("Unrecognized node type")
            
            p0_reach_prob = node.value.player_ranges[0, card1, card2]
            p1_reach_prob = node.value.player_ranges[1, card3, card4]

            p0_value = node.value.values[0, card1, card2]
            p1_value = node.value.values[1, card3, card4]

            print(f"{fill}")
            print(f"{fill} P0 reach prob = {p0_reach_prob}")
            print(f"{fill} P1 reach prob = {p1_reach_prob}")
            print(f"{fill}")
            print(f"{fill} P0 value = {p0_value}")
            print(f"{fill} P1 value = {p1_value}")
            print(f"{fill}") 

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
    # Returns a list of queries made to the cfvn in the node's subtree
    #
    def update_values(self) -> list[tuple[np.ndarray]]: # NOTE - THIS NEEDS TO BE UPDATED
                                                        #         QUERY FORMAT HAS CHANGED
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
            # Convert the nodes information into an input vector for the cfvn
            #
            input = DecisionNode.get_cfvn().to_vect(self.game, self.player_ranges, True)
            #
            # Query the network
            #
            self.values, self.strategy = DecisionNode.get_cfvn().query(input)
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
        # List to store querries to the cfvn made in the subtree
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
    # Return True if successful, False otherwise
    #
    # NOTE - I think its debatable whether we want to allow growth past chance nodes.
    #        We would be better off exploring decision nodes in the same stage as our
    #        decision point.
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
        sampled_idxs = set() # Remember the outcomes we sample
        used_cards = {card for hand in hands for card in hand}
        while len(sampled_idxs) < len(self.outcomes):
            idx = np.random.choice(len(self.outcomes))
            # Skip outcomes that have been tried before
            if idx in sampled_idxs:
                continue
            sampled_idxs.add(idx)
            # Skip outcomes that contain invalid cards
            if any(card.to_int() in used_cards for card in self.outcomes[idx]):
                continue
            # Attempt to add a node to the outcome's subtree, return True if successful
            if self.children[idx].grow_tree(hands):
                return True
        #
        # Subtree is full
        #
        # Note: This should never be the case in practice since the chance node
        #       has a high branching factor.
        #
        return False
    
    #
    # Debug function
    # 
    # Activate all nodes using DFS traversal
    #
    def activate_full_tree(self):
        self.activate()
        for child in self.children.values():
            child.activate_full_tree()
