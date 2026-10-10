from __future__ import annotations  # Enables forward references

# External imports
import copy
import gc
import math
import numpy as np
import random
import sys
import time
from typing import TYPE_CHECKING

# Internal imports
from rlcard.agents.gt_cfr_agent.nodes import CFRTree
from rlcard.agents.gt_cfr_agent.utils import uniform_range, random_range, get_card_coors, get_1d_coor
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from rlcard.games.nolimitholdem.round import Action

# Avoid a circular import
if TYPE_CHECKING:
    from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork # Only imports for type hints

#
# This class immplements the GT-CFR algorithm
#
# Given - A game state object
#
# Return - The acting player's strategy at the game state
#          and all the player's expected values at the game state
#
class GTCFRSolver():
    #
    # Initialize the GT-CFR Solver parameters and
    # the counterfactual value network.
    #
    def __init__(self, input_cfvn: CounterfactualValueNetwork =None, 
                       prob_query_solve: float =0.9,
                       n_expansions: int =10,
                       n_expansions_per_regret_update: float =0.01,
                       full_solve: bool =False,
                       n_trainers: int =1,
                       n_query_solvers: int =1,
                       total_actions: int =3,
        ):
        # Import at runtime
        from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
        #
        # Wheter to fully solve the game tree or not
        #
        self.full_solve = full_solve
        #
        # Initialize the counterfactual value model, if one is not given.
        #
        # NOTE - right now the cfvn is automatically initialized with default params
        #
        if self.full_solve:
            self.cfvn = None # We don't need to estimate game states if the tree is fully expanded
        else:
            self.cfvn = CounterfactualValueNetwork(n_trainers=n_trainers, n_query_solvers=n_query_solvers) if input_cfvn is None else input_cfvn
        #
        # Probability of fully solving a given cfvn query
        # Expected number of CFVN queries logged per search for the query solvers
        # (paper's q_search for self-play searches, q_recursive for query solves).
        # Not a per-query probability: a search makes hundreds of queries.
        assert prob_query_solve >= 0, "Invalid queries-per-search rate"
        self.prob_query_solve = prob_query_solve
        #
        # Number of expansion simulations per gt-cfr run
        #
        # Note 1: Sometimes refered to as the 's' parameter in the literature
        #
        if self.full_solve:
            self.n_expansions = 0 # Game tree will already be fully expanded
        else:
            self.n_expansions = n_expansions
        #
        # Ratio of tree expansion per cfr regret updates
        # Fractional - 0.01 = 100 regret updates per tree expansion
        #
        # Note 1: This notation is historical (AlphaGo)
        #
        # Note 2: Sometimes refered to as the 'c' paramter in the literature
        #
        self.n_expansions_per_regret_updates = n_expansions_per_regret_update
        #
        # Root of the game tree
        #
        # Set to None upon class initialization.
        #
        self.tree = None
        #
        # Boolean whether to fully expand the tree or not.
        #
        self.full_tree = self.full_solve
        #
        # Decision point
        #
        # The decision point is the decision node in the game tree from which 
        # we sample an action after solving. This is always a descendant of the 
        # root node, and the end point of the trajectory seed.
        #
        # The trajectory seed directs the game tree growth toward the decision node 
        #
        self.trajectory_seed = None
        self.decision_point = None
        #
        # True while solving a logged query: both root ranges are given and fixed (no gadget)
        # and the acting player's hand is sampled from the range during growth.
        #
        self.fixed_beliefs = False
    
    #
    # Find the node of the current tree that matches the given game state, following
    # the actions played since the tree's root.
    #
    # Returns (node_id, exact). exact=False means the walk stopped at the deepest
    # ancestor present in the tree (a frontier node with no children).
    #
    def node_search(self, target_game: NolimitholdemGame) -> tuple[int, bool]:
        assert self.tree is not None, "No tree to search"
        root_traj = self.tree.game_states[0].trajectory
        assert target_game.trajectory[:len(root_traj)] == root_traj, "Game state is not below the tree's root"
        node = 0
        adjacency = self.tree.tree.numpy()
        node_types = self.tree.node_types.numpy()
        for action_id in target_game.trajectory[len(root_traj):]:
            if node_types[node] != 0:
                raise ValueError("Search walked through a terminal node")
            if action_id not in self.tree.all_actions:
                raise NotImplementedError("Chance outcomes in the trajectory are not supported yet (D11)")
            action_idx = self.tree.all_actions.index(action_id)
            children = np.where(adjacency[node] == action_idx)[0]
            if len(children) == 0:          # frontier node: no children in the tree
                return node, False
            node = int(children[0])
        return node, True

    #
    # Given the acting player's range, assign randomized ranges
    # to the opponent players.
    #
    def compute_initial_ranges(input_game, player_range):
        #
        # Allocate memory for the player ranges,
        # initialized to zero.
        #
        n_players = input_game.num_players
        ranges = np.zeros((n_players, 1326))
        #
        # For each player in the game...
        #
        for pid in range(n_players):
            #
            # If this player is the acting player,
            # then assign the input range to them.
            #
            if pid == input_game.game_pointer:
                ranges[pid] = player_range
            #
            # Otherwise, the player is an opponent,
            # assign them a random range.
            #
            # Note: opponent ranges at the root node
            #       are set by the gadget game after the
            #       first iteration of CFR
            #
            else:
                ranges[pid] = random_range(input_game.public_cards)
        return ranges

    #
    # Initialize the starting game tree
    #
    # input_game - game state that was input for solving,
    #              serves as the root node of the game tree
    #
    # input_player_range - probability distribution over possible
    #                      hands the player could given the input game state
    #
    # trajectory_seed - list of actions, add the game states associated with the
    #                   chain of actions to the game tree before starting
    #
    # Resolving requires two additional inputs:
    #
    #     1. The player's range
    #     
    #     2. The opponent's values
    #
    # Solving can be initialized with three different sets of information:
    #
    #   Case 1 - No information
    #      (self.tree=None, opponent_values=None, player_range=None, trajectory_seed=None)
    #
    #        - Every player's range is uniform over the hands compatible with the board
    #
    #        - No gadget game: there is no earlier solve to take opponent values from
    #
    #        - self.tree is rooted at the given game state
    #
    #        * This is used by gt_cfr_agent at the start of an episode
    #
    #   Case 2 - Evaluated game tree from a previous solve call
    #      (self.tree=CFRTree(...), opponent_values=None, player_range=None, trajectory_seed=None)
    #
    #        - Search the existing game tree for the input game state
    #
    #        - If the input game state is found, 
    #          then set it to be the new root node of the game tree.
    #
    #        - If the input game state is not found,
    #          then set the closest leaf node of the game tree to be the new root node.
    #          Add the input game state to the initial game tree.
    #
    #        * This is used by successive gt_cfr_agent calls during an episode
    #
    #   Case 3 - Empty game tree. Both players' ranges given (solving a logged query).
    #      (self.tree=None, input_player_ranges=np.array((2, 1326)))
    #
    #        - Both root ranges are set from the input and stay fixed: no gadget game
    #          (paper: training solves a subgame with fixed beliefs for both players).
    #
    #        * This is used by the query solvers during training
    #
    def init_game_tree(self, input_game: NolimitholdemGame, input_player_ranges: np.ndarray = None) -> None:
        #
        # Case 1 - No starting information
        #
        if self.tree is None and input_player_ranges is None:
            #
            # The first decision of a hand has no strategic history: every player's
            # range is uniform over the hands compatible with the board, and there is
            # no previous solve to constrain the opponent with. So this is a plain
            # fixed-belief solve, no gadget game, no heuristic opponent values.
            # Later decisions in the hand re-solve from this tree (case 2).
            #
            player_ranges = np.tile(uniform_range(input_game.public_cards), (input_game.num_players, 1))
            #
            # Initialize the root node of the public game tree
            #
            # NOTE - Do we need this to be a deepcopy?
            #        Nothing should be editing the game state while we solve.
            #
            self.tree = CFRTree(self.cfvn, use_gadget=False)
            self.tree.add_root_node(copy.deepcopy(input_game), player_ranges)
            self.full_tree = False
            #
            # The root node is the decision node we are solving. Beliefs are not
            # 'fixed' in the query-solve sense: in self-play the acting player's true
            # hand still steers tree growth (see actor_hand).
            #
            self.fixed_beliefs = False
            self.trajectory_seed = []
        #
        # Case 2 - Existing tree from the previous solve of this hand: re-solve from it.
        #
        #   Safe re-solving needs two pieces of summary information from the previous
        #   search at the node matching the current state: the acting player's own range
        #   there, and the opponent's counterfactual values there (the gadget's opt-out
        #   values). Both are taken verbatim from the old tree so that the opt-out values
        #   and the new root values share one scale (see the convention note in
        #   CFRTree.update_gadget_regrets).
        #
        elif self.tree is not None and input_player_ranges is None:
            node, exact = self.node_search(input_game)
            if not exact:
                raise NotImplementedError("Current state is not in the previous tree; "
                                          "rooting at the nearest ancestor is implemented with chance nodes (D11)")
            actor = input_game.game_pointer
            opponent = (actor + 1) % 2
            old = self.tree
            player_range = old.ranges[old.range_map[node, actor]].numpy().copy()
            opponent_values = old.values[node, opponent].numpy().copy()
            player_ranges = GTCFRSolver.compute_initial_ranges(input_game, player_range)
            self.tree = CFRTree(self.cfvn)
            self.tree.add_root_node(copy.deepcopy(input_game), player_ranges, opponent_values=opponent_values)
            self.full_tree = False
            self.fixed_beliefs = False
            self.trajectory_seed = []
        #
        # Case 3 - No tree. Both players' ranges given: a fixed-belief solve (logged query).
        #
        elif self.tree is None and input_player_ranges is not None:
            player_ranges = np.asarray(input_player_ranges, dtype=np.float32)
            assert player_ranges.shape == (input_game.num_players, 1326), "Expected both players' ranges, shape (num_players, 1326)"
            self.tree = CFRTree(self.cfvn, use_gadget=False)
            self.tree.add_root_node(copy.deepcopy(input_game), player_ranges.copy())
            self.full_tree = False
            self.fixed_beliefs = True
            self.trajectory_seed = []
        #
        #
        # Error case - Unrecognized initialization input configuration
        #
        else:
            raise ValueError("A previous tree exists; call reset() before a fixed-belief solve")
        #
        # If we are solving the full game tree, then activate all nodes in the tree.
        #
        if self.full_solve:
            self.tree.activate_full_tree()
            self.full_tree = True # The tree is now full
            self.decision_point = 0
        else:
            #
            # Activate the root node
            #
            # NOTE - root node id is always zero
            #
            if not self.tree.active_nodes[0]:
                self.tree.activate(0)
            #
            # The decision point is the root. Seeded trees (decision point below the
            # root, paper's modified re-solving) arrive with chance nodes (D11).
            #
            if self.trajectory_seed:
                raise NotImplementedError("Trajectory seeds are not supported yet (D11)")
            self.decision_point = 0 # root node id
            #
            # Activate the decision point's children
            #
            self.tree.activate_children(self.decision_point)

    #
    # Public tree counterfactual regret minimization.
    #
    # CFR runs over every node of self.tree (see CFRTree.cfr_update), from
    # the game tree nodes.
    #
    def cfr(self, train=False) -> None:
        strt = time.time()
        self.tree.cfr_update(math.ceil(1/self.n_expansions_per_regret_updates))
        print(f'CFR: {time.time() - strt}')
        if train:
            self.log_queries()

    #
    # Log CFVN queries for the query solvers.
    #
    # The paper logs on average q_search queries per search (q_recursive per query solve),
    # not a fraction of all queries. Called after each CFR batch: draws Poisson(q / n_batches)
    # frontier nodes uniformly and records (game, ranges) for each, so a search contributes
    # q queries in expectation.
    #
    def log_queries(self) -> None:
        if self.cfvn is None or self.prob_query_solve <= 0:
            return
        frontier = self.tree.frontier_nodes()
        if len(frontier) == 0:
            return
        n_batches = self.n_expansions + 1
        for _ in range(np.random.poisson(self.prob_query_solve / n_batches)):
            node = int(np.random.choice(frontier))
            self.cfvn.add_to_query_queue(self.tree.frontier_query(node))

    #
    # The acting player's hand for tree growth: the real hand in self-play (see decision A12),
    # None in a fixed-belief solve, where grow samples it from the range like any other hand.
    #
    def actor_hand(self):
        if self.fixed_beliefs:
            return None
        game = self.tree.game_states[0]
        return get_1d_coor(*sorted(card.to_int() for card in game.players[game.game_pointer].hand))
    
    #
    # Growing Tree Counterfacutal Regret
    #
    def gt_cfr(self, train=False) -> None:
        #
        # Each iteration computes the values of each node in the public state tree,
        # then adds a new leaf node to the tree.
        #
        # Initial CFR solve
        #
        self.cfr(train=train)
        #
        # Expand iterations
        #
        for i in range(self.n_expansions):
            #
            # Add a new node to the game tree
            #
            strt = time.time()
            self.tree.grow(actor_hand=self.actor_hand())
            print(f'GROW: {time.time() - strt}')
            #
            # Run cfr to update the policy and regret estimates 
            # for each state in the tree
            #
            self.cfr(train=train)
    
    #
    # Wrapper function around gt-cfr to handle fully solving a 
    # subset of cfvn queries made during gt-cfr
    #
    def training_gt_cfr(self):
        #
        # Check starting game tree
        # We start with a root node and its children
        #
        assert self.tree, "The game tree needs to be initialized before GT-CFR is ran"
        #
        # Run gt-cfr
        #
        self.gt_cfr(train=True)

    #
    # Return a policy and value estimate for the given game state using gt-cfr
    #
    def solve(self, game: NolimitholdemGame, input_player_ranges: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
        start = time.time()
        #
        # Check if the weights need to be updated from shared memory
        #
        # Note: if CFVN is not using shared weights, this will do nothing
        #
        if not self.full_solve:
            self.cfvn.update_weights()
        #
        # Initialize the game tree for cfr
        #
        self.init_game_tree(game, input_player_ranges)
        #
        # GT-CFR training run 
        #
        self.training_gt_cfr()
        #
        # Return the computed strategies and values for the root node
        #
        print(f'Solve time: {time.time() - start}')
        return self.tree.get_cum_strategy(self.decision_point), self.tree.get_values(self.decision_point) # the average strategy, as in the paper
    
    #
    # Reset
    #
    def reset(self) -> None:
        # Remove the game tree
        self.tree, self.decision_point = None, None
        # Trigger Python garbage collection
        gc.collect() 

#
# This function handles the self-play loop that uses the GT-CFR solver 
# to solve game states encountered during self-play episodes.
#
class GTCFRAgent():
    #
    # Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    #
    def __init__(self, env: NolimitholdemEnv, 
                       full_solve: bool =False, 
                       prob_query_solve: int =0.9, 
                       n_expansions: int =10, 
                       n_expansions_per_regret_update: float =0.01, 
                       n_trainers: int =1, 
                       n_query_solvers: int=1,
                       input_cfvn: CounterfactualValueNetwork =None # share an existing network (and its queues/workers)
    ):
        #
        # Poker environment
        #
        self.env = env
        #
        # GT-CFR Solver
        #
        self.solver = GTCFRSolver(
            prob_query_solve=prob_query_solve,
            n_expansions=n_expansions,
            n_expansions_per_regret_update=n_expansions_per_regret_update,
            full_solve=full_solve, 
            n_trainers=n_trainers, 
            n_query_solvers=n_query_solvers,
            total_actions = env.game.get_num_actions(),
            input_cfvn=input_cfvn
        )
        #
        # Cap the number of moves that can be made in a self play episode
        #
        self.max_moves = np.inf # disabled by default
        #
        # Exploration parameter
        #
        self.epsilon = 0.1 # self-play uniform policy mix
        #
        # Select the greedy action during self play after n moves
        #
        self.greedy_after_n_moves = np.inf # disabled by default
        #
        # Probability of adding a CFR solver output in the
        # self-play trajectory to the cfvn's replay buffer
        #
        self.prob_add_to_buffer = 0 # called 'p_td1' in the literature
                                    # disabled by default

    #
    # Reset the solver's state
    #
    def reset(self) -> None:
        self.solver.reset()
    
    #
    # Play through one hand of poker using gt-cfr to estimate
    # policies and values.
    #
    def self_play(self) -> None:
        #
        # Start with a fresh solver game tree
        #
        self.solver.reset()

        #
        # Start from the beggining of the game
        #
        self.env.reset()
        
        #
        # Count the number of moves that have been made in the game
        #
        num_moves = 0

        #
        # Store the game trajectory as (belief state, cfr_values, cfr_policy)
        #
        trajectory = []

        #
        # Game loop
        #
        while not self.env.is_over() and num_moves < self.max_moves:
            #
            # Get the active player and legal actions for the current game state
            #
            pid = self.env.get_player_id()
            state = self.env.get_state(pid)
            legal_actions = self.env.game.get_legal_actions()
            player_hand = self.env.game.players[pid].hand
            player_stack = self.env.game.players[pid].remained_chips

            #
            # Use GT-CFR to solve for the following:
            #
            #   - strategies for all possible hands the player could have
            #     in this public state
            #     = upper triangular array w/ shape (num actions, 52, 52)
            #
            #   - Expected value for all possible hands the player could have 
            #     in this public state according to the given strategies
            #     = upper triangular array w/ shape (52, 52)
            #
            cfr_policies, cfr_values = self.solver.solve(self.env.game)

            #
            # Get the EV and policy for the player's specific hand
            #
            # Sort the card idxs because the arrays are upper triangular
            #
            hand_idxs = sorted([card.to_int() for card in player_hand])
            hand = get_1d_coor(hand_idxs[0], hand_idxs[1]) # 1326-hand index, see utils
            ev = float(cfr_values[pid, hand])
            cfr_policy = cfr_policies[:, hand]
            print(f'Board {str([str(c) for c in self.env.game.public_cards])}')
            print(f'Pot {self.env.game.dealer.pot}')
            print()
            print(f'Player {pid} ({str(player_hand[0])}, {str(player_hand[1])}) {player_stack} bb')
            print()
            print(f'Expected value {ev}')
            print()
            print(f'Strategy:')
            for i, action in enumerate(legal_actions):
                print(f'    {action} =  {round(cfr_policy[i], 3)}')
            print()
            
            #
            # Mix the controller's policy with a uniform prior to encourage
            # exploration.
            #
            uniform_policy = np.ones(cfr_policy.shape) / cfr_policy.shape[0]
            mixed_policy = (1-self.epsilon) * cfr_policy.astype(np.float64) + self.epsilon * uniform_policy
            mixed_policy /= mixed_policy.sum() # float32 strategy rounding; keep the sampler's exact-sum check happy

            #
            # Select an action
            #
            if num_moves < self.greedy_after_n_moves:
                action = np.random.choice(legal_actions, p=mixed_policy)
            else:
                action = legal_actions[np.argmax(mixed_policy)]

            print(f'Action: {action}')
            print()
            
            #
            # Take action
            #
            self.env.step(action)
            num_moves += 1

            #
            # Update game trajectory
            #
            if self.prob_add_to_buffer > 0: # disabled by default
                trajectory.append((state, cfr_values, cfr_policies))

        #
        # Add to replay buffer
        #
        if self.prob_add_to_buffer > 0: # disabled by default
            for token in trajectory: 
                if random.random() < self.prob_add_to_buffer:
                    self.solver.cfvn.add_to_replay_buffer(token)

