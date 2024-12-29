# External imports
import copy
import math
import numpy as np
import random

# Internal imports
from rlcard.agents.gt_cfr_agent.cvfn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.nodes import CFRNode, DecisionNode
#from rlcard.agents.gt_cfr_agent.utils import rand_value, initial_hand_values
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.nolimitholdem.game import NolimitholdemGame

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
    def __init__(self, input_cfvn: CounterfactualValueNetwork =None, prob_query_solve: float =0.9):
        #
        # Initialize the counterfactual value model, if one is not given.
        #
        # NOTE - right now the cfvn is automatically initialized with default params
        #
        self.cfvn = CounterfactualValueNetwork() if input_cfvn is None else input_cfvn
        
        #
        # Probability of fully solving a given cfvn query
        # encountered during GT-CFR solving.
        #
        assert 0 < prob_query_solve <= 1, "Invalid solve query probability"
        self.prob_query_solve = prob_query_solve
        
        #
        # Number of expansion simulations per gt-cfr run
        #
        # Note 1: Sometimes refered to as the 's' parameter in the literature
        #
        self.n_expansions = 10

        #
        # Ratio of tree expansion per cfr regret updates
        # Fractional - 0.01 = 100 regret updates per tree expansion
        #
        # Note 1: This notation is historical (AlphaGo)
        #
        # Note 2: Sometimes refered to as the 'c' paramter in the literature
        #
        self.n_expansions_per_regret_updates = 0.01

        #
        # Root of the game tree
        #
        # Set to None upon class initialization.
        #
        self.root = None

    #
    # Initialize the starting game tree
    #
    # input_game - game state that was input for solving,
    #              serves as the root node of the game tree.
    #
    def init_game_tree(self, input_game: NolimitholdemGame):
        #
        # Compute the decision player's range at the root state
        #
        #     Case 1: This is the start of the game. The player's
        #     range is the probability of being dealt each hand comination.
        #
        #     Case 2: This is not the start of the game. Use the player's range from the
        #     previous CFR run.
        #
        # The opponent players' ranges can be randomized.
        #
        """TODO - impliment this part"""
        player_ranges = np.zeros((input_game.num_players, 52, 52)) # NOTE - PLACE FILLER
        """"""
        #
        # Initialize the player's CFR values to zero.
        #
        player_values = np.zeros((input_game.num_players, 52, 52))
        #
        # Initialize the root node of the public game tree
        #
        self.root = DecisionNode(True, copy.deepcopy(input_game), player_ranges, player_values)
        #
        # Save the player's id in the game tree
        #
        CFRNode.set_root_pid = input_game.game_pointer
        #
        # Store a reference to the CVFN in the game tree
        #
        self.root.set_cvfn(self.cfvn)
        #
        # Activate the root node and its children
        #
        self.root.activate()
        for child in self.root.children:
            child.activate()

    #
    # Public tree counterfactual regret minimization.
    #
    # CFR starts on the root state, self.root, and recurses down through
    # the game tree nodes.
    #
    def cfr(self) -> None:
        #
        # Run for a fixed number of value updates on the tree.
        #
        for _ in range(math.ceil(1/self.n_expansions_per_regret_updates)):
            #
            # Perform one iteration of value and strategy updates on the game tree
            #
            self.root.update_values()

    #
    # Add a node to the game tree
    #
    # Which node to add is determined by sampling a hand configuration
    # weighted by the player's ranges at the root node, then actions are 
    # sampled down the tree until an action for which the resulting node 
    # is not in tree. Then that node is added to the tree.
    #
    # In some instances, the sampled trajectory will lead
    # to a terminal node. In which case, another trajectory should be sampled.
    #
    def grow(self):
        #
        # Sample hand assignments for each player, weighted by their ranges
        # in the root node.
        #
        hands = [] # list of player's hands
        used_cards = set() # track cards that've been used by previous players
        #
        # For each player...
        #
        for pid in range(self.root.game.num_players):
            #
            # Flatten the player's range to make sampling easier
            #
            hand_probs = self.root.player_ranges[pid].flatten()
            #
            # Mask out cards that have already been taken
            #
            for card1, card2 in used_cards:
                hand_probs[52 * card1 + card2] = 0
            #
            # Normalize the hand probabilities
            #
            hand_probs /= hand_probs.sum()
            #
            # Sample a hand
            #
            idx = np.random.choice(hand_probs.size, p=hand_probs)
            #
            # Convert the 1D idx back to 2D hand, (card1, card2)
            #
            hand = divmod(idx, 52)
            #
            # Check that the hand is valid
            #
            assert hand[0] < hand[1], "Ill-formatted hand selected"
            #
            # Add the hand to the hand list
            #
            hands.append(hand)
            #
            # Update used cards set
            #
            set.add(hand)
        #
        # Try to add a node to the subtree,
        # using the given hand to sample a trajectory
        #
        max_attempts = 10
        attempts = 0
        while not self.root.grow_tree(hands) and attempts < max_attempts:
            attempts += 1
    
    #
    # Growing Tree Counterfacutal Regret
    #
    def gt_cfr(self) -> None:
        #
        # Each iteration computes the values of each node in the public state tree,
        # then adds a new leaf node to the tree.
        #
        for _ in range(self.n_expansions):
            #
            # Run cfr to update the policy and regret estimates 
            # for each state in the tree
            #
            self.cfr()
            #
            # Add a new state node to the game tree
            #
            self.grow()
    
    #
    # Wrapper function around gt-cfr to handle fully solving a 
    # subset of cfvn queries made during gt-cfr
    #
    def training_gt_cfr(self):
        #
        # Check starting game tree
        # We start with a root node and its children
        #
        assert self.root, "The game tree needs to be initialized before GT-CFR is ran"
        assert all([self.root.children[a] for a in self.root.actions]), "Ill-formatted initial tree"
        #
        # Run gt-cfr
        #
        queries = self.gt_cfr()
        #
        # Fully solve a subset of cfvn queries from this gt_cfr run
        #
        for q in queries:
            if random.random() < self.prob_query_solve:
                self.cfvn.add_to_query_queue(q)

    #
    # Return a policy and value estimate for the current game state using gt-cfr
    #
    # Three ways solving can start:
    #
    #   1. No starting information is given
    #
    #
    def solve(self, game: NolimitholdemGame, player_ranges: np.ndarray =None) -> tuple[np.ndarray, np.ndarray]:
        #
        # Initialize the game tree for cfr
        #
        self.init_game_tree(game, ranges=player_ranges)
        #
        # GT-CFR training run 
        #
        self.training_gt_cfr()
        #
        # Return the computed strategies and values for the root node
        #
        return np.copy(self.root.strategy), np.copy(self.root.values)

#
# This function handles the self-play loop that uses the GT-CFR solver 
# to solve game states encountered during self-play episodes.
#
class GTCFRAgent():
    #
    # Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    #
    def __init__(self, env: NolimitholdemEnv):
        #
        # Poker environment
        #
        self.env = env

        #
        # GT-CFR Solver
        #
        self.solver = GTCFRSolver()

        #
        # Cap the number of moves that can be made in a self play episode
        #
        self.max_moves = np.inf # disabled by default

        #
        # Minimum counterfactual value to continue solving
        #
        self.resign_threshold = 0 # disabled by default

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
        # self-play trajectory to the cvfn's replay buffer
        #
        self.prob_add_to_buffer = 0 # called 'p_td1' in the literature
                                    # disabled by default
    
    #
    # Play through one hand of poker using gt-cfr to estimate
    # policies and values. Add to reply buffer.
    #
    def self_play(self):
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
            legal_actions = state['legal_actions'].keys()
            player_hand = self.env.game.players[pid].hand

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
            ev = cfr_values[hand_idxs]
            cfr_policy = cfr_policies[:, hand_idxs]
            
            #
            # Dont waste compute on already decided games
            #
            if ev < self.resign_threshold: # disabled by default
                return
            
            #
            # Mix the controller's policy with a uniform prior to encourage
            # exploration.
            #
            uniform_policy = np.ones(cfr_policy.shape) / cfr_policy.shape[0]
            mixed_policy = (1-self.epsilon) * cfr_policy + self.epsilon * uniform_policy

            #
            # Select an action
            #
            if num_moves < self.greedy_after_n_moves:
                action = np.random.choice(legal_actions, mixed_policy)
            else:
                action = legal_actions[np.argmax(mixed_policy)]
            
            #
            # Take action
            #
            self.env.step(action)

            #
            # Update game trajectory
            #
            if self.prob_add_to_buffer > 0: # disabled by default
                trajectory.append((state, cfr_values, cfr_policies))

        #
        # Add to replay buffer - disabled by default
        #
        # At this step, we know the actual outcome of the game that
        # was played, so we can use this as a target value for the cfvn.
        #
        # NOTE - I'm not sure the code I have written here is exactly correct.
        #        I'm choosing to leave it as is because it is currently disabled.
        #        If this code were to become active, it should be checked for 
        #        correctness.
        #
        if self.prob_add_to_buffer > 0: # disabled by default
            for token in trajectory: 
                if random.random() < self.prob_add_to_buffer:
                    self.solver.cfvn.add_to_replay_buffer(token)