import math
import numpy as np
import random

from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.agents.gt_cfr_agent.nodes import CFRNode, DecisionNode
from rlcard.agents.gt_cfr_agent.utils import rand_value, initial_hand_values

class GTCFRAgent():
    #
    # Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    #
    def __init__(self, env: NolimitholdemEnv):
        # Poker environment
        self.env = env
        assert(self.env.allow_step_back)
        
        # Replay buffer
        self.replay_buffer = []

        # Cap the number of moves that can be made in a self play cycle
        self.max_moves = 10

        # Minimum counterfactual value to continue solving
        self.resign_threshold = 0

        # Exploration parameter
        self.epsilon = 0.1

        # Select the greedy action during self play after n moves
        self.greedy_after_n_moves = 10

        # Probability of adding an entry into the replay buffer
        self.prob_add_to_buffer = 0.5

        # Probability of solving a given cvpn query
        self.prob_query_solve = 0.5

        # Buffer of cvpn querries to solve
        self.queries_to_solve = []

        # Number of expansion simulations per gt-cfr run
        # Note: sometimes refered to as the 's' parameter
        self.n_expansions = 10

        # Ratio of tree expansion per cfr regret updates
        # Fractional - 0.01 = 100 regret updates per tree expansion
        # Note: this notation is historical
        # Note: sometimes refered to as the 'c' paramter
        self.n_expansions_per_regret_updates = 0.01

    def update_gadget_regrets(self):
        pass

    #
    # Public tree counterfactual regret minimization
    # cfr starts on the root state, self.root
    #
    def cfr(self):
        #
        # Run for a fixed number of value updates on the tree.
        #
        for t in range(math.ceil(1/self.n_expansions_per_regret_updates)):
            #
            # Perform one iteration of value and strategy updates on the game tree
            #
            self.root.update_values()
            #
            # Update gadget regrets
            #
            self.update_gadget_regrets()

    #
    # Add a node to the game tree
    #
    def grow(self):
        pass

    #
    # Growing Tree Counterfacutal Regret
    #
    def gt_cfr(self):
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
        assert(self.root)
        assert(all([self.root.children[a] for a in self.root.actions]))
        
        #
        # Run gt-cfr
        #
        values, policy, queries = self.gt_cfr()
        
        #
        # Fully solve a subset of cvpn queries from this gt_cfr run
        #
        for q in queries:
            if random.random() < self.prob_query_solve:
                self.queries_to_solve.append(q)
        #
        # Return gt-cfr results
        #
        return values, policy
    
    #
    # Initialize the starting game tree
    #
    def init_game_tree(self):
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
        """
        TODO - impliment this part
        """
        player_ranges = np.zeros((self.env.game.num_players, 52, 52)) # NOTE - PLACE FILLER
        #
        # Compute the intial gadget values for the opposing players
        #
        # This is to account for the fact that the opponents can steer the game 
        # away from this public state if they choose to.
        #
        #     Case 1: This is the start of the game. We need to use a heuristic
        #     to estimate the player's values. This can be done by precomputing the
        #     winning percentage of each hand if the game checked to showdown.
        #
        #     Case 2: This is not the start of the game. Use the opponent players' values
        #     from the previous CFR run.
        #
        """
        TODO - impliment this part
        """
        gadget_regrets = np.zeros(len(self.env.game.get_legal_actions()))
        gadget_values = np.zeros((self.env.game.num_players, 52, 52)) # NOTE - PLACE FILLER
        #
        # Initialize the player's CFR values to zero.
        #
        player_values = np.zeros((self.env.game.num_players, 52, 52))
        #
        # Initialize the root node of the public game tree
        #
        self.root = DecisionNode(self.env.game, player_ranges, player_values)
        #
        # Save the player's id in the game tree
        #
        CFRNode.set_root_pid = self.root.game.game_pointer
        #
        # Initialize the root node's public state node children
        #
        for a in self.root.actions:
            self.env.step(a)
            child_pid = self.env.get_player_id()
            child_state = self.env.get_state(child_pid)
            child_actions = child_state['legal_actions'].keys()
            self.root.children[a] = StateNode(child_pid, child_state, child_actions)
            self.env.step_back()

    # Return a policy and value estimate for the current game state using gt-cfr
    def self_play_controller(self):
        self.init_game_tree()


    # Play through one hand of poker using gt-cfr to estimate
    # policies and values. Add to reply buffer.
    def self_play(self):
        # Start from the beggining of the game
        self.env.reset()
        
        # Count the number of moves that have been made in the game
        num_moves = 0

        # Store the game trajectory as (belief state, cfr_values, cfr_policy)
        trajectory = []

        # Game loop
        while not self.env.is_over() and num_moves < self.max_moves:
            
            # Get the active player and legal actions for the current game state
            pid = self.env.get_player_id()
            state = self.env.get_state(pid)
            legal_actions = state['legal_actions'].keys()

            # Compute counterfactual values and the average policy for the
            # player at the current game state.
            cfr_values, cfr_policy = self.self_play_controller()
            
            # Dont waste compute on already decided games
            if cfr_values < self.resign_threshold:
                return
            
            # Mix the controller's policy with a uniform prior to encourage
            # exploration.
            uniform_policy = np.ones(cfr_policy.shape) / cfr_policy.shape[0]
            mixed_policy = (1-self.epsilon) * cfr_policy + self.epsilon * uniform_policy

            # Select an action
            if num_moves < self.greedy_after_n_moves:
                action = np.choice(legal_actions, mixed_policy)
            else:
                action = legal_actions[np.argmax(mixed_policy)]
            
            # Take action
            self.env.step(action)

            # Update game trajectory
            trajectory.append((state, cfr_values, cfr_policy))

        # Add to replay buffer
        for token in trajectory:
            if random.random() < self.prob_add_to_buffer:
                self.replay_buffer.append(token)


            

            