# External imports
import copy
import math
import numpy as np
import random

# Internal imports
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.agents.gt_cfr_agent.nodes import CFRNode, DecisionNode
from rlcard.agents.gt_cfr_agent.utils import rand_value, initial_hand_values

class GTCFRAgent():
    #
    # Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    #
    def __init__(self, env: NolimitholdemEnv):
        #
        # Poker environment
        #
        self.env = env
        assert(self.env.allow_step_back)
        
        #
        # Replay buffer
        #
        self.replay_buffer = []

        #
        # Cap the number of moves that can be made in a self play cycle
        #
        self.max_moves = 10

        #
        # Minimum counterfactual value to continue solving
        #
        self.resign_threshold = 0

        #
        # Exploration parameter
        #
        self.epsilon = 0.1

        #
        # Select the greedy action during self play after n moves
        #
        self.greedy_after_n_moves = 10

        #
        # Probability of adding an entry into the replay buffer
        #
        self.prob_add_to_buffer = 0.5

        #
        # Probability of solving a given cvpn query
        #
        self.prob_query_solve = 0.5

        #
        # Buffer of cvpn querries to solve
        #
        self.queries_to_solve = []

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
    #       e.x. if they know Player 1 always chooses to play toward the paper node, then
    #            their optimal strategy at that node would be to always play Scissors.
    #
    #
    # This is modeled here by the "regret gadget" 
    # 
    # The oppponent has two (fictitious) actions:
    #
    #     - "Follow"    (F) - choose to play toward the root state
    #
    #     - "Terminate" (T) - reject the root state and select actions to play away from the root state
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
    # NOTE - implement this function for >2 player games, shouldn't be too hard
    #
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
        gadget_regrets_pos = np.maximum(self.gadget_regrets, 0)
        gadget_follow_strat = gadget_regrets_pos[0] / (gadget_regrets_pos[0] + gadget_regrets_pos[1])

        #
        # Set the opponent's range in the cfr root node to the gadget's follow strategy 
        #
        # Reasoning:
        #     Gadget follow strat = probability of choosing to play toward the cfr root state
        #     Opp. root range = prob. of the opp. reaching this state given her strategy
        #     Therefore, gadget follow strat = opponent's range at the root node. 
        #
        opp_pid = (self.env.game.game_pointer + 1) % 2
        self.root.player_ranges[opp_pid] = gadget_follow_strat

        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        new_gadget_values = gadget_follow_strat * self.gadget_values[0] + (1 - gadget_follow_strat) * self.gadget_values[1]

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
        self.gadget_regrets[0] += self.root.player_values[opp_pid] - new_gadget_values[0] # gadget value @ t
        self.gadget_regrets[1] += self.terminate_values - self.gadget_values[1] # gadget value @ t + 1

        #
        # Update the gadget values to the new values
        #
        self.gadget_values = new_gadget_values

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
        """TODO - impliment this part"""
        player_ranges = np.zeros((self.env.game.num_players, 52, 52)) # NOTE - PLACE FILLER
        """"""
        #
        # Compute the opponent's values for choosing the Terminate gadget action
        #
        # This is to account for the fact that the opponents can steer the game 
        # away from this public state in their ancestor decision nodes.
        #
        #     Case 1: This is the start of the game. We need to use a heuristic
        #     to estimate the player's values. This can be done by precomputing the
        #     winning percentage of each hand if the game checked to showdown.
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
        """TODO - impliment this part - currently a place filler"""
        self.terminate_values = np.zeros((52, 52)) # = t_values = v_2 in the literature
        ''''''
        self.gadget_regrets = np.zeros(2, 52, 52) # 2 gadget actions, (Follow, Terminate)
        self.gadget_values = np.zeros(52, 52)
        #
        # Initialize the player's CFR values to zero.
        #
        player_values = np.zeros((self.env.game.num_players, 52, 52))
        #
        # Initialize the root node of the public game tree
        #
        self.root = DecisionNode(copy.deepcopy(self.env.game), player_ranges, player_values)
        #
        # Save the player's id in the game tree
        #
        CFRNode.set_root_pid = self.root.game.game_pointer
        #
        # Initialize the root node's public state node children
        #
        for a in self.root.actions:
            self.root.add_child(a)

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

            #
            # Use GT-CFR to solve for the following:
            #
            #   - a strategy for the player in this public state
            #
            #   - the player's expected value in this public state
            #     according to the given strategy
            #
            cfr_policy, ev = self.self_play_controller()
            
            #
            # Dont waste compute on already decided games
            #
            # Note: this feature is deactivated by default
            #
            if ev < self.resign_threshold:
                return
            
            #
            # Mix the controller's policy with a uniform prior to encourage
            # exploration.
            #
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
            trajectory.append((state, ev, cfr_policy))

        # Add to replay buffer
        for token in trajectory:
            if random.random() < self.prob_add_to_buffer:
                self.replay_buffer.append(token)


            

            