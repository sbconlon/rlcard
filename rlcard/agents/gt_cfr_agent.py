import math
import numpy as np
from ordered_set import OrderedSet
import random

from rlcard.envs.nolimitholdem import NolimitholdemEnv

#
# State node in the public tree
#   - regret values and the policy are computed using cfr
#   - children are added according to growing-tree cfr
#
class StateNode:

    # NOTE - how much of this information should be stored here versus kept in env?

    def __init__(self, public_state, pid, nplayers, player_range : np.array, opponent_values : dict):
        # Public state for this node
        self.public_state = public_state

        # Number of players
        self.nplayers = nplayers
        
        # Player id for the player making the decision
        self.pid = pid

        # List of legal actions
        self.actions = self.public_state['raw_obs']['legal_actions']
        
        # Child states that result from taking action in this state
        self.children = {a: None for a in self.actions}
        
        # Player's probability distribution over actions in this state
        # initalized to a random strategy.
        random_strategy = np.random.rand(len(self.actions))
        random_strategy /= random_strategy.sum()
        self.strategy = dict(zip(self.actions, random_strategy))
        
        # Regret values over possible player actions
        self.regrets  = {a: 0. for a in self.actions}

        # Regret values for the range gadget
        self.gadget_regrets = {a: 0. for a in self.actions}
        
        #
        # CFR value of holding each possible hand according to the current strategy profile
        #
        # This is represented as a 52x52 upper triangular matrix with the diagonal entries
        # set to zero
        #
        # The input opponent values are taken as gadget values, this is to account
        # for the fact that the opponent can steer the game away from this public state
        # if she chooses to.
        #
        # The values computed by cfr for both players are initialized to zero.
        #
        # Note: the goal of CFR is to compute the active player's values.
        #
        self.gadget_values = opponent_values
        self.cfr_values = {a: np.zeros((52, 52), dtype=np.float64) for a in self.actions}

        #
        # Probability of the player holding a hand in the public state
        # 
        # This is represented as a 52x52 upper triangular matrix with the diagonal entries
        # set to zero and the entire matrix is normalized.
        #
        # public_cards = list of indicies corresponding to publicly observed cards
        #
        def random_range():
            #
            # Initialize an upper triangular matrix of random values 
            # with zeros on the diagonal
            #
            rand_range = np.triu(np.random.rand(52, 52), k=1)
            #
            # Set the probability of holding public cards to zero
            #
            rand_range[self.public_cards, :] = 0.
            rand_range[:, self.public_cards] = 0.
            #
            # Normalize
            #
            rand_range /= rand_range.sum()

            return rand_range
        #
        # Note - state['raw_obs']['public_cards'] is a vector of Card objects
        #
        self.public_cards = [card.to_int() for card in self.public_state['raw_obs']['public_cards']]
        self.ranges = [
                        random_range() if i != pid else player_range 
                        for i in range(nplayers)
                      ]

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

    #
    # Given a state node in the public state tree,
    # compute the updated cfr values, cfr regrets, and
    # policies for each infoset in the public state.
    #
    def cfr_values(self, node : StateNode):
        #
        # Base case - the given node is a leaf node
        #
        if not any([child for child in node.children]):
            #
            # If this node is a terminal state in the game, then
            # return the corresponding utilities.
            #
            if self.env.is_over():
                #
                # Note: No actions are taken at terminal states so we do
                #       not need to update regrets or strategies here.
                #
                #
                # Get the set of possible cards in the player's hands
                #
                possible_cards = list(filter(lambda x: x not in node.public_cards, range(52)))
                #
                # Remember the actual cards the players have
                #
                true_hands = [self.env.game.players[i].hand for i in range(node.nplayers)]
                #
                # Iterate over all possible hands in this infoset
                #
                for i, card1 in enumerate(possible_cards):
                    for card2 in possible_cards[i+1:]:
                        #
                        # NOTE - This section needs to be rewritten for more than 2 players
                        #

                        #
                        # Hypothetical opponent hand
                        #
                        hypot_hand = (card1, card2)
                        #
                        # Set hypothetical hand to player 2 and get payoffs
                        #
                        self.env.game.players[1] = hypot_hand
                        #
                        # Get utilities for this hypothetical hand combination
                        #
                        utils = self.env.game.get_payoffs()
                        #
                        # Player 1's value for this hypothetical opponent hand is equal
                        # to the returned payoff weighted by the probability of the
                        # oponent holding the hand.
                        #
                        node.cfr_values[0][card1, card2] = node.ranges[1][card1, card2] * utils[0]
                        #
                        # Reset Player 2's hand to its true value
                        #
                        self.env.game.players[1] = true_hands[1]
                        #
                        # Repeat the steps for Player 2's values
                        #
                        self.env.game.players[0] = hypot_hand
                        utils = self.env.game.get_payoffs()
                        node.cfr_values[1][card1, card2] = node.ranges[0][card1, card2] * utils[1]
                        self.env.game.players[0] = true_hands[0]
            #
            # TODO - handle chance nodes
            #
            elif


    # Public tree counterfactual regret minimization
    # cfr starts on the root state, self.root
    def cfr(self):
        #
        # Run for a fixed number of value updates on the tree.
        #
        for t in range(math.ceil(1/self.n_expansions_per_regret_updates)):
            #
            # Get the values for each player in this public state
            #
            values = self.cfr_values(self.root)

    # Add a leaf node to the game tree
    def grow(self):
        pass

    # Growing Tree Counterfacutal Regret
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

    # Wrapper function around gt-cfr to handle fully solving a 
    # subset of cfvn queries made during gt-cfr
    def training_gt_cfr(self):
        # Check starting game tree
        # We start with a root node and its children
        assert(self.root)
        assert(all([self.root.children[a] for a in self.root.actions]))

        # Run gt-cfr
        values, policy, queries = self.gt_cfr()

        # Fully solve a subset of cvpn queries from this gt_cfr run
        for q in queries:
            if random.random() < self.prob_query_solve:
                self.queries_to_solve.append(q)
        
        # Return gt-cfr results
        return values, policy
    
    #
    # Initialize the starting game tree
    #
    def init_game_tree(self):
        #
        # Get state info from the game environment
        #
        root_player = self.env.get_player_id()
        root_state = self.env.get_state(root_player)
        nplayers = self.env.num_players
        #
        # Compute the player's range at the root state
        #
        #     Case 1: This is the start of the game. The player's
        #     range is the probability of being dealt each hand comination.
        #
        #     Case 2: This is not the start of the game. Use the player's range from the
        #     previous CFR run.
        """
        TODO - impliment this part
        """
        #
        # Compute the opponent player's counterfactual values at the root state
        #
        #     Case 1: This is the start of the game. We need to use a heuristic
        #     to estimate the player's values. This can be done by precomputing the
        #     winning percentage of each hand if the game checked to showdown.
        #
        #     Case 2: This is not the start of the game. Use the opponent players' values
        #     from the previous CFR run.
        """
        TODO - impliment this part
        """
        #
        # Initialize the root node of the public game tree
        #
        self.root = StateNode(root_state, root_player, nplayers)
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


            

            