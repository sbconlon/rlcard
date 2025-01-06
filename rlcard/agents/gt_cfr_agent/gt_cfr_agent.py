# External imports
import copy
from __future__ import annotations  # Enables forward references
import math
import numpy as np
import random
from typing import TYPE_CHECKING

# Internal imports
from rlcard.agents.gt_cfr_agent.nodes import CFRNode, DecisionNode
from rlcard.agents.gt_cfr_agent.utils import uniform_range, starting_hand_values
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.nolimitholdem.game import NolimitholdemGame

# Avoid a circular import
if TYPE_CHECKING:
    from rlcard.agents.gt_cfr_agent.cvfn import CounterfactualValueNetwork # Only imports for type hints

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
    def init_gadget_game(self, input_game: NolimitholdemGame, input_opponents_values: np.ndarray =None):
        if input_opponents_values:
            self.terminate_values = input_opponents_values
        else:
            self.terminate_values = starting_hand_values(input_game) # = t_values = v_2 in the literature
        self.gadget_regrets = np.zeros(2, 52, 52) # 2 gadget actions, (Follow, Terminate)
        self.gadget_values = np.zeros(52, 52)
    
    #
    # Search the game tree for the input game state,
    # along the given trajectory.
    #
    def node_search(self, target_game: NolimitholdemGame) -> DecisionNode:
        #
        # Can't search a null tree
        #
        if self.root is None:
            return None
        #
        # Return the node in the game tree, or its closest ancestor
        #
        return self.root.search(target_game.trajectory, target_game.game_pointer)

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
    #      (self.root=None, opponent_values=None, player_range=None, trajectory_seed=None)
    #
    #        - Opponent's range is estimated using a heuristic
    #
    #        - Player's range is assumed to be uniform across all possible hand combinations
    #
    #        - self.root is initialized at the given game state
    #
    #        * This is used by gt_cfr_agent at the start of an episode
    #
    #   Case 2 - Evaluated game tree from a previous solve call
    #      (self.root=DecisionNode(...), opponent_values=None, player_range=None, trajectory_seed=None)
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
    #   Case 3 - Empty game tree. Given opponent values, player range, and action seeds.
    #      (self.root=None, opponent_values=np.array(...), player_range=np.array(...), trajectory_seed=[...])
    #
    #        - Initialize the root node with the input opponent values and player range
    #
    #        - Add the nodes to the game tree corresponding to the game states along the given trajectory seed
    #
    #        * This is used by the CFVN during training
    #
    def init_game_tree(self, input_game: NolimitholdemGame,
                             input_opponent_values: np.ndarray =None,
                             input_player_range: np.ndarray =None,
                             trajectory_seed: list[int] =None) -> None:
        #
        # Case 1 - No starting information
        #
        if all(x is None for x in (self.root, input_opponent_values, input_player_range, trajectory_seed)):
            #
            # Assume the decision player's range at the root state is
            # distributed uniformly over possible hands.
            #
            # The opponent players' ranges are randomized during node initialization.
            #
            player_range = uniform_range(input_game.public_cards)
            #
            # Initialize the root node of the public game tree
            #
            self.root = DecisionNode(copy.deepcopy(input_game), player_range)
            #
            # Initialize the gadget game
            #
            self.init_gadget_game(input_game)
            #
            # The root node is the decision node we are solving
            #
            self.trajectory_seed = []
        #
        # Case 2 - Existing game tree. No input player range or opponent values.
        #
        elif self.root is not None and all(x is None for x in (input_opponent_values, input_player_range, trajectory_seed)):
            #
            # Search the game tree for the input game state
            #
            result = self.node_search(input_game)
            assert result is not None, "Input game not found in the existing game tree"
            #
            # Make this the new root node
            #
            self.root = result
            #
            # Take the opponent's values to be the gadget values
            #
            # Note: the player's range is already stored in the root node.
            #
            # NOTE - For now, the gadget game is only defined for a 2 player game.
            #
            pid = input_game.game_pointer
            self.init_gadget_game(input_game, opponents_values=np.copy(self.root.values[(pid+1) % 2]))
            #
            # Reset values to zero for the players
            #
            self.root.zero_values()
            #
            # If the trajectory of root node differs from the
            # input game state, then store the difference as the trajectory seed.
            #
            assert len(input_game.trajectory) >= len(self.root.game.trajectory)
            self.trajectory_seed = input_game.trajectory[len(result.game.trajectory):]
        #
        # Case 3 - Empty game tree. Input range and values.
        #
        elif self.root is None and all(x is not None for x in (input_player_range, input_opponent_values)):
            #
            # Initialize the root node with the player's values
            #
            self.root = DecisionNode(copy.deepcopy(input_game), np.copy(input_player_range))
            #
            # Initialize the gadget game with the opponent's values
            #
            self.init_gadget_game(input_game, opponents_values=input_opponent_values)
            #
            # Set the trajectory seed
            #
            self.trajectory_seed = trajectory_seed if trajectory_seed is not None else []
        #
        # Error case - Unrecognized initialization input configuration
        #
        else:
            raise ValueError("Input values for initialization not recognized")        
        #
        # Save the player's id in the game tree
        #
        CFRNode.set_root_pid = pid
        #
        # Store a reference to the CVFN in the game tree
        #
        CFRNode.set_cvfn(self.cfvn)
        #
        # Activate the root node
        #
        if not self.root.is_active:
            self.root.activate()
        #
        # Activate the nodes along the seed trajectory,
        # if a seed trajectory is given.
        #
        if self.trajectory_seed:
            node = self.root
            for i in self.trajectory_seed:
                if not node.is_active:
                    node.activate()
                if isinstance(node, DecisionNode):
                    node = node.children[self.trajectory_seed[i]]
                elif isinstance(node, ChanceNode):
                    node = node.outcomes[self.trajectory_seed[i]]
                else:
                    raise ValueError("Terminal node should not be encountered on a seed trajectory")
            self.decision_point = node
        #
        # Otherwise, activate the root node's children
        #
        else:
            for child in self.root.children:
                if not child.is_active:
                    child.activate()
            self.decision_point = self.root

    #
    # Update the regrets in the gadget game
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
        gadget_regrets_positives = np.maximum(self.gadget_regrets, 0)
        gadget_follow_strat = gadget_regrets_positives[0] / (gadget_regrets_positives[0] + gadget_regrets_positives[1])

        #
        # Set the opponent's range in the cfr root node to the gadget's follow strategy 
        #
        # Reasoning:
        #     Gadget follow strat = probability of choosing to play toward the cfr root state
        #     Opp. root range = prob. of the opp. reaching this state given her strategy
        #     Therefore, gadget follow strat = opponent's range at the root node. 
        #
        opp_pid = (self.root.game.game_pointer + 1) % 2
        self.root.player_ranges[opp_pid] = gadget_follow_strat

        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        new_gadget_values = (gadget_follow_strat * self.gadget_values[0] + 
                             (1 - gadget_follow_strat) * self.gadget_values[1])

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
        self.gadget_regrets[0] += self.root.values[opp_pid] - new_gadget_values[0] # gadget value @ t
        self.gadget_regrets[1] += self.terminate_values - self.gadget_values[1] # gadget value @ t + 1

        #
        # Update the gadget values to the new values
        #
        self.gadget_values = new_gadget_values

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
            # Update gadget regrets
            #
            self.update_gadget_regrets()

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
        for pid in range(self.decision_point.game.num_players):
            #
            # Flatten the player's range to make sampling easier
            #
            hand_probs = self.decision_point.player_ranges[pid].flatten()
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
        while not decision_point.grow_tree(hands) and attempts < max_attempts:
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
        # Fully solve a subset of cvpn queries from this gt_cfr run
        #
        for q in queries:
            if random.random() < self.prob_query_solve:
                self.cvfn.add_to_query_queue(q)

    #
    # Return a policy and value estimate for the given game state using gt-cfr
    #
    def solve(self, game: NolimitholdemGame, 
                    input_opponent_values: np.ndarray = None,
                    input_player_range: np.ndarray = None,
                    trajectory_seed: list[int] = None) -> tuple[np.ndarray, np.ndarray]:
        #
        # Initialize the game tree for cfr
        #
        self.init_game_tree(game, input_opponent_values, input_player_range, trajectory_seed)
        #
        # GT-CFR training run 
        #
        self.training_gt_cfr()
        #
        # Return the computed strategies and values for the root node
        #
        return np.copy(self.decision_point.strategy), np.copy(self.decision_point.values)

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
    # policies and values.
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
        # Add to replay buffer
        #
        if self.prob_add_to_buffer > 0: # disabled by default
            for token in trajectory: 
                if random.random() < self.prob_add_to_buffer:
                    self.solver.cfvn.add_to_replay_buffer(token)