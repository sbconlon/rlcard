from __future__ import annotations  # Enables forward references

# External imports
import copy
import math
import numpy as np
import random
import sys
import time
from typing import TYPE_CHECKING

# Internal imports
from rlcard.agents.gt_cfr_agent.nodes import CFRNode, DecisionNode, ChanceNode
from rlcard.agents.gt_cfr_agent.utils import uniform_range, random_range, starting_hand_values
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
        # encountered during GT-CFR solving.
        #
        assert 0 <= prob_query_solve <= 1, "Invalid solve query probability"
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
        self.root = None
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
        if input_opponents_values is not None:
            self.terminate_values = input_opponents_values
        else:
            self.terminate_values = starting_hand_values(input_game) * input_game.dealer.pot # = t_values = v_2 in the literature
        self.gadget_regrets = np.zeros((2, 52, 52)) # 2 gadget actions, (Follow, Terminate)
        self.gadget_values = np.zeros((52, 52))
    
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
        return self.root.search(target_game.trajectory)

    #
    # Given the acting player's range, assign randomized ranges
    # to the opponent players.
    #
    def compute_initial_ranges(input_game, player_range):
        #
        # Allocate memory for the player ranges,
        # initialized to zero.
        #
        nplayers = input_game.num_players
        ranges = np.zeros((nplayers, 52, 52))
        #
        # For each player in the game...
        #
        for pid in range(nplayers):
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
            player_ranges = GTCFRSolver.compute_initial_ranges(input_game, player_range)
            #
            # Initialize the root node of the public game tree
            #
            # NOTE - Do we need this to be a deepcopy?
            #        Nothing should be editing the game state while we solve.
            #
            self.root = DecisionNode(copy.deepcopy(input_game), player_ranges)
            self.full_tree = False
            #
            # Initialize the gadget game
            #
            self.init_gadget_game(input_game)
            #
            # The root node is the decision node we are solving
            #
            self.trajectory_seed = []
            """
            # DEBUG - Activate the full game tree for debugging purpuses
            self.root.activate_full_tree()
            """
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
            # Use the node's info to start the new tree
            #
            # NOTE - we could use the subtree rooted at the result node from the prior solve iteration;
            #        however, the tree will grow too large after too many successive solves, resulting
            #        in too slow of execution.
            #
            self.root = DecisionNode(result.game, result.player_ranges)
            #
            # Take the opponent's values from the prior solve iteration to be the gadget values
            #
            # NOTE - For now, the gadget game is only defined for a 2 player game.
            #
            pid = input_game.game_pointer
            self.init_gadget_game(input_game, input_opponents_values=np.copy(result.values[(pid+1) % 2]))
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
            # The acting player's range is given, randomize the opponent ranges.
            #
            player_ranges = GTCFRSolver.compute_initial_ranges(input_game, input_player_range)
            #
            # Initialize the root node with the player's values
            #
            # NOTE - Same as Case 1, not sure if this needs to be a deepcopy.
            #
            self.root = DecisionNode(copy.deepcopy(input_game), player_ranges)
            self.full_tree = False
            #
            # Initialize the gadget game with the opponent's values
            #
            self.init_gadget_game(input_game, input_opponents_values=input_opponent_values)
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
        CFRNode.set_root_pid = input_game.game_pointer
        #
        # Store a reference to the CFVN in the game tree
        #
        CFRNode.set_cfvn(self.cfvn)
        #
        # If we are solving the full game tree, then activate all nodes in the tree.
        #
        if self.full_solve:
            self.root.activate_full_tree()
            self.full_tree = True # The tree is now full
            self.decision_point = self.root
        else:
            #
            # Activate the root node
            #
            if not self.root.is_active:
                self.root.activate()
            #
            # Activate the nodes along the seed trajectory,
            # if a seed trajectory is given. Then, set
            # the decision point node.
            #
            if self.trajectory_seed:
                node = self.root
                for value in self.trajectory_seed:
                    # Get next node in the trajectory
                    if isinstance(node, DecisionNode):
                        action = Action(value)
                        node = node.children[action]
                    elif isinstance(node, ChanceNode):
                        node = node.outcomes[value]
                    else:
                        raise ValueError('Terminal node should not be encountered on a seed trajectory')
                    # Activate the node
                    if not node.is_active:
                        node.activate()
                self.decision_point = node
            #
            # Otherwise, the root node is the decision point
            #
            else:
                self.decision_point = self.root
            #
            # Active the decision point's children
            #
            for child in self.decision_point.children.values():
                if not child.is_active:
                    child.activate()

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
        gadget_regrets_positives = np.maximum(self.gadget_regrets, 0) # Should already be non-negative

        """
        # DEBUG
        print()
        print(f'Regrets - Follow: {gadget_regrets_positives[0, card1, card2]}  Terminate: {gadget_regrets_positives[1, card1, card2]}')
        print(f'Value  -  {self.gadget_values[card1, card2]}')
        """

        denom = gadget_regrets_positives[0] + gadget_regrets_positives[1]
        safe_denom = np.where(denom == 0, 1, denom) # remove zeros from denom to avoid dividing by zero
        gadget_follow_strat = np.where(denom == 0, 0.5, gadget_regrets_positives[0] / safe_denom)

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
        public_card_idxs = [c.to_int() for c in self.root.game.public_cards]
        for idx in public_card_idxs:
            gadget_follow_strat[:, idx] = 0.
            gadget_follow_strat[idx, :] = 0.
        gadget_follow_strat[np.tril_indices(52)] = 0.

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
        opp_pid = (self.root.game.game_pointer + 1) % 2
        if np.sum(gadget_follow_strat) != 0:
            self.root.player_ranges[opp_pid] = gadget_follow_strat / np.sum(gadget_follow_strat)
        else:
            self.root.player_ranges[opp_pid] = gadget_follow_strat # ALL ZEROS

        #
        # Compute the updated gadget values
        #
        # This is a standard expected value computation.
        #
        new_gadget_values = (gadget_follow_strat * self.root.values[opp_pid] + 
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
        self.gadget_regrets[0] = np.maximum(self.gadget_regrets[0] + self.root.values[opp_pid] - new_gadget_values, 0) # gadget value @ t
        self.gadget_regrets[1] = np.maximum(self.gadget_regrets[1] + self.terminate_values - self.gadget_values, 0)    # gadget value @ t + 1

        #
        # Update the gadget values to the new values
        #
        self.gadget_values = new_gadget_values
        
        """
        # DEBUG
        import ipdb; ipdb.set_trace()
        """

    #
    # Public tree counterfactual regret minimization.
    #
    # CFR starts on the root state, self.root, and recurses down through
    # the game tree nodes.
    #
    def cfr(self, train=False) -> None:
        #
        # Run for a fixed number of value updates on the tree.
        #
        got, added = 0, 0
        for i in range(math.ceil(1/self.n_expansions_per_regret_updates)):
            #print(f'{i} / {math.ceil(1/self.n_expansions_per_regret_updates)}')
            #
            # Perform one iteration of value and strategy updates on the game tree
            #
            """
            self.root.print_tree()
            import ipdb; ipdb.set_trace()
            """
            querries = self.root.update_values()
            #
            # Fully solve a subset of cfvn queries from this cfr update
            #
            got += len(querries)
            if train:
                #print(f'Total querries generated: {len(querries)}')
                for q in querries:
                    if random.random() < self.prob_query_solve:
                            added += 1
                            self.cfvn.add_to_query_queue(q)
            #
            # Update gadget regrets
            #
            self.update_gadget_regrets()
        print(f'Got {got} total querries, added {added}')
        """
        print(self.root.print_tree())
        import ipdb; ipdb.set_trace()
        """

    #
    # Add a node to the game tree
    #
    # Which node to add is determined by sampling a hand configuration
    # weighted by the player's ranges at the root node, then actions are 
    # sampled down the tree until an action for which the resulting node 
    # is not activated. Then that node is activated.
    #
    # If the tree is full, then set a parameter to skip growth attempts
    # in the future.
    #
    def grow(self) -> None:
        #
        # Skip the grow attempt if the game tree is full
        #
        if self.full_tree:
            return
        #
        # The acting player recieves their true hand assignemnt. 
        #
        # All other player's hand assignments are sampled at random, weighted 
        # by their ranges at the decision point.
        #
        hands = [None] * self.root.game.num_players # list of player's hands
        used_cards = {card.to_int() for card in self.decision_point.game.public_cards} # track cards that've been used by previous players
        #
        # Give the acting player their true hand
        #
        pid = self.decision_point.game.game_pointer
        hands[pid] = tuple(card.to_int() for card in self.decision_point.game.players[pid].hand)
        used_cards.add(hands[pid][0])
        used_cards.add(hands[pid][1])
        #
        # For each opponent player...
        #
        for opp_pid in range(self.decision_point.game.num_players):
            #
            # Skip the acting player
            #
            if pid == opp_pid:
                continue
            #
            # Get the probability the player is holing each hand
            #
            hand_probs = np.copy(self.decision_point.player_ranges[opp_pid])
            
            #
            # If a given decision node is so bad for an opponent that
            # they follow strategy is all zeros, then we can just
            # sample a hand at random.
            #
            if np.sum(hand_probs) == 0:
                hand_probs = np.triu(np.ones(hand_probs.shape), k=1)
            
            #
            # Mask out cards that have already been taken
            #
            for card_idx in used_cards:
                hand_probs[card_idx, :] = 0.
                hand_probs[:, card_idx] = 0.

            hand_probs /= hand_probs.sum()
            """
            # DEBUG
            self.decision_point.check_matrix(np.array([hand_probs]))
            """
            #
            # Sample a hand
            #
            idx = np.random.choice(hand_probs.size, p=hand_probs.flatten())
            #
            # Convert the 1D idx back to 2D hand, (card1, card2)
            #
            hand = divmod(idx, 52)
            #
            # Check that the hand is valid
            #
            assert hand[0] < hand[1], 'Ill-formatted hand selected'
            #
            # Add the hand to the hand list
            #
            hands[opp_pid] = hand
            #
            # Update used cards set
            #
            used_cards.add(hand[0])
            used_cards.add(hand[1])
        #
        # Try to add a node to the subtree,
        # using the given hand to sample a trajectory
        #
        self.full_tree = not self.decision_point.grow_tree(hands)
    
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
            self.grow()
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
        assert self.root, "The game tree needs to be initialized before GT-CFR is ran"
        assert all([self.root.children[a] for a in self.root.actions]), "Ill-formatted initial tree"
        #
        # Run gt-cfr
        #
        self.gt_cfr(train=True)

    #
    # Return a policy and value estimate for the given game state using gt-cfr
    #
    def solve(self, game: NolimitholdemGame, 
                    input_opponent_values: np.ndarray = None,
                    input_player_range: np.ndarray = None,
                    trajectory_seed: list[int] = None) -> tuple[np.ndarray, np.ndarray]:
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
        self.init_game_tree(game, input_opponent_values, input_player_range, trajectory_seed)
        #
        # GT-CFR training run 
        #
        self.training_gt_cfr()
        #
        # Return the computed strategies and values for the root node
        #
        print(f'Solve time: {time.time() - start}')
        return np.copy(self.decision_point.cummulative_strategy()), np.copy(self.decision_point.values)   
    
    #
    # Reset
    #
    def reset(self) -> None:
        # Remove the game tree
        self.root, self.decision_point = None, None

#
# This function handles the self-play loop that uses the GT-CFR solver 
# to solve game states encountered during self-play episodes.
#
class GTCFRAgent():
    #
    # Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    #
    def __init__(self, env: NolimitholdemEnv, full_solve: bool =False, prob_query_solve: int =0.9, n_expansions: int =10, n_expansions_per_regret_update: float =0.01, n_trainers: int =1, n_query_solvers: int=1):
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
            n_query_solvers=n_query_solvers
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
            ev = cfr_values[pid, hand_idxs[0], hand_idxs[1]]
            cfr_policy = cfr_policies[:, hand_idxs[0], hand_idxs[1]]
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
            mixed_policy = (1-self.epsilon) * cfr_policy + self.epsilon * uniform_policy

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

