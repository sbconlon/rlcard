import numpy as np
import random

from rlcard.envs.nolimitholdem import NolimitholdemEnv

class GTCFRAgent():
    ''' Implement Growing Tree Counter Factual Regret (GT-CFR) algorithm
    '''
    def __init__(self, env: NolimitholdemEnv):
        # Poker environment
        self.env = env
        
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

    # Return a policy and value estimate for the current game state using gt-cfr
    def self_play_controller():
        pass

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

            # Comput counterfactual values and the average policy for the
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


            

            