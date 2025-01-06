# ========================================================================= #
#                                                                           #
# This file defines the Counterfactual Value Network (CVFN)                 #
#                                                                           #
# The model evolves in a cycle:                                             #
#                                                                           #
#    1. GT-CFR querries the network at many leaf nodes in the game tree     #
#       while solving a game state encountered during self-play.            #
#                                                                           #
#    2. A subset of these querries are selected to be solved fully using    #
#       GT-CFR which produces its own policy and value estimate.            #
#                                                                           #
#    3. The result of fully solving a query is then used as a target        #
#       in the network's training set.                                      #
#                                                                           #
#                                                                           #
# Note 1: This cycle does not have to be sequential, a cvfn thread can      #
#         run gt-cfr to solve a game state in its buffer while a self-play  #
#         thread runs gt-cfr to solve a game state encountered during       #
#         self-play.                                                        #
#                                                                           #
# Note 2: This is a self reinforcing cycle. Network updates improve         #
#         gt-cfr solver outputs which in turn produce more accurate         #
#         training targets for network updates. And so on and so forth.     #
#                                                                           #
# ========================================================================= #

# External imports
from multiprocessing import Process, Queue
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import time

# Internal imports
from rlcard.agents.gt_cfr_agent.fifo_buffer import FIFOBuffer
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRSolver
from rlcard.games.nolimitholdem.game import NolimitholdemGame

#
# The goal of the Counterfactual Value Network is to estimate
# the acting player's strategy and all the player's expected values
# for a given game state.
#
class CounterfactualValueNetwork:

    #
    # Counterfactual Value Network initialization
    # 
    # Using the input parameters, 
    # construct a multi-layer perceptron.
    #
    # Network input:
    #   - feature vector characterizing a game state constructed by self.to_vect(...)
    #
    # Network output:
    #
    #   - Stragety - the acting player's strategy at the game state
    #                shape=(num_actions, 1326)
    #                prob. of taking illegal actions in this state should be zero.
    #
    #   - Values - value distribution over hands for each player at the game state
    #              shape=(num players, 1326)
    #
    # Note 1: For now, the network is hard coded to use a MLP architecture
    #         because that is what was used in the literature.
    #
    # Note 2: Network parameters default to the values used in the literature.
    #
    def __init__(self, num_players: int =2,                     # Tot. num. of players in the game
                       num_actions: int =5,                     # Tot. num. of actions in the game
                       num_neurons_per_layer: int =2048,        # Neurons per layer
                       num_layers: int =6,                      # Num. of hidden layers
                       activation_func: str ='relu',            # Activation function for hidden layers
                       batch_size: int =1024,                   # Batch size
                       optimizer: str ='adam',                  # Optimizer
                       init_learning_rate: float =0.0001,       # Initial learning rate
                       decay_rate: float =0.5,                  # Rate at which the lr is decayed
                       decay_steps: int =int(2e6),              # Num. steps lr is decayed by decay_rate
                       policy_w: float =0.01,                   # Policy head weight
                       values_w: float =1,                      # Value  head weight
                       max_replay_buffer_size: int =int(1e6),   # Max replay buffer size 
                       max_grad_updates_per_exmpl: int =10,     # Max times an exmpl can be used in the replay buffer
                       q_recursive: float =0.1,                 # Prob. of adding a recursive query to the query queue
                       n_query_solvers: int =2):                # Num. of procs solving querries off the query queue
        #
        # Store training parameters
        #
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.policy_w = policy_w
        self.values_w = values_w

        #
        # Initialize the optimizer
        #
        # NOTE - Right now only the adam optimizer is supported
        #
        if optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.init_learning_rate)
        else:
            raise f"Unsupported optimizer type: {optimizer}"

        #
        # Compute the input dimension
        # according to the given number of players in the game
        #
        self.num_players = num_players
        self.num_actions = num_actions
        self.input_dim = (2*self.num_players + 53) + (1326*self.num_players) # See to_vect()

        #
        # Set the data type for the model's inputs/outputs
        #
        # NOTE - For now hard code this to float64, in the future
        #        lower precision options could be interesting to explore.
        #
        self.data_type = np.float64

        #
        # --> Construct the MLP network
        #
        # Input layer
        #
        inputs = Input(shape=(self.input_dim,))
        #
        # Add hidden layers
        #
        layer = inputs
        for _ in range(num_layers):
            layer = Dense(num_neurons_per_layer, activation=activation_func)(layer)
        #
        # Strategy output, (num_actions, 1326)
        #
        # NOTE - I think 'softmax' is the activation we want here because it yeilds
        #        normalized probability estimates.
        #
        strategy_output = Dense(num_actions * 1326, activation='softmax')(layer)
        strategy_output = tf.keras.layers.Reshape((num_actions, 1326))(strategy_output)
        #
        # Values output, (num_players, 1326)
        #
        # NOTE - Is there a better activation than 'linear'?
        #
        values_output = Dense(num_players * 1326, activation='linear')(layer)
        values_output = tf.keras.layers.Reshape((num_players, 1326))(values_output)
        #
        # Putting it all together into a model object
        #
        self.network = Model(inputs=inputs, outputs=[strategy_output, values_output])

        #
        # This is a queue of querries to be fully solved using GT-CFR
        #
        # Querries are added to this queue by the GT-CFR algorithm
        #
        #    Two objects run GT-CFR
        #
        #        1. GT-CFR is run during self-play, this is the most common.
        #
        #        2. GT-CFR is run by the CFVN when fully solving querries.
        #           Note: querries that are added by the cvfn are called "recursive querries"
        #
        # Querries are taken off this queue by cvfn workers who fully solve
        # the query using gt-cfr and add the result to the replay buffer.
        #
        self.q_recursive = q_recursive # prob. of adding a recursive query to the queue
        self.query_queue = Queue()

        #
        # Initialize query solver processes
        #
        # These processes work in parallel.
        #
        # Query solver loop:
        #   1. Pull querries off the query queue
        #   2. Fully solve the query using GT-CFR
        #   3. Add the solved targets onto the replay buffer
        #   4. Repeat.
        #
        assert n_query_solvers >= 0, "Cant have a negative number of workers"
        self.n_query_solvers = n_query_solvers # Num. of solver processes
        self.init_query_solvers()

        #
        # This is a buffer of solved querries.
        #
        # Note: this is the CVFN training set.
        #
        # Solved querries are add to the replay buffer in two ways:
        #
        #    1. By query solver workers that fully solve queries off the query queue
        #
        #    2. By the GT-CFR agent after self-play game states have been solved
        #       (Note: this feature is DISABLED by default)
        #
        # Note: this is a thread safe FIFO buffer of fixed size
        #
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer = FIFOBuffer(max_size=max_replay_buffer_size,
                                        evict_after_n_samples=max_grad_updates_per_exmpl)

    #
    # Initialize the process pool for query solvers.
    #
    def init_query_solvers(self) -> None:
        #
        # If the class constructor was called with 0 workers set,
        # then we disable asynchronous query solving.
        #
        # Note: If the query solvers are disabled, then the process of 
        #       solving querries on the query queue and moving the solved 
        #       targets to the replay buffer needs to be handled somewhere else.
        #
        #       Practically speaking, the query solvers should never be disabled.
        #       But, for debugging purposes, it can be useful to control the
        #       flow of querries manually.
        #
        if self.n_query_solvers == 0:
            self.solvers = None
            return
        #
        # Initialize the list of GT-CFR solvers for each worker
        #
        # self.solvers[worker id]
        #   = GT-CFR solver object for the query solver with thread id = worker id
        #
        self.solvers = [GTCFRSolver(input_cfvn=self, prob_query_solve=self.q_recursive) for _ in self.n_query_solvers]
        #
        # Spawn the worker processes
        #
        for worker_id in range(self.n_query_solvers):
            p = Process(target=self.query_solver_loop, args=(worker_id))
    
    #
    # Loop for the query solver tasks
    #
    # worker_id - id of the thread executing this function
    #
    def query_solver_loop(self, worker_id: int) -> None:
        #
        # Loop indefinately...
        #
        while True:
            #
            # Fail gracefully if a worker thead encounters
            # an error when solving a query.
            #
            try:
                #
                # Get the next query off the query queue
                #
                # Wait if the query queue is empty
                #
                # query 
                #    = tuple(input vector, player values, player strategy)
                #
                query = self.query_queue.get(block=True)
                #
                # Solve the query
                #
                target_strat, target_values = self.solvers[worker_id].solve(*query)
                #
                # Place the solved query on the replay buffer
                #
                self.replay_buffer.put((query[0].to_vect(), target_strat, target_values))
            
            #
            # Output the worker error to the console
            #
            except Exception as e:
                print(f'Worker {worker_id} error: {e}')
        

    #
    # Given a game object and player ranges,
    # return a feature vector characterizing the state.
    #
    # Featurizing the game object,
    #
    #   - N hot encoding of board cards (52)
    #
    #   - each player's commitment to the pot, normalized by their stack (num. players)
    #
    #   - 1 hot encoding of the acting player, including the 'chance player' (num. players + 1)
    #
    #     Note: the input chance_actor bool flag indicates when the chance player is acting,
    #           since this information is not included in the game object.
    #
    # Total length of featurized game state
    #     = 52 + 2 * num_players + 1 = 57 (for a 2 player game)
    #
    # Featurizing the player ranges,
    #
    #   - 1326 possible hand combinations** (i.e. infosets), for each player
    #
    # Total length of player ranges
    #     = 1326 * num_players = 2652 (for a 2 player game)
    #
    # ** Note - Possible hand combinations = 52 choose 2 = 1326 hands
    #           Each hand is a pairing of two distinct cards where order does not matter
    #
    # Returned vector = concat(featurized game vector, featurized player ranges)
    #                 = 2709 features (for a 2 player game)
    #
    # Note: These are the features used in the literature. Other features could be
    #       explored in the future.
    #
    def to_vect(self, game : NolimitholdemGame, ranges : np.ndarray, chance_actor : bool):
        #
        # All elements of the returned vector must be of the same type.
        #
        # For now, we hard code this to be np.float64, the largest data type
        # of the features.
        #
        self.data_type = np.float64
        #
        # --> Featurize game object
        #
        # N-hot encoding of public cards
        #
        public_card_encoding = np.zeros(52, dtype=self.data_type)
        public_card_encoding[[card.to_int() for card in game.public_cards]] = 1
        #
        # Player pot commitments (normalized by stack size at the start of the hand)
        #
        pot_commitments = np.array([player.in_chips / (player.remained_chips + player.in_chips)
                                        for player in game.players], dtype=self.data_type)
        #
        # 1-hot acting player encoding
        #
        acting_player = np.zeros(game.num_players + 1, dtype=self.data_type)
        acting_player[-1 if chance_actor else game.game_pointer] = 1
        #
        # --> Featurize player ranges
        #
        # The input range for each player is a 52x52 upper triangular matrix
        # with zeros across the diagonal.
        #
        # Each of these matrices needs to be flattened into 1326 vector.
        #
        # Note: These matrices have at most 1326 non-zero entries, 
        #       matching our expected vector size.
        #
        #       Upper triangular matrix elements = 52*(52+1)/2 = 1378 non-zero elems
        #       
        #       Exluding diagonal entries = 1378 - 52 = 1326 non-zero elems = num. hand combos 
        #
        assert self.num_players == game.num_players, "Game state and network num players mismatch"
        assert ranges.shape == (self.num_players, 52, 52), "Unrecognized player range shape"
        range_vect = np.zeros(1326*game.num_players, dtype=self.data_type)
        triu_indices = np.triu_indices(52, k=1) # indicies of elems in the upper triangular matrix
        for pid in range(game.num_players):
            range_vect[1326*pid:1326*(pid+1)] = ranges[pid][triu_indices]
        #
        # --> Concat the game and range features into a single feature vector
        #
        return np.concatenate(
                                [
                                  public_card_encoding, 
                                  pot_commitments,
                                  acting_player,
                                  range_vect
                                ]
                            )

    #
    # Run inference
    #
    # Given an input vector encoding a game state and
    # player ranges at the game state, return the policy
    # for the acting player and values for the players.
    #
    # Note: the input should come from the to_vect() function.
    #
    def query(self, input : np.ndarry) -> tuple[np.ndarray]:
        assert input.shape == (self.input_dim,), "Unexpected input dimension"
        return self.network(input)

    #
    # Add a query to the query queue for solving
    #
    def add_to_query_queue(self, query: np.ndarray) -> None:
        assert query.shape == (self.input_dim,)
        self.query_queue.put(query)

    #
    # Add a solved query to the replay buffer
    #
    # Solved query
    #   = tuple(input vector, output strategy, output values)
    #
    def add_to_replay_buffer(self, solved_query: tuple[np.ndarray]) -> None:
        #
        # Verify the given solved query is well-formatted
        #
        assert len(solved_query) == 3, "Solved query must be of format, (input, output strategy, output values)"
        assert all(isinstance(solved_query[i], np.ndarray) for i in range(3)), "Unexpected solved query type"
        assert solved_query[0].shape == (self.input_dim,), "Unexpected input dimension"
        assert solved_query[1].shape == (self.num_actions, 1326), "Unexpected output strategy dimension"
        assert solved_query[2].shape == (self.num_players, 1326), "Unexpected output values dimension"
        #
        # Add the solved query to the FIFO buffer
        #
        self.replay_buffer.put(solved_query)

    #
    # Sample batch_size number of training targets from the repaly buffer
    #
    # Lump the targets into numpy arrays
    #
    #   - inputs, shape=(batch_size, input_dim)
    #
    #   - target_strategy, shape=(batch_size, num_actions, 1326)
    #
    #   - target_values, shape=(batch_size, num_players, 1326)
    #
    # Return these arrays.
    #
    def get_batch_data(self) -> tuple[np.ndarray]:
        #
        # Sample query targets from the replay buffer.
        #
        # query_targets = [(input, strategy, values), ...]
        #
        # Note: The sampling process has a bias towards selecting
        #       newer training targets as they should be more
        #       accurate than older targets.
        #
        query_targets = self.replay_buffer.sample(self.batch_size)
        #
        # Initialize the output arrays
        #
        inputs = np.zeros((self.batch_size, self.input_dim), dtype=self.data_type)
        target_strategies = np.zeros((self.batch_size, self.num_actions, 1326), dtype=self.data_type)
        target_values = np.zeros((self.batch_size, self.num_players, 1326), dtype=self.data_type)
        #
        # Populate the output arrays
        #
        for i, token in enumerate(query_targets):
            inputs[i, :] = token[0]
            target_strategies[i, :, :] = token[1]
            target_values[i, :, :] = token[2]
        #
        # Return the output arrays
        #
        return inputs, target_strategies, target_values
    
    #
    # Perform a single batch update on the network
    #
    # Return the model's batch loss for this update
    #
    def batch_update(self) -> float:
        #
        # Get batch data
        #
        inputs, target_strats, target_values = self.get_batch_data()
        #
        # Update the learning rate
        #
        num_iters = tf.keras.backend.get_value(self.optimizer.iterations)
        learning_rate = self.init_learning_rate * (self.decay_rate ** (num_iters / self.decay_steps))
        tf.keras.backend.set_value(self.optimizer.learning_rate, learning_rate)
        #
        # Perform prediction and compute the total loss
        #
        with tf.GradientTape() as tape:
            #
            # Forward pass
            #
            pred_strats, pred_values = self.network(inputs, training=True)
            #
            # Compute losses
            #
            # Strategy loss = KL Divergence loss
            #
            strat_loss = tf.reduce_mean(
                tf.keras.losses.KLDivergence()(target_strats, pred_strats)
            )
            #
            # Value loss = Huber loss
            #
            value_loss = tf.reduce_mean(
                tf.keras.losses.Huber()(target_values, pred_values)
            )
            #
            # Total loss, weighted sum.
            #
            total_loss = self.policy_w * strat_loss + self.values_w * value_loss
        #
        # Compute gradients
        #
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        #
        # Apply gradients using Adam optimizer
        #
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        #
        # Return the total loss as a numpy array
        #
        return total_loss
    
    #
    # Train the network
    #
    # niters = number of batch updates to run per train() call
    #
    # wait_time = number of seconds to wait for new targets on the replay buffer.
    #
    # By default, niters -> infinity, training updates run continously.
    #
    def train(self, niters: int =np.inf, wait_time: int =0):
        #
        # Track the number of updates we perform
        #
        update_iter = 0
        #
        # Perform batch updates...
        #
        while update_iter < niters:
            #
            # Only run a batch update if we have enough targets
            # in the replay buffer.
            #
            if self.replay_buffer.size() >= self.batch_size:
                #
                # Perform batch update
                #
                update_loss = self.batch_update()
                #
                # NOTE - In the future, we should be storing the loss updates
                #        to plot them over time.
                #
                # Print the update loss every 1000 updates
                #
                if update_iter % 1000 == 0:
                    print(f'--> {update_iter}: Batch loss {update_loss}')
            #
            # Else, we need to wait for cfvn solver workers to put
            # more targets on the replay buffer.
            #
            # Note: In practice, the replay buffer should rarely, if ever,
            #       have less than 32 targets in it.
            #
            else:
                time.sleep(wait_time)
            #
            # Increment batch counter
            #
            update_iter += 1