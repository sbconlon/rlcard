from __future__ import annotations  # Enables forward references

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
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
from typing import TYPE_CHECKING
import time

# Internal imports
from rlcard.agents.gt_cfr_agent.replay_buffer import ReplayBuffer
from rlcard.agents.gt_cfr_agent.utils import normalize_columns
from rlcard.agents.gt_cfr_agent.rwlock import ReadRWLock, WriteRWLock, ReadWriteLock
from rlcard.games.nolimitholdem.round import Action
from rlcard.games.nolimitholdem.game import NolimitholdemGame

if TYPE_CHECKING:
    from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRSolver

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
    #
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
                       num_actions: int =7,                     # Tot. num. of actions in the game
                       num_neurons_per_layer: int =2048,        # Neurons per layer
                       num_layers: int =6,                      # Num. of hidden layers
                       activation_func: str ='relu',            # Activation function for hidden layers
                       batch_size: int =2,                   # Batch size
                       optimizer: str ='adam',                  # Optimizer
                       init_learning_rate: float =0.0001,       # Initial learning rate
                       decay_rate: float =0.5,                  # Rate at which the lr is decayed
                       decay_steps: int =int(2e6),              # Num. steps lr is decayed by decay_rate
                       policy_w: float =0.01,                   # Policy head weight
                       values_w: float =1,                      # Value  head weight
                       max_replay_buffer_size: int =int(1e6),   # Max replay buffer size 
                       max_grad_updates_per_exmpl: int =10,     # Max times an exmpl can be used in the replay buffer
                       q_recursive: float =0.1,                 # Prob. of adding a recursive query to the query queue
                       n_query_solvers: int =5,                 # Num. of procs solving querries off the query queue
                       input_query_queue: Queue =None,          # Input multiprocess queue
                       n_trainers: int =1,                      # Num. of procs performing batched training updates
                       use_shared_weights: bool =True,          # Use shared weights
                       weights_shm_lock: ReadRWLock =None,       # Read lock for the shared memory buffer
                       weights_shm_name: str =None):            # Name of the shared memory buffer
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
        #        A: Using a custom 'normalize_columns' output layer
        #
        strategy_output = tf.keras.layers.Dense(num_actions * 1326, activation='linear')(layer)
        strategy_output = tf.keras.layers.Reshape((num_actions, 1326))(strategy_output)
        strategy_output = tf.keras.layers.Softmax(axis=1)(strategy_output)
        #strategy_output = tf.keras.layers.Lambda(normalize_columns)(strategy_output)
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
        self.query_queue = input_query_queue if input_query_queue is not None else Queue()

        # We can either have 0 child processes or both query and trainer processes
        assert n_query_solvers >= 0 and n_trainers >= 0 and (n_query_solvers == 0) == (n_trainers == 0), \
            'Query solvers and trainers must both be non-negative and must both be zero or both be non-zero'
        
        self.n_trainers = n_trainers           # Num. of trainer processes
        self.n_query_solvers = n_query_solvers # Num. of solver processes

        #
        # If we are using shared weights, then load or create a shared memory buffer
        #
        # Option 1 - Use shared weights but no shared buffer is given
        #
        if use_shared_weights and weights_shm_name is None:
            #
            # Allocate a shared memory buffer to store the CVFN weights
            #
            # The trainer process writes the updated weights to the buffer after
            # each training update.
            #
            # The query solvers read the updated weights from the buffer and use
            # them to solve the queries.
            #
            # Count the size of the weights in bytes
            buffer_sz = 0
            for weight_tf in self.network.weights:
                weight_np = weight_tf.numpy()
                buffer_sz += weight_np.nbytes
            # Initialize the shared weights buffer
            self.shared_weights_buffer = SharedMemory(
                create=True,
                size=buffer_sz,
                name=weights_shm_name,
            )
            #
            # Create a read/write lock for the shared weights buffer
            #
            assert weights_shm_lock is None, "Lock should be input with the buffer name"
            self.shared_weights_lock = ReadWriteLock()
        #
        # Option 2 - Use shared weights and a shared buffer is given
        #
        elif use_shared_weights and weights_shm_name is not None:
            # Load the shared weights from the given shared memory buffer name
            self.shared_weights_buffer = SharedMemory(name=weights_shm_name)
            # Save the input read/write lock for the shared weights buffer
            assert weights_shm_lock is not None, "Lock should be input with the buffer name"
            self.shared_weights_lock = weights_shm_lock
        #
        # Option 3 - use local weights, do nothing.
        #
        else:
            self.shared_weights_buffer = None
            self.shared_weights_lock = None

        #
        # Initialize read and write locks for the shared weights buffer
        #
        if use_shared_weights:
            self.shm_read_lock = ReadRWLock(self.shared_weights_lock)
            self.shm_write_lock = WriteRWLock(self.shared_weights_lock)

        #
        # Initialize the shared weights buffer values
        #
        self.write_weights()

        #
        # Initialize the replay buffer queue
        #
        if self.n_trainers > 0:
            #
            # Multiprocessing queue for the query solver processes to put
            # solved queries onto, and trainer process to get targets from.
            #
            self.replay_buffer_queue = Queue()
        #
        # If we are not training, then we don't need replay buffer variables
        #
        else:
            self.replay_buffer_queue = None
        
        self.max_replay_buffer_size = max_replay_buffer_size
        self.max_grad_updates_per_exmpl = max_grad_updates_per_exmpl
        self.replay_buffer = None
        
        #
        # Spawn the child processes
        #
        if self.n_trainers > 0:
            #
            # Spawn the query solver processes
            #
            # These processes work in parallel.
            #
            # Query solver loop:
            #   1. Pull a query off the query queue
            #   2. Solve the query using GT-CFR
            #   3. Add the solved targets onto the replay buffer
            #   4. Repeat.
            #
            self.init_query_solvers()
            #
            # Spawn the trainer processes
            #
            # Trainer solver loop:
            #   1. Sample training targets from the replay buffer
            #   2. Perform a single training update
            #   3. Repeat.
            #
            self.init_trainers()

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
    def init_replay_buffer(self) -> None:
        self.replay_buffer = ReplayBuffer(max_size=self.max_replay_buffer_size,
                                          evict_after_n_samples=self.max_grad_updates_per_exmpl)

    #
    # Get the updated weight values from the shared memory buffer
    # and assign them to our local copy of the network.
    #
    def update_weights(self) -> None:
        # Skip if we are not using shared weights
        if not self.shared_weights_buffer:
            return
        # Aquire the shared weights read lock
        with self.shm_read_lock:
            # Byte offset into shared memory buffer
            offset = 0
            # For each weight tensor...
            for weight_tf in self.network.weights:
                # Convert the network weight tensor to a numpy array
                weight_np = weight_tf.numpy()
                # Get the size of the weight array in bytes
                weight_sz = weight_np.nbytes
                # Get the shape of the weight tensor
                weight_shape = weight_np.shape
                # Access the portion of the shared memory buffer for this tensor
                weight_shm = np.ndarray(
                    shape=weight_np.shape,
                    dtype=weight_np.dtype,
                    buffer=self.shared_weights_buffer.buf[offset:offset+weight_sz]
                )
                # Copy the shared memory weights into the weight tensor
                weight_tf.assign(weight_shm)
                # Update the offset for the next tensor
                offset += weight_sz
    
    #
    # Write the weight values to the shared memory buffer
    #
    def write_weights(self) -> None:
        # Skip if we are not using shared weights
        if not self.shared_weights_buffer:
            return
        # Aquire the shared weights write lock
        with self.shm_write_lock:
            # Byte offset into shared memory buffer
            offset = 0
            # For each weight tensor...
            for weight_tf in self.network.weights:
                # Convert the network weight tensor to a numpy array
                weight_np = weight_tf.numpy()
                # Get the size of the weight array in bytes
                weight_sz = weight_np.nbytes
                # Access the portion of the shared memory buffer for this tensor
                shared_array = np.ndarray(
                    shape=weight_np.shape,
                    dtype=weight_np.dtype,
                    buffer=self.shared_weights_buffer.buf[offset:offset+weight_sz]
                ) 
                # Copy the weight tensor into the shared memory buffer
                shared_array[:] = weight_np[:]
                # Update the offset for the next tensor
                offset += weight_sz
    
    #
    # Initialize the process pool for trainers.
    #
    def init_trainers(self) -> None:
        #
        # Skip if we are not using trainers
        #
        if self.n_trainers == 0:
            return
        #
        # Spawn the trainer processes
        #
        self.trainer_workers = [None] * self.n_trainers
        for trainer_id in range(self.n_trainers):
            self.trainer_workers[trainer_id] = Process(
                target=self.trainer_process,
                args=(trainer_id,
                      self.replay_buffer_queue,
                      self.shared_weights_lock,
                      self.shared_weights_buffer.name),
                daemon=True
            )
            self.trainer_workers[trainer_id].start()
    
    #
    # Trainer process main loop
    #
    # Inputs:
    #   trainer_id - id of the trainer
    #   parent_replay_buffer_queue - mp queue of the CFVN that spawned this process
    #   weights_shm_lock - rw lock for the shared weights buffer
    #   weights_shm_name - name of the shared weights buffer
    #
    @staticmethod
    def trainer_process(trainer_id: int,
                        parent_replay_buffer_queue: Queue,
                        weights_shm_lock: ReadWriteLock,
                        weights_shm_name: str):
        # Wrap process in a try-except block to debug errors
        try:
            # Runtime import
            from queue import Empty
            import time
            #
            # Log the start of the process
            #
            print(f'Started trainer process {trainer_id}')
            #
            # Initialize this process's CFVN object instance
            #
            # NOTE - For now, we use the default params for the CFVN.
            #        Eventually, we will have to pass the params through as 
            #        args to query_solver_process()
            #
            my_cvfn = CounterfactualValueNetwork(
                n_query_solvers=0, # Dont spawn query solver processes
                n_trainers=0, # Dont spawn trainer processes
                weights_shm_lock=weights_shm_lock, # Use the shared weights buffer
                weights_shm_name=weights_shm_name # Name of the shared weights buffer
            )
            my_cvfn.init_replay_buffer()
            #
            # Loop indefinately...
            #
            while True:
                #
                # Exhaust the replay buffer queue
                #
                while True:
                    try:
                        # Get training targets from the replay buffer queue
                        training_targets = parent_replay_buffer_queue.get(block=False)
                        # Add the training targets to the replay buffer
                        assert training_targets is not None, "Training targets should not be None"
                        my_cvfn.add_to_replay_buffer(training_targets)
                        print(f'Trainer {trainer_id} added a new target to the replay buffer (replay_buffer.size() = {my_cvfn.replay_buffer.size()})')
                        
                    except Empty:
                        # If queue is empty, check if we have enough targets
                        if my_cvfn.replay_buffer.size() >= my_cvfn.batch_size:
                            # We have enough targets, break to perform update
                            break
                        else:
                            # Wait for more targets
                            print(f'Trainer {trainer_id} waiting for more targets')
                            time.sleep(5*60)
                            continue
                #
                # Log
                #
                print(f'Trainer {trainer_id} has {my_cvfn.replay_buffer.size()} targets')
                #
                # Perform a batch update on the network
                #
                my_cvfn.train(trainer_id, niters=1) # 1 batch update
                #
                # Save the updated weights to the shared memory buffer
                #
                print(f'Trainer {trainer_id} saving weights')
                my_cvfn.write_weights()
        #
        # Spawn a remote pdb session
        #
        except Exception as e:
            print('==================================')
            print(f'Trainer {trainer_id} error:')
            print(e)
            print('==================================')
            import remote_pdb
            remote_pdb.set_trace()

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
        #       We initialize the query solver's cvfn objects to have zero
        #       query solvers in order to avoid recursive processes spawning.
        #       The query solver's add their recursive querries to the query
        #       queue that is shared across all processes.
        #
        if self.n_query_solvers == 0:
            return
        #
        # Spawn the worker processes
        #
        self.query_workers = [None] * self.n_query_solvers
        for query_solver_id in range(self.n_query_solvers):
            # Init process object
            self.query_workers[query_solver_id] = Process(
                # Query solver main function
                target=self.query_solver_process, 
                # Arguements
                args=(query_solver_id, 
                      self.query_queue,
                      self.replay_buffer_queue,
                      self.shared_weights_lock,
                      self.shared_weights_buffer.name),
                # Kill the process when the main process dies
                daemon=True
            )
            # Log
            print(f'Starting query solver {query_solver_id}')
            # Start the process
            self.query_workers[query_solver_id].start()

    #
    # Loop for the query solver tasks
    #
    # Inputs:
    #   worker_id - query solver process id of the process executing this function
    #   parent_query_queue - mp query queue of the CFVN that spawned this process
    #   parent_replay_buffer_queue - mp replay buffer queue of the CFVN that spawned this process 
    #   weights_shm_lock - rw lock for the shared weights buffer
    #   weights_shm_name - name of the shared weights buffer
    #
    @staticmethod
    def query_solver_process(query_solver_id: int, 
                             parent_query_queue: Queue,
                             parent_replay_buffer_queue: Queue,
                             weights_shm_lock: ReadWriteLock, 
                             weights_shm_name: str) -> None:
        # Runtime import
        from rlcard.agents.gt_cfr_agent.nodes import ChanceNode
        #
        # Log the start of the process
        #
        print(f'Started query solver process {query_solver_id}')
        #
        # Initialize this process's CFVN object instance
        #
        # NOTE - For now, we use the default params for the CFVN.
        #        Eventually, we will have to pass the params through as 
        #        args to query_solver_process()
        #
        my_cvfn = CounterfactualValueNetwork(
            input_query_queue=parent_query_queue, # Place recursive queries back onto the parent process's query queue
            n_query_solvers=0, # Dont spawn query solver processes
            n_trainers=0, # Dont spawn trainer processes
            weights_shm_lock=weights_shm_lock, # Use the shared weights buffer
            weights_shm_name=weights_shm_name # Name of the shared weights buffer
        )
        #
        # Initialize this process's GT-CFR solver instance
        #
        from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRSolver
        my_solver = GTCFRSolver(input_cfvn=my_cvfn, prob_query_solve=my_cvfn.q_recursive)
        #
        # Loop indefinately...
        #
        while True:
            #
            # Fail gracefully if a query solver process encounters
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
                print(f'Query Solver #{query_solver_id} -> Getting next query')
                query = parent_query_queue.get(block=True)
                print(f'Query Solver #{query_solver_id} -> Got query')
                #
                # Solve the query
                #
                print(f'Query Solver #{query_solver_id} -> Starting solve')
                target_strat, target_values = my_solver.solve(*query)
                print(f'Query Solver #{query_solver_id} -> Finished solve')
                #
                # Get the input CFVN vector for the given query
                #
                node = my_solver.decision_point
                is_chance_node = isinstance(node, ChanceNode)
                input_vect = my_cvfn.to_vect(node.game, node.player_ranges, is_chance_node)
                #
                # Reshape the target strategy and values to match the network output dimensions
                #
                triu_indices = np.triu_indices(52, k=1)
                target_strat = target_strat[:, triu_indices[0], triu_indices[1]]
                target_values = target_values[:, triu_indices[0], triu_indices[1]]
                #
                # Pad the target strategy with zeros for invalid actions
                #
                target_strat_padded = np.zeros((my_cvfn.num_actions, 1326))
                all_actions = node.game.round.get_all_actions()
                valid_action_idxs = [all_actions.index(action) for action in node.actions]
                for idx, padded_idx in enumerate(valid_action_idxs):
                    target_strat_padded[padded_idx] = target_strat[idx]
                #
                # Place the solved query onto the parent process's replay buffer queue
                #
                parent_replay_buffer_queue.put((input_vect, target_strat_padded, target_values))
                print(f'Query Solver #{query_solver_id} -> Placed the solved query onto the replay buffer queue')
                #
                # Reset the solver game tree after solving
                #
                my_solver.reset()
            #
            # Output the worker error to the console
            #
            except Exception as e:
                #
                # Spawn a remote pdb session
                #
                import remote_pdb
                import traceback
                print('==================================')
                print(f'Query Solver #{query_solver_id} error:')
                print()
                print(e)
                print()
                print(traceback.format_exc())
                print()
                print('==================================')
                remote_pdb.set_trace()

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
    # valid_actions - list of valid actions for the acting player
    #
    # Note: the input should come from the to_vect() function.
    #
    def query(self, input : np.ndarry, valid_action_idxs : list[Action]) -> tuple[np.ndarray]:
        #
        # Verify that the input vector matches the dimensions of the network
        #
        assert input.shape == (self.input_dim,), "Unexpected input dimension"
        #
        # Run inference
        #
        tf_strategy, tf_values =  self.network(np.expand_dims(input, axis=0))
        #
        # Format the tensorflow output back into triu numpy arrays
        #
        np_strategy = np.zeros((self.num_actions, 52, 52))
        idxs = np.triu_indices(52, k=1)
        # Ensure tensor output is converted to NumPy and squeezed to remove batch dimension
        tf_strategy = tf_strategy.numpy().squeeze(axis=0)  # Shape: (num_actions, 1326)
        # Map each action's strategy into the upper triangular array
        for action in range(self.num_actions):
            np_strategy[action][idxs] = tf_strategy[action]  # Fill upper-triangle entries
        # Repeat for the values array
        np_values = np.zeros((self.num_players, 52, 52))
        tf_values = tf_values.numpy().squeeze(axis=0) # Shape: (num_players, 1326)
        for player in range(self.num_players):
            np_values[player][idxs] = tf_values[player]
        #
        # Set strategy and values for invalid hands to zero.
        # A hand is considered invalid if it contains a public card.
        #
        public_card_idxs = np.where(input[:52] == 1)[0]
        np_values[:, public_card_idxs, :] = 0.
        np_values[:, :, public_card_idxs] = 0.
        np_strategy[:, public_card_idxs, :] = 0.
        np_strategy[:, :, public_card_idxs] = 0.
        #
        # Shrink the strategy matrix to only include valid actions
        #
        np_strategy = np_strategy[valid_action_idxs]
        #
        # Renormalize the strategy matrix
        #
        strat_sums = np.sum(np_strategy, axis=0, keepdims=True)
        strat_sums[strat_sums == 0] = 1 # Don't divide by zero
        np_strategy /= strat_sums
        #
        # Return the numpy triu matrices as a tuple
        #
        return np_strategy, np_values

    #
    # Add a query to the query queue for solving
    #
    def add_to_query_queue(self, query: tuple[np.ndarray]) -> None:
        assert len(query) == 4 # [game object, opp. values, player range]
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
        # Add the solved query to the replay buffer
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
    def train(self, trainer_id, niters: int =np.inf):
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
                    print(f'{trainer_id} --> Batch loss {update_loss}')
            #
            # Return if we don't have enough targets
            #
            else:
                return
            #
            # Increment batch counter
            #
            update_iter += 1
