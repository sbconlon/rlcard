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
import numpy as np
from queue import Queue
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

# Internal imports
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
    # Note 1: For now, the network is hard coded to use a MLP network
    #         because this is what was used in the literature.
    #
    # Note 2: Network parameters default to the values used in the literature.
    #
    def __init__(self, num_players: int =2,                     # Tot. num. of players in the game
                       num_actions: int =5,                     # Tot. num. of actions in the game
                       num_neurons_per_layer: int =2048,        # Neurons per layer
                       num_layers: int =6,                      # Num. of hidden layers
                       activation_func: str ='relu',            # Activation function for hidden layers
                       max_replay_buffer_size: int = int(1e6)): # Max replay buffer size 
        #
        # Compute the input dimension
        # according to the given number of players in the game
        #
        self.num_players = num_players
        self.num_actions = num_actions
        self.input_dim = (2*self.num_players + 53) + (1326*self.num_players) # See to_vect()
        
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
        # NOTE - 'linear' is a place holder, what activation do we actually want?
        #
        strategy_output = Dense(num_actions * 1326, activation='linear')(layer)
        strategy_output = tf.keras.layers.Reshape((num_actions, 1326))(strategy_output)
        #
        # Values output, (num_players, 1326)
        #
        # NOTE - 'linear' is a place holder, what activation do we actually want?
        #
        values_output = Dense(num_players * 1326, activation='linear')(layer)
        values_output = tf.keras.layers.Reshape((num_players, 1326))(values_output)

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
        self.query_queue = Queue()

        #
        # This is a buffer of solved querries.
        #
        # Note: this is the CVFN training set.
        #
        # Solved querries are add to the replay buffer in two ways:
        #
        #    1. By CVFN worker processes that fully solve queries off the query queue
        #
        #    2. By the GT-CFR agent after self-play game states have been solved
        #
        # Note: this is a FIFO buffer of fixed size
        #
        self.max_replay_buffer_size = max_replay_buffer_size
        self.replay_buffer = Queue()

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
        data_type = np.float64
        #
        # --> Featurize game object
        #
        # N-hot encoding of public cards
        #
        public_card_encoding = np.zeros(52, dtype=data_type)
        public_card_encoding[[card.to_int() for card in game.public_cards]] = 1
        #
        # Player pot commitments (normalized by stack size at the start of the hand)
        #
        pot_commitments = np.array([player.in_chips / (player.remained_chips + player.in_chips)
                                        for player in game.players], dtype=data_type)
        #
        # 1-hot acting player encoding
        #
        acting_player = np.zeros(game.num_players + 1, dtype=data_type)
        acting_player[-1 if chance_actor else game.game_pointer] = 1
        #
        # --> Featurize player ranges
        #
        # The input range for each player is a 52x52 upper triangular matrix
        # with zeros across the diagonal.
        #
        # Each of these matrices needs to be flattened into 1326 vector.
        #
        # Note: These matrices have at most 1326 non-zero entries, matching our
        #       expected vector size.
        #
        #       Upper triangular matrix elements = 52*(52+1)/2 = 1378 non-zero elems
        #       
        #       Exluding diagonal entries = 1378 - 52 = 1326 non-zero elems = num. hand combos 
        #
        assert self.num_players == game.num_players, "Game state and network num players mismatch"
        assert ranges.shape == (self.num_players, 52, 52), "Unrecognized player range shape"
        range_vect = np.zeros(1326*game.num_players, dtype=data_type)
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
    def add_to_replay_buffer(self, solved_query) -> None:
        if self.replay_buffer.qsize() > self.max_replay_buffer_size:
            self.replay_buffer.