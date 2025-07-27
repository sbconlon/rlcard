# External imports
import copy
from itertools import combinations, permutations
import numpy as np
from pathlib import Path
import tensorflow as tf

# Internal imports
from rlcard.games.base import Card
from rlcard.games.nolimitholdem import Dealer
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck

#
# | ------------------------------------- |
# |                                       |
# | Helper functions for handling indexes |
# |                                       |
# | ------------------------------------- | 
#

#
# Converts 2d coors (i, j) for an upper-triangular matrix
# with the diagonal set to zero, to the 1d coor of the flattened matrix
#
def get_1d_coor(i: int, j: int) -> int:
    assert i < j, f"Invalid 2d coors: ({i}, {j})"
    return ((2*52 - i - 1)*i)//2 + (j - i - 1)

#
# Returns the 1d indices for all elements in the column c
#
def get_col_coors(c: int) -> list[int]:
    assert 0 <= c < 52, f"Invalid column: {c}"
    return [get_1d_coor(i, c) for i in range(c)]

#
# Returns the 1d indicies for all elements in the row r
#
def get_row_coors(r: int) -> list[int]:
    assert 0 <= r < 52, f"Invalid column: {r}"
    return [get_1d_coor(r, j) for j in range(r+1, 52)]

#
# Returns all 1d indices for all elements associated with the card
#
def get_card_coors(card: int) -> list[int]:
    assert 0 <= card < 52, f"Invalid card id: {card}"
    return get_col_coors(card) + get_row_coors(card)



#
# | ---------------------------------------------- |
# |                                                |
# | Functions for generating ranges and strategies |
# |                                                |
# | ---------------------------------------------- | 
#

#
# Compute a uniform range over all possible hands,
# given the publicly observable cards.
#
# The range is a flattened upper-triangular matrix with the diagonal
# set to zero, where range[player id, r,c] is the range value for the hand 
# with cards r and c.
# 
# The function get_1d_coor maps the 2d coordinates (r, c) of the
# upper-triangular matrix to the 1d index i of the flattened vector.  
#
# range[i] = prob. the player is holding hand i.
#
# public_cards = list of indicies corresponding to publicly observed cards
#
def uniform_range(public_cards : list[Card]) -> np.ndarray:
    # Initialize all hand combos to 1
    player_range = np.ones(1326, dtype=np.float32)
    # Set the probability of holding a public card to zero
    for card in public_cards:
        player_range[get_card_coors(card.to_int())] = 0.
    # Normalize the distribution
    player_range /= player_range.sum()
    return player_range

#
# Compute a random range over all possible hands,
# given the publicly observable cards.
#
def random_range(public_cards: list[Card]) -> np.ndarray:
    # Initialize the range to random values
    player_range = np.random.rand(1326)
    # Set the probability of holding a public card to zero
    for card in public_cards:
        player_range[get_card_coors(card.to_int())] = 0.
    # Normalize the distribution
    player_range /= player_range.sum()
    return np.array(player_range, dtype=np.float32)

#
# Return a valid random strategy matrix
#
def random_strategy(n_actions: int, public_cards: list[Card]) -> np.ndarray:
    # Generate random values
    strat = np.random.rand(n_actions, 1326)
    # Normalize
    sums = strat.sum(axis=0, keepdims=0)
    strat /= sums
    # Zero hands containing public cards
    for card in public_cards:
        strat[:, get_card_coors(card.to_int())] = 0
    return strat



#
# | ------------------------------------------------------- |
# |                                                         |
# | Helper functions for estimating starting hand strengths |
# |                                                         |
# | ------------------------------------------------------- | 
#

#
# Encode the public cards into a filename
#
def public_cards_to_filename(public_cards: list[Card]) -> str:
    # Give the preflop setting its own filename
    if public_cards == []:
        return 'preflop.npy'
    # Else, make the filename by concatenating the public cards
    filename = ''
    public_cards.sort(key=lambda x: x.to_int())
    for card in public_cards:
        filename += str(card)
    filename += '.npy'
    return filename

#
# Get the precomputed hand values for the start of the game
#
def starting_hand_values(game: NolimitholdemGame) -> np.ndarray:
    #
    # Check the directory './cached_starting_hand_values'
    # to see if the hand values for the given public cards
    # has been computed and stored already.
    #
    directory = './cached_starting_hand_values/'
    filename = public_cards_to_filename(game.public_cards)
    path = Path(directory + filename)
    #
    # If the values are cached, then return them
    #
    if path.exists():
        return np.load(directory+filename)
    #
    # Else, compute the starting hand values
    #
    values = compute_starting_hand_values(game)
    #
    # Save the computed values before returning them
    #
    np.save(directory+filename, values)
    return values

#
# Estimate the starting hand values by simulating the game
# N times, collecting the average return from each hand.
#
def compute_starting_hand_values(game: NolimitholdemGame, N: int = 10000) -> np.ndarray:
    print('Starting computing_hand_values()')
    import time; start = time.time()
    #
    # Accumulate the total value collected by each hand,
    # initialized to zero. And the number of times
    # the hand was sampled.
    #
    acc_values = np.zeros(1326)
    acc_visits = np.zeros(1326)
    #
    # Run N simulations
    #
    for _ in range(N):
        #
        # Create a copy of the given game to simulate
        #
        sim_game = copy.deepcopy(game)
        #
        # Deal two hypotetical hands to each player
        #
        sim_game.np_random = np.random.RandomState()
        sim_game.dealer = Dealer(sim_game.np_random)
        for card in sim_game.public_cards:
            sim_game.dealer.remove_card(card)
        for player in sim_game.players:
            player.hand = [sim_game.dealer.deal_card(), sim_game.dealer.deal_card()]
        #
        # Simulate the game to an endpoint
        #
        while not sim_game.is_over():
            #
            # Advance the game forward by always taking the CHECK/CALL action
            #
            # You can always check or call in any game state
            #
            legal_actions = sim_game.get_legal_actions()
            if Action.CHECK in legal_actions:
                sim_game.step(Action.CHECK)
            else:
                sim_game.step(Action.CALL)
        #
        # Compute the payoffs for this hand configuration 
        # in this node's game state
        #
        hand_payoffs = sim_game.get_payoffs()
        #
        # Update the accumulated values and visits
        #
        for pid, player in enumerate(sim_game.players):
            card1, card2 = sorted((player.hand[0].to_int(), player.hand[1].to_int()))
            hand = get_1d_coor(card1, card2)
            acc_values[hand] +=  hand_payoffs[pid] / sim_game.dealer.pot
            acc_visits[hand] += 1
    print('Finished compute_starting_hand_values')
    print(f'{time.time()-start} seconds')
    #
    # Divide by the number of simulations to get the average
    # payoff for the hands in the simulated games.
    #
    return np.divide(acc_values, acc_visits, 
                     out=np.zeros_like(acc_values), where=acc_visits != 0)



#
# | --------------------- |
# |                       |
# | CFVN helper functions |
# |                       |
# | --------------------- | 
#

#
# Tensorflow network layer that normalizes the columns of the input matrix
#
def normalize_columns(x, epsilon=1e-10):
    #import ipdb; ipdb.set_trace()
    col_sums = tf.reduce_sum(x, axis=0, keepdims=True)
    col_sums = tf.where(col_sums == 0, epsilon, col_sums)  # Avoid divide-by-zero
    return tf.divide(x, col_sums)
