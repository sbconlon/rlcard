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
# Compute a uniform range over all possible hands,
# given the publicly observable cards.
#
# range[i, j] = prob. the player is holding cards i and j.
# 
# This is represented as a 52x52 upper triangular matrix with the 
# diagonal entries set to zero and the entire matrix is normalized.
#
# public_cards = list of indicies corresponding to publicly observed cards
#
def uniform_range(public_cards : list[Card]) -> np.ndarray:
    # Initialize all hand combos to 1
    player_range = np.triu(np.ones((52, 52)), k=1)
    # Set the probability of holding a public card to zero
    for card in public_cards:
        player_range[card.to_int(), :] = 0.
        player_range[:, card.to_int()] = 0.
    # Normalize the distribution
    player_range /= player_range.sum()
    return player_range

#
# Compute a random range over all possible hands,
# given the publicly observable cards.
#
def random_range(public_cards: list[Card]) -> np.ndarray:
    # Initialize the range to random values
    player_range = np.triu(np.random.rand(52, 52), k=1)
    # Set the probability of holding a public card to zero
    for card in public_cards:
        player_range[card.to_int(), :] = 0.
        player_range[:, card.to_int()] = 0.
    # Normalize the distribution
    player_range /= player_range.sum()
    return player_range

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
    #
    # Accumulate the total value collected by each hand,
    # initialized to zero. And the number of times
    # the hand was sampled.
    #
    acc_values = np.zeros((52, 52))
    acc_visits = np.zeros((52, 52))
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
            sim_game.step(Action.CHECK_CALL)
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
            acc_values[card1, card2] +=  hand_payoffs[pid] / sim_game.dealer.pot
            acc_visits[card1, card2] += 1
    #
    # Divide by the number of simulations to get the average
    # payoff for the hands in the simulated games.
    #
    return np.where(acc_visits != 0, acc_values / acc_visits, 0)

#
# Tensorflow network layer that normalizes the columns of the input matrix
#
def normalize_columns(x, epsilon=1e-10):
    import ipdb; ipdb.set_trace()
    col_sums = tf.reduce_sum(x, axis=0, keepdims=True)
    col_sums = tf.where(col_sums == 0, epsilon, col_sums)  # Avoid divide-by-zero
    return tf.divide(x, col_sums)
