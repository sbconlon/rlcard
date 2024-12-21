# External imports
import numpy as np

# Internal imports
from rlcard.games.base import Card

#
# Probability of the player holding a hand in the public state
# 
# This is represented as a 52x52 upper triangular matrix with the 
# diagonal entries set to zero and the entire matrix is normalized.
#
# public_cards = list of indicies corresponding to publicly observed cards
#
def random_range(public_cards : list[Card]) -> np.array:
    #
    # Convert Card objects to array indexes
    #
    idxs = [card.to_int() for card in public_cards]
    #
    # Initialize an upper triangular matrix of random values 
    # with zeros on the diagonal
    #
    rand_range = np.triu(np.random.rand(52, 52), k=1)
    #
    # Set the probability of holding public cards to zero
    #
    rand_range[idxs, :] = 0.
    rand_range[:, idxs] = 0.
    #
    # Normalize
    #
    rand_range /= rand_range.sum()

    return rand_range

#
# Get the precomputed hand values for the start of the game
#
def initial_hand_values() -> np.array:
    #
    # NOTE - Pull from a csv file here
    #
    pass
