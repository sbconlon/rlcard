# External imports
import copy
from itertools import combinations, permutations
import numpy as np
from pathlib import Path

# Internal imports
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.game import NolimitholdemGame
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck

#
# Uniform of the player holding a hand in the public state
# 
# This is represented as a 52x52 upper triangular matrix with the 
# diagonal entries set to zero and the entire matrix is normalized.
#
# public_cards = list of indicies corresponding to publicly observed cards
#
def uniform_range(public_cards : list[Card]) -> np.array:
    # Initialize all hand combos to 1
    player_range = np.triu(np.ones((52, 52)), k=1)
    # Set the probability of holding a public card to zero
    for card in public_cards:
        player_range[card.to_int():] = 0.
        player_range[:card.to_int()] = 0.
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
def compute_starting_hand_values(game: NolimitholdemGame, N: int = 1000) -> np.ndarray:
    #
    # Accumulate the total value collected by each hand,
    # initialized to zero.
    #
    values_sum = np.zeros((52, 52))
    update_count = 0
    #
    # Run N simulations
    #
    for i in range(N):
        print(i)
        #
        # Simulate the game to an endpoint
        #
        sim_game = copy.deepcopy(game)
        while not sim_game.is_over():
            #
            # Advance the game forward by always taking the CHECK/CALL action
            #
            sim_game.step(Action.CHECK_CALL)
        #
        # NOTE - This code is largely taken from the ChanceNode.cache_payoffs() function.
        #        We should probably seperate this out into a stand alone function
        #        to maximize code reuse.
        #
        # Get the set of possible cards 
        # the players can have in their hands
        #
        possible_cards = [card for card in init_standard_deck() if card not in sim_game.public_cards]
        #
        # For each possible hand combination...
        #
        # Note -
        #     combinations(possible_cards, 2) 
        #         = list of all possible 2 card hands
        #
        #     permulations(..., num_player) 
        #         = list of all possible hand assignments to each player
        #
        for hands in permutations(combinations(possible_cards, 2), sim_game.num_players):
            #
            # Filter hand combinations that have overlapping cards
            #
            if not set(hands[0]).isdisjoint(set(hands[1])):
                continue
            print([(str(hand[0]), str(hand[1])) for hand in hands])
            #
            # Update the normalization factor
            #
            update_count += 1
            #
            # Assign the hypothetical hands to each player in the game instance
            #
            for pid, hand in enumerate(hands):
                sim_game.players[pid].hand = list(hand)
            #
            # Compute the payoffs for this hand configuration 
            # in this node's game state
            #
            hand_payoffs = sim_game.get_payoffs()
            #
            # Update the accumulated values function.
            #
            # Weight this payout by the probability of the opponent 
            # having their hand.
            #
            # Note: We use Player 1's payoffs here, but it doesn't matter
            #       which player's payoffs we use.
            #
            values_sum[hands[0][0].to_int(), hands[0][1].to_int()] +=  hand_payoffs[0]
    #
    # Divide by the number of simulations to get the average
    # payoff for the hands in the simulated games.
    #
    import ipdb; ipdb.set_trace()
    return values_sum / update_count


