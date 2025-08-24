# ====================== #
#  Train a GT-CFR Agent  #
# ====================== #

# External imports
from collections import defaultdict
import numpy as np

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent
from rlcard.agents.gt_cfr_agent.utils import get_2d_coors, get_1d_coor, get_hand_class
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck

def main():
    #
    # Training parameters
    #
    num_episodes = 25

    #
    # Initialize nolimit holdem environment
    #
    config = {
                'fixed_public_cards': [
                                        Card('H', 'T'), 
                                        Card('S', '8'), 
                                        Card('S', '7'), 
                                        Card('H', 'J'), 
                                        Card('H', '8')
                                    ],

                'starting_stage': 'river',
                'chips_for_each': 10,
                'disabled_actions': {#Action.BET_HALF_POT,
                                     Action.BET_POT,
                                     Action.BET_5POT,
                                     #Action.RAISE_2X,
                                     Action.RAISE_3X,
                                     Action.RAISE_5X},
                'dealer_id': 0 # Fix Player 0 as the dealer
    }
    env: NolimitholdemEnv
    env = rlcard.make('no-limit-holdem', config=config)

    #
    # Initialize the GT-CFR Agent
    #
    env.reset()
    agent = GTCFRAgent(env, full_solve=True)
    agent.solver.n_expansions_per_regret_updates = 1/10000
    """
    #
    # Solve
    #
    agent.solver.solve(env.game)
    #
    # Save the results
    #
    strat = agent.solver.tree.get_cum_strategy(0)
    value = agent.solver.tree.get_values(0)
    np.save('strat-fs-7s8s8hThJh-tf.npy', strat)
    np.save('values-fs-7s8s8hThJh-tf.npy', value)
    print('--> Saved files')
    """
    strat = np.load('strat-fs-7s8s8hThJh-tf.npy')
    value = np.load('values-fs-7s8s8hThJh-tf.npy')
    #
    # Print the results
    #
    deck = init_standard_deck()
    actions = env.game.get_legal_actions()
    player = env.game.game_pointer
    # Group the hands into hand classes
    hand_classes = defaultdict(list)
    class_names = {}
    for hand in range(1326):
        clss, class_name = get_hand_class(hand, env.game.public_cards)
        hand_classes[clss].append(hand)
        class_names[clss] = class_name
    # Sort the hands within each class
    for clss in hand_classes.keys():
        hand_classes[clss] = sorted(hand_classes[clss], key=lambda hand: value[player, hand], reverse=True)
    # Sort each hand into a bucket
    def bucket(hand: int) -> str:
        buckets = ['CHECK', 'ALL-IN', 'BET']
        return buckets[np.argmax(strat[:, hand])]
    # Sort the classes
    ranked_classes = sorted(hand_classes.keys())
    for clss in ranked_classes:
        if clss == -1: # skip invalid hands
            continue
        print(f'---> {class_names[clss]}')
        for hand in hand_classes[clss]:
            card1, card2 = get_2d_coors(hand)
            val = value[player, hand]
            print(f"{' ' if val >= 0 else ''} {val:.3f} | "
                f"{deck[card1]} {deck[card2]} -  "
                f"C {strat[0, hand]:.3f}   "
                f"A {strat[1, hand]:.3f}   "
                f"B {strat[2, hand]:.3f}   "
                f'{bucket(hand)}')
    
    import ipdb; ipdb.set_trace()

if __name__ == '__main__': # Needed for multiprocessing
    main()