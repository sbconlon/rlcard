# ====================== #
#  Train a GT-CFR Agent  #
# ====================== #

# External imports
import numpy as np

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action

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
                'disabled_actions': {Action.BET_POT,
                                     Action.BET_5POT,
                                     Action.RAISE_3X,
                                     Action.RAISE_5X},
                'dealer_id': 0 # Fix Player 0 as the dealer
    }
    env = rlcard.make('no-limit-holdem', config=config)

    #
    # Initialize the GT-CFR Agent
    #
    env.reset()
    agent = GTCFRAgent(env, full_solve=True)
    agent.solver.n_expansions_per_regret_updates = 1/10000
    agent.solver.solve(env.game)
    np.save('strat-fs-7s8s8hThJh.npy', agent.solver.root.strategy)
    np.save('values-fs-7s8s8hThJh.npy', agent.solver.root.values)
    print('--> Saved files')



if __name__ == '__main__': # Needed for multiprocessing
    main()