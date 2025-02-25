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
                                     Action.RAISE_5X}
    }
    env = rlcard.make('no-limit-holdem', config=config)

    #
    # Initialize the GT-CFR Agent
    #
    agent = GTCFRAgent(env)

    #
    # Load the target strategy and values matrices
    #
    target_strats = np.load('strat-fs-7s8s8hThJh.npy')
    target_values = np.load('values-fs-7s8s8hThJh.npy')

    #
    # Define the error as the average absolute distance between the
    # matrix values
    #
    error = lambda X, Y: np.average(np.abs(X - Y))

    #
    # For a set number of iterations, run solve on the starting
    # game state and compare the results to the full solve values.
    #
    N_SOLVES = 10000
    for i in range(N_SOLVES):
        # Reset the game and the solver
        env.reset()
        agent.reset()
        # Solve using GT-CFR
        strats, values = agent.solver.solve(env.game)
        # Compare the results to the targets
        print()
        print('=============================================')
        print(f'--> Solve #{i}')
        print(f'Strats error: {error(strats, target_strats)}')
        print(f'Values error: {error(values, target_values)}')
        print('=============================================')
        print()

if __name__ == "__main__":
    main()