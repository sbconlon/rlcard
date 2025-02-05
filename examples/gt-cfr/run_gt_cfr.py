# ====================== #
#  Train a GT-CFR Agent  #
# ====================== #

# Debug + Testing imports
import cProfile
import ipdb

from print_profiler_stats import print_rlcard_function_stats

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action

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
            'disabled_actions': {Action.BET_HALF_POT,
                                 Action.BET_POT,
                                 Action.BET_2POT,
                                 Action.BET_5POT,
                                 Action.RAISE_2X,
                                 Action.RAISE_3X,
                                 Action.RAISE_5X}
}
env = rlcard.make('no-limit-holdem', config=config)

#
# Initialize the GT-CFR Agent
#
agent = GTCFRAgent(env)

# Profile self-play
profiler = cProfile.Profile()
profiler.enable()

#
# Run self-play training episodes
#
try:

    for episode in range(num_episodes):
        print('=====================================================')
        print()
        print(f'--> Episode {episode + 1}')
        print()
        agent.self_play()

except Exception as e:
    print(e)
    import ipdb; ipdb.post_mortem()

# Display profiler data
profiler.disable()
print_rlcard_function_stats(profiler)

import ipdb; ipdb.set_trace()
