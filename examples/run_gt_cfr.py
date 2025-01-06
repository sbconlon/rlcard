# ====================== #
#  Train a GT-CFR Agent  #
# ====================== #

# External imports
import ipdb

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent

# NOTE - step through the start of the training squence line-by-line
ipdb.set_trace()

#
# Training parameters
#
num_episodes = 1

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

          'starting_stage': 'river'
}
env = rlcard.make('no-limit-holdem', config=config)

#
# Initialize the GT-CFR Agent
#
agent = GTCFRAgent(env)

#
# Run self-play training episodes
#
for episode in range(num_episodes):
    agent.self_play()