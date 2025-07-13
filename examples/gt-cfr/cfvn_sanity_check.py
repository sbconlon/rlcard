# External imports
import numpy as np

# Internal imports
from print_profiler_stats import print_rlcard_function_stats
import rlcard
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.utils import uniform_range
from rlcard.games.base import Card
from rlcard.utils.utils import init_standard_deck
from rlcard.games.nolimitholdem.round import Action

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
            'dealer_id': 0
}
env = rlcard.make('no-limit-holdem', config=config)
env.reset()
game = env.game


#
# Load truth data
#
true_strat = np.load('strat-fs-7s8s8hThJh.npy')
true_values = np.load('values-fs-7s8s8hThJh.npy')

#
# Initialize the CFVN
#
cfvn = CounterfactualValueNetwork(n_trainers=0, n_query_solvers=0)
cfvn.init_replay_buffer()

#
# Input vector
#
player_ranges = np.array([uniform_range(game.public_cards), uniform_range(game.public_cards)])
print([card.to_int() for card in game.public_cards])
vect = cfvn.to_vect(game, player_ranges, False)

#
# Build the target
#
# Reshape the true strategy and values to match the network output dimensions
triu_indices = np.triu_indices(52, k=1)
target_strat = true_strat[:, triu_indices[0], triu_indices[1]]
target_values = true_values[:, triu_indices[0], triu_indices[1]]
# Pad the target strategy with zeros for invalid actions
target_strat_padded = np.zeros((cfvn.num_actions, 1326))
all_actions = game.round.get_all_actions()
legal_actions = game.get_legal_actions()
valid_action_idxs = [all_actions.index(action) for action in legal_actions]
for idx, padded_idx in enumerate(valid_action_idxs):
    target_strat_padded[padded_idx] = target_strat[idx]
target = (vect, target_strat_padded, target_values)

#
# Int to card map
#
int_to_card = {c.to_int(): c for c in init_standard_deck()}

#
# Train directly on the truth values
#
n_updates = 0
while True:
    #
    # Test the network quality
    #
    print()
    print('=====================================================================')
    print(f'Batch #{n_updates}')
    print('Starting test solve')
    pred_strat, pred_values = cfvn.query(vect, valid_action_idxs)
    mse_strat, mse_values = np.square(pred_strat - true_strat), np.square(pred_values - true_values)
    print(f'Strategy MSE = {mse_strat.mean()}')
    print(f'Values MSE   = {mse_values.mean()}')
    print()
    worst_strat_hand = np.unravel_index(np.argmax(np.sum(mse_strat, axis=0)), mse_strat.shape[1:])
    worst_values_hand = np.unravel_index(np.argmax(np.sum(mse_values, axis=0)), mse_values.shape[1:])
    print(f'Worst Strategy')
    print(f'{[str(int_to_card[worst_strat_hand[0]]), str(int_to_card[worst_strat_hand[1]])]}')
    print(f'Expected = {true_strat[:, worst_strat_hand[0], worst_strat_hand[1]]}')
    print(f'Got = {pred_strat[:, worst_strat_hand[0], worst_strat_hand[1]]}')
    print()
    print(f'Worst Values')
    print(f'{[str(int_to_card[worst_values_hand[0]]), str(int_to_card[worst_values_hand[1]])]}')
    print(f'Expected = {true_values[:, worst_values_hand[0], worst_values_hand[1]]}')
    print(f'Got = {pred_values[:, worst_values_hand[0], worst_values_hand[1]]}')
    print()
    if np.all(np.isclose(pred_strat, true_strat)) and np.all(np.isclose(pred_values, true_values)):
        print('SUCCESS')
        break
    #
    # Perform a batch update
    #
    for _ in range(32):
        cfvn.add_to_replay_buffer(target)
    cfvn.train(0, niters=1)
    n_updates += 1

