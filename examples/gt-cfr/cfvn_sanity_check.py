# ================================================================= #
#  CFVN sanity check: can the network memorize one training target?  #
# ================================================================= #
#
# Builds the ground-truth target for the river root (uniform ranges) and trains the
# network on 32 copies of it until its prediction matches. Checks the model, the loss,
# and the featurization, independently of any search.

# External imports
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # leave GPU memory for cuBLAS; must precede the TensorFlow import
import numpy as np
import tensorflow as tf

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.utils import uniform_range, get_2d_coors
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck

#
# The river ground-truth spot
#
config = {
    'fixed_public_cards': [Card('H', 'T'), Card('S', '8'), Card('S', '7'), Card('H', 'J'), Card('H', '8')],
    'starting_stage': 'river',
    'chips_for_each': 10,
    'disabled_actions': {Action.BET_THIRD_POT, Action.BET_POT, Action.BET_5POT, Action.RAISE_3X, Action.RAISE_5X},
    'dealer_id': 0,
}
env = rlcard.make('no-limit-holdem', config=config)
env.reset()
game = env.game

true_strat = np.load('strat-fs-7s8s8hThJh-tf.npy')      # (3, 1326): rows in legal-action order
true_values = np.load('values-fs-7s8s8hThJh-tf.npy')    # (2, 1326)

cfvn = CounterfactualValueNetwork(n_trainers=0, n_query_solvers=0, use_shared_weights=False)
cfvn.init_replay_buffer()

#
# One target: uniform ranges at the river root, the full-solve answer as the label
#
player_ranges = np.stack([uniform_range(game.public_cards), uniform_range(game.public_cards)])
target = cfvn.make_training_target(game, player_ranges, true_strat, true_values)
vect, target_strat_padded, target_values = target
legal_rows = [game.get_all_actions().index(a) for a in game.get_legal_actions()]

deck = sorted(init_standard_deck(), key=lambda c: c.to_int())
def hand_str(hand):
    c1, c2 = get_2d_coors(hand)
    return f'{deck[c1]}{deck[c2]}'

n_updates = 0
while True:
    pred_strat, pred_values = cfvn.query(tf.constant(vect[None, :]))
    pred_strat = pred_strat.numpy()[0][legal_rows]     # (3, 1326), legal rows only
    pred_values = pred_values.numpy()[0]
    err_strat = np.abs(pred_strat - true_strat)
    err_values = np.abs(pred_values - true_values)
    print(f'Batch #{n_updates:4d}   strategy MAE {err_strat.mean():.5f}   values MAE {err_values.mean():.5f}')
    if n_updates % 10 == 0:
        worst = int(np.argmax(err_strat.sum(axis=0)))
        print(f'   worst strategy hand {hand_str(worst)}: expected {np.round(true_strat[:, worst], 3)} got {np.round(pred_strat[:, worst], 3)}')
        worst = int(np.argmax(err_values.sum(axis=0)))
        print(f'   worst value hand    {hand_str(worst)}: expected {np.round(true_values[:, worst], 3)} got {np.round(pred_values[:, worst], 3)}')
    if np.allclose(pred_strat, true_strat, atol=1e-2) and np.allclose(pred_values, true_values, atol=1e-2):
        print('SUCCESS')
        break
    for _ in range(cfvn.batch_size):
        cfvn.add_to_replay_buffer(target)
    cfvn.train(0, niters=1)
    n_updates += 1
