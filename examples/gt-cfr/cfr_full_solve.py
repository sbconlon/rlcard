# ================================================== #
#  Full-tree solve of a river ground-truth spot       #
# ================================================== #
#
# Two reference spots on the same board (Th 8s 7s Jh 8h), dealer 0, river start:
#
#   5bb    chips_for_each=10 (a 5 bb game: the big blind is 2 chips), blinds-only pot,
#          actions {half pot, 2x pot, raise 2x} + fold/check/call/all-in, no raise cap.
#          33 public nodes. The original golden master; kept as a solver regression spot.
#
#   100bb  chips_for_each=200, starting_pot=40 (20 bb), the product's river set
#          {1/3 pot, pot, 2x pot, raise 2x} + fold/check/call/all-in, two raises per street.
#          105 public nodes. The M0 reference (plan: Repos/rlcard/plan/m0-river-network).
#
# Default: load the saved ground truth for --spot and print the per-hand strategy and values.
# --solve: re-solve the spot (default 10000 CFR iterations) and overwrite its files.

# External imports
import argparse
from collections import defaultdict
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # leave GPU memory for cuBLAS; must precede the TensorFlow import
import numpy as np

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent
from rlcard.agents.gt_cfr_agent.utils import get_2d_coors, get_hand_class
from rlcard.envs.nolimitholdem import NolimitholdemEnv
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action as A
from rlcard.utils.utils import init_standard_deck

BOARD = [Card('H', 'T'), Card('S', '8'), Card('S', '7'), Card('H', 'J'), Card('H', '8')]

SPOTS = {
    '5bb': dict(stem='fs-7s8s8hThJh-tf', chips_for_each=10, starting_pot=None, max_raises_per_street=None,
                enabled={A.FOLD, A.CHECK, A.CALL, A.ALL_IN, A.BET_HALF_POT, A.BET_2POT, A.RAISE_2X}),
    '100bb': dict(stem='fs-100bb-7s8s8hThJh-tf', chips_for_each=200, starting_pot=40, max_raises_per_street=2,
                  enabled={A.FOLD, A.CHECK, A.CALL, A.ALL_IN, A.BET_THIRD_POT, A.BET_POT, A.BET_2POT, A.RAISE_2X}),
}


def make_env(spot: dict) -> NolimitholdemEnv:
    config = {
        'fixed_public_cards': BOARD,
        'starting_stage': 'river',
        'chips_for_each': spot['chips_for_each'],
        'starting_pot': spot['starting_pot'],
        'max_raises_per_street': spot['max_raises_per_street'],
        'disabled_actions': set(A) - spot['enabled'],
        'dealer_id': 0,  # Fix Player 0 as the dealer
    }
    return rlcard.make('no-limit-holdem', config=config)


def main():
    args = argparse.ArgumentParser(description='Full-tree solve of a river ground-truth spot')
    args.add_argument('--spot', choices=sorted(SPOTS), default='5bb', help='which reference spot')
    args.add_argument('--solve', action='store_true', help='re-solve and overwrite the ground-truth files')
    args.add_argument('--iters', type=int, default=10000, help='CFR iterations for --solve')
    args = args.parse_args()
    spot = SPOTS[args.spot]
    strat_file, values_file = f"strat-{spot['stem']}.npy", f"values-{spot['stem']}.npy"

    env = make_env(spot)
    env.reset()
    agent = GTCFRAgent(env, full_solve=True)

    if args.solve:
        agent.solver.n_expansions_per_regret_updates = 1 / args.iters
        agent.solver.solve(env.game)
        strat = agent.solver.tree.get_cum_strategy(0)
        value = agent.solver.tree.get_values(0).numpy()
        np.save(strat_file, strat)
        np.save(values_file, value)
        print(f'--> Saved ground truth {strat_file} / {values_file} after {args.iters} iterations ({int(agent.solver.tree.n_nodes)} nodes)')
    else:
        strat = np.load(strat_file)
        value = np.load(values_file)

    #
    # Print the results: one column per legal root action, hands grouped by class
    #
    deck = init_standard_deck()
    actions = env.game.get_legal_actions()
    player = env.game.game_pointer
    print(f"Spot {args.spot}: pot {env.game.dealer.pot}, stacks {[p.remained_chips for p in env.game.players]}, "
          f"player {player} to act, actions {[a.name for a in actions]}")
    hand_classes = defaultdict(list)
    class_names = {}
    for hand in range(1326):
        clss, class_name = get_hand_class(hand, env.game.public_cards)
        hand_classes[clss].append(hand)
        class_names[clss] = class_name
    for clss in hand_classes.keys():
        hand_classes[clss] = sorted(hand_classes[clss], key=lambda hand: value[player, hand], reverse=True)
    for clss in sorted(hand_classes.keys()):
        if clss == -1:  # skip invalid hands
            continue
        print(f'---> {class_names[clss]}')
        for hand in hand_classes[clss]:
            card1, card2 = get_2d_coors(hand)
            val = value[player, hand]
            cols = '   '.join(f'{a.name} {strat[i, hand]:.3f}' for i, a in enumerate(actions))
            print(f"{' ' if val >= 0 else ''} {val:.3f} | {deck[card1]} {deck[card2]} -  {cols}   -> {actions[int(np.argmax(strat[:, hand]))].name}")


if __name__ == '__main__':  # Needed for multiprocessing
    main()
