# ============================================================ #
#  Single-process training cycle on the river ground-truth spot  #
# ============================================================ #
#
# One process plays the whole Student-of-Games loop in sequence:
#
#   1. test solve:   GT-CFR on the river root, compared to the full-solve ground truth
#   2. self-play:    one hand; every search logs ~q_search frontier queries
#   3. query solve:  a second solver solves each logged query with fixed beliefs (no gadget)
#                    and packages (input, strategy, values) training targets
#   4. training:     batch updates while the replay buffer holds a full batch
#
# Repeat until the test solve matches the ground truth within --eps.

# External imports
import argparse
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # leave GPU memory for cuBLAS; must precede the TensorFlow import
import numpy as np

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent, GTCFRSolver
from rlcard.agents.gt_cfr_agent.utils import get_2d_coors
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck


def main():
    args = argparse.ArgumentParser(description='Single-process GT-CFR training cycle')
    args.add_argument('--q-search', type=float, default=0.9, help='expected queries logged per self-play search')
    args.add_argument('--q-recursive', type=float, default=0.1, help='expected queries logged per query solve')
    args.add_argument('--expansions', type=int, default=3, help='GT-CFR expansions per search (s)')
    args.add_argument('--c', type=float, default=0.01, help='expansions per CFR update (c): 0.01 = 100 CFR iterations per expansion')
    args.add_argument('--eps', type=float, default=1e-3, help='stop when the test solve is within eps of the ground truth')
    args.add_argument('--max-iters', type=int, default=1000, help='stop after this many cycles regardless')
    args = args.parse_args()

    #
    # The river ground-truth spot
    #
    config = {
        'fixed_public_cards': [Card('H', 'T'), Card('S', '8'), Card('S', '7'), Card('H', 'J'), Card('H', '8')],
        'starting_stage': 'river',
        'chips_for_each': 10,
        'disabled_actions': {Action.BET_POT, Action.BET_5POT, Action.RAISE_3X, Action.RAISE_5X},
        'dealer_id': 0,
    }
    env = rlcard.make('no-limit-holdem', config=config)

    #
    # Self-play agent and query solver share one CFVN (which owns the query queue and the
    # replay buffer). No worker processes: everything below runs inline.
    #
    agent = GTCFRAgent(env, prob_query_solve=args.q_search, n_expansions=args.expansions,
                       n_expansions_per_regret_update=args.c, n_query_solvers=0, n_trainers=0)
    cfvn = agent.solver.cfvn
    cfvn.init_replay_buffer()
    query_solver = GTCFRSolver(input_cfvn=cfvn, prob_query_solve=args.q_recursive,
                               n_expansions=args.expansions, n_expansions_per_regret_update=args.c,
                               n_query_solvers=0, n_trainers=0)

    true_strat = np.load('strat-fs-7s8s8hThJh-tf.npy')
    true_values = np.load('values-fs-7s8s8hThJh-tf.npy')
    deck = sorted(init_standard_deck(), key=lambda c: c.to_int())
    def hand_str(hand):
        c1, c2 = get_2d_coors(hand)
        return f'{deck[c1]}{deck[c2]}'

    for cycle in range(args.max_iters):
        #
        # 1. Test solve against the ground truth
        #
        print('\n=====================================================================')
        print(f'Cycle #{cycle}')
        agent.solver.reset()
        env.reset()
        pred_strat, pred_values = agent.solver.solve(env.game)
        pred_values = np.asarray(pred_values)
        err_strat = np.abs(pred_strat - true_strat)
        err_values = np.abs(pred_values - true_values)
        print(f'Strategy MAE = {err_strat.mean():.4f}   Values MAE = {err_values.mean():.4f}')
        worst = int(np.argmax(err_strat.sum(axis=0)))
        print(f'Worst strategy hand {hand_str(worst)}: expected {np.round(true_strat[:, worst], 3)} got {np.round(pred_strat[:, worst], 3)}')
        worst = int(np.argmax(err_values.sum(axis=0)))
        print(f'Worst value hand    {hand_str(worst)}: expected {np.round(true_values[:, worst], 3)} got {np.round(pred_values[:, worst], 3)}')
        if err_strat.max() < args.eps and err_values.max() < args.eps:
            print('--> SUCCESSFUL CONVERGENCE')
            break

        #
        # 2. One self-play hand (its searches log queries onto cfvn.query_queue)
        #
        agent.self_play()
        print(f'Query queue size after self-play = {cfvn.query_queue.qsize()}')

        #
        # 3. Solve every logged query with fixed beliefs and add the targets to the replay buffer
        #
        while not cfvn.query_queue.empty():
            game, player_ranges = cfvn.query_queue.get()
            target_strat, target_values = query_solver.solve(game, input_player_ranges=player_ranges)
            cfvn.add_to_replay_buffer(cfvn.make_training_target(game, player_ranges, target_strat, target_values))
            query_solver.reset()
        print(f'Replay buffer size = {cfvn.replay_buffer.size()}')

        #
        # 4. Train while a full batch is available
        #
        n_updates = 0
        while cfvn.replay_buffer.size() >= cfvn.batch_size:
            cfvn.train(0, niters=1)
            n_updates += 1
        print(f'Batch updates this cycle = {n_updates}')


if __name__ == '__main__':  # Needed for multiprocessing
    main()
