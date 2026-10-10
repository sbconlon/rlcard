# ===================================================================== #
#  Single-process Student-of-Games training loop on river-only games    #
# ===================================================================== #
#
# Every game starts on the river (random board, or one fixed board), so the
# public tree has no chance nodes and GT-CFR search, fixed-belief query
# solves and CFVN training all run in this one process. One cycle is:
#
#   1. evaluate:   query the CFVN on the golden river spot (network alone) and,
#                  optionally, run a GT-CFR search there (network + search);
#                  both are compared to the full-solve ground truth
#   2. self-play:  --hands-per-cycle hands; every search logs ~q_search frontier
#                  queries onto the CFVN's query queue
#   3. targets:    each logged query is solved with fixed beliefs (no gadget),
#                  either exactly (full tree, --targets exact) or by GT-CFR search
#                  (--targets searched, which logs recursive queries), and packaged
#                  as a (input, strategy, values) training target
#   4. train:      --updates-per-cycle batch updates while the buffer holds a batch
#   5. checkpoint: weights, optimizer state, replay buffer, cycle counter, metrics row
#
# Resume with the same --out directory and --resume.
#
# Example (two cycles, exact targets, paper learning rate):
#
#   python train_river.py --out runs/river-smoke --cycles 2 --lr 1e-4
#
# External imports
import argparse
import contextlib
import csv
import io
import json
import os
import pickle
import queue
import random
import sys
import time
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')  # leave GPU memory for cuBLAS; must precede the TensorFlow import
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')   # CFRTree recompiles on purpose whenever the tree changes; hide the retracing warnings

# Internal imports
import rlcard
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent, GTCFRSolver
from rlcard.agents.gt_cfr_agent.utils import uniform_range
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_BOARD = 'Th8s7sJh8h'   # the river ground-truth spot of cfr_full_solve.py
GOLDEN_STRAT = os.path.join(SCRIPT_DIR, 'strat-fs-7s8s8hThJh-tf.npy')
GOLDEN_VALUES = os.path.join(SCRIPT_DIR, 'values-fs-7s8s8hThJh-tf.npy')
DISABLED_ACTIONS = {Action.BET_POT, Action.BET_5POT, Action.RAISE_3X, Action.RAISE_5X}

METRICS_COLUMNS = ['cycle', 'timestamp', 'board', 'actions', 'hand_time_s',
                   'n_queries', 'query_time_s', 'buffer_size', 'n_updates', 'mean_loss',
                   'net_strat_mae', 'net_values_mae', 'search_strat_mae', 'search_values_mae',
                   'eval_time_s', 'cycle_time_s']


# --------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description='Single-process GT-CFR training on river-only games',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    run = p.add_argument_group('run')
    run.add_argument('--out', default=os.path.join('runs', time.strftime('river-%Y%m%d-%H%M%S')),
                     help='run directory: checkpoint, metrics.csv, state.json')
    run.add_argument('--resume', action='store_true', help='continue from the checkpoint in --out')
    run.add_argument('--cycles', type=int, default=2, help='cycles to run (in total, counting resumed ones)')
    run.add_argument('--hands-per-cycle', type=int, default=1, help='self-play hands per cycle')
    run.add_argument('--seed', type=int, default=0, help='numpy / python / TensorFlow seed')
    run.add_argument('--verbose', action='store_true', help='show the solver and trainer prints')

    game = p.add_argument_group('game')
    game.add_argument('--board', default='random',
                      help="river board for self-play, e.g. Th8s7sJh8h, or 'random' for a new board each hand")
    game.add_argument('--chips', type=int, default=10, help='starting stack (big blinds)')

    search = p.add_argument_group('search')
    search.add_argument('-s', '--expansions', type=int, default=3, help='GT-CFR expansions per search (s)')
    search.add_argument('-c', '--c', type=float, default=0.01,
                        help='expansions per CFR update (c): 0.01 = 100 CFR iterations per expansion')
    search.add_argument('--q-search', type=float, default=0.9, help='expected queries logged per self-play search')
    search.add_argument('--q-recursive', type=float, default=0.1, help='expected queries logged per searched query solve')

    tgt = p.add_argument_group('targets')
    tgt.add_argument('--targets', choices=['exact', 'searched'], default='exact',
                     help='exact = full-tree fixed-belief solve; searched = GT-CFR search with the current CFVN')
    tgt.add_argument('--exact-iters', type=int, default=1000, help='CFR iterations of an exact target solve')
    tgt.add_argument('--max-queries-per-cycle', type=int, default=64, help='cap on query solves per cycle')

    net = p.add_argument_group('network')
    net.add_argument('--lr', type=float, default=1e-4, help='initial learning rate (paper: 1e-4)')
    net.add_argument('--batch-size', type=int, default=32, help='batch size (paper: 1024)')
    net.add_argument('--layers', type=int, default=3, help='hidden layers (paper: 6)')
    net.add_argument('--width', type=int, default=2048, help='neurons per hidden layer (paper: 2048)')
    net.add_argument('--updates-per-cycle', type=int, default=8, help='batch updates per cycle (if a batch is available)')
    net.add_argument('--buffer-size', type=int, default=int(1e6), help='replay buffer capacity')
    net.add_argument('--max-uses', type=int, default=10, help='evict a target after this many samples')

    ev = p.add_argument_group('evaluation')
    ev.add_argument('--eval-every', type=int, default=1, help='evaluate every n cycles (0 = never)')
    ev.add_argument('--no-eval-search', action='store_true',
                    help='skip the GT-CFR search on the golden spot (network-only evaluation)')
    return p.parse_args()


def parse_board(text: str) -> list:
    """'Th8s7sJh8h' -> [Card('H','T'), ...]"""
    text = text.strip()
    assert len(text) == 10, f'A river board has 5 cards (10 characters), got {text!r}'
    cards = [Card(text[i + 1].upper(), text[i].upper()) for i in range(0, 10, 2)]
    assert len({c.to_int() for c in cards}) == 5, f'Repeated card in board {text!r}'
    return cards


def board_str(cards) -> str:
    return ''.join(f'{c.rank}{c.suit.lower()}' for c in cards)


def action_names(game, n_prefix: int) -> list:
    """The hand's actions. A river start fast-forwards the earlier streets, so the trajectory
    opens with a fixed-length prefix (hole-card deals, synthetic CALL/CHECKs, board deals when the
    board is random) measured once after env.reset(); the Action values follow."""
    return [Action(v).name for v in game.trajectory[n_prefix:]]


def make_env(board, chips: int):
    config = {'fixed_public_cards': parse_board(board) if board != 'random' else [],
              'starting_stage': 'river', 'chips_for_each': chips, 'dealer_id': 0,
              'disabled_actions': DISABLED_ACTIONS}
    return rlcard.make('no-limit-holdem', config=config)


@contextlib.contextmanager
def quiet(enabled: bool):
    """Swallow the solver's per-call prints unless --verbose."""
    if enabled:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


# --------------------------------------------------------------------- #
#  Checkpointing
# --------------------------------------------------------------------- #

def paths(run_dir):
    return {k: os.path.join(run_dir, v) for k, v in dict(
        weights='weights.h5', optimizer='optimizer.pkl', buffer='replay_buffer.pkl',
        state='state.json', metrics='metrics.csv').items()}


def atomic_write(path, write_fn):
    tmp = path + '.tmp'
    write_fn(tmp)
    os.replace(tmp, path)


def save_checkpoint(run_dir, cfvn, next_cycle, args):
    p = paths(run_dir)
    atomic_write(p['weights'], lambda t: cfvn.network.save_weights(t, save_format='h5'))
    atomic_write(p['optimizer'], lambda t: pickle.dump(cfvn.optimizer.get_weights(), open(t, 'wb')))
    # ~58 KB per target (2709 + 9 * 1326 float32); fine for thousands of targets
    atomic_write(p['buffer'], lambda t: pickle.dump(
        (list(cfvn.replay_buffer.buffer), list(cfvn.replay_buffer.sample_counts)), open(t, 'wb')))
    atomic_write(p['state'], lambda t: json.dump(
        {'next_cycle': next_cycle, 'args': vars(args), 'network_config': cfvn.network_config},
        open(t, 'w'), indent=2, default=str))


def load_checkpoint(run_dir, cfvn, args):
    p = paths(run_dir)
    state = json.load(open(p['state']))
    for key in ('num_neurons_per_layer', 'num_layers', 'batch_size'):
        if state['network_config'][key] != cfvn.network_config[key]:
            sys.exit(f'--resume: checkpoint {key}={state["network_config"][key]} '
                     f'but the command line asks for {cfvn.network_config[key]}')
    cfvn.network.load_weights(p['weights'])
    try:
        opt_weights = pickle.load(open(p['optimizer'], 'rb'))
        if opt_weights:
            cfvn.optimizer._create_all_weights(cfvn.network.trainable_variables)   # Keras OptimizerV2 (TF 2.10)
            cfvn.optimizer.set_weights(opt_weights)
    except Exception as e:  # noqa: BLE001 - a lost optimizer state is a warning, not a stop
        print(f'WARNING: optimizer state not restored ({e}); Adam moments start fresh')
    items, counts = pickle.load(open(p['buffer'], 'rb'))
    for item, count in zip(items, counts):
        cfvn.replay_buffer.buffer.append(item)
        cfvn.replay_buffer.sample_counts.append(count)
    return state['next_cycle']


# --------------------------------------------------------------------- #
#  Evaluation on the golden river spot
# --------------------------------------------------------------------- #

def legal_rows(game):
    """Rows of the CFVN's (N_ACTIONS, 1326) strategy that are legal at this node, in legal-action order."""
    all_actions = game.get_all_actions()
    return [all_actions.index(a) for a in game.get_legal_actions()]


def eval_network(cfvn, game, golden_strat, golden_values):
    ranges = np.tile(uniform_range(game.public_cards), (game.num_players, 1)).astype(np.float32)
    vect = np.asarray(cfvn.to_vect(game, ranges), dtype=np.float32)
    strat, values = cfvn.query(tf.constant(vect[None, :]))
    strat = strat.numpy()[0][legal_rows(game)]
    values = values.numpy()[0]
    return float(np.abs(strat - golden_strat).mean()), float(np.abs(values - golden_values).mean())


def eval_search(solver, env, golden_strat, golden_values, verbose):
    env.reset()
    with quiet(verbose):
        strat, values = solver.solve(env.game)
    solver.reset()
    values = np.asarray(values)
    return float(np.abs(strat - golden_strat).mean()), float(np.abs(values - golden_values).mean())


# --------------------------------------------------------------------- #
#  Main loop
# --------------------------------------------------------------------- #

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    p = paths(args.out)
    random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)

    #
    # Environments: self-play (random or fixed board) and the golden evaluation spot
    #
    env = make_env(args.board, args.chips)
    eval_env = make_env(GOLDEN_BOARD, args.chips)
    eval_env.reset()
    golden_strat = np.load(GOLDEN_STRAT)
    golden_values = np.load(GOLDEN_VALUES)
    n_actions = env.game.get_num_actions()
    env.reset()
    n_prefix = len(env.game.trajectory)   # see action_names

    #
    # One CFVN owns the query queue and the replay buffer. No worker processes.
    #
    cfvn = CounterfactualValueNetwork(num_actions=n_actions,
                                      num_neurons_per_layer=args.width, num_layers=args.layers,
                                      batch_size=args.batch_size, init_learning_rate=args.lr,
                                      max_replay_buffer_size=args.buffer_size,
                                      max_grad_updates_per_exmpl=args.max_uses,
                                      q_recursive=args.q_recursive,
                                      n_query_solvers=0, n_trainers=0, use_shared_weights=False)
    cfvn.init_replay_buffer()

    #
    # Self-play agent, query solver, evaluation solver (logs nothing)
    #
    agent = GTCFRAgent(env, prob_query_solve=args.q_search, n_expansions=args.expansions,
                       n_expansions_per_regret_update=args.c, n_query_solvers=0, n_trainers=0,
                       input_cfvn=cfvn)
    if args.targets == 'exact':
        query_solver = GTCFRSolver(full_solve=True, n_expansions_per_regret_update=1.0 / args.exact_iters,
                                   n_query_solvers=0, n_trainers=0, total_actions=n_actions)
    else:
        query_solver = GTCFRSolver(input_cfvn=cfvn, prob_query_solve=args.q_recursive,
                                   n_expansions=args.expansions, n_expansions_per_regret_update=args.c,
                                   n_query_solvers=0, n_trainers=0, total_actions=n_actions)
    eval_solver = GTCFRSolver(input_cfvn=cfvn, prob_query_solve=0.0,
                              n_expansions=args.expansions, n_expansions_per_regret_update=args.c,
                              n_query_solvers=0, n_trainers=0, total_actions=n_actions)

    #
    # Resume or start fresh
    #
    first_cycle = 0
    if args.resume:
        if not os.path.exists(p['state']):
            sys.exit(f'--resume: no checkpoint in {args.out}')
        first_cycle = load_checkpoint(args.out, cfvn, args)
        print(f'Resumed from {args.out}: next cycle {first_cycle}, replay buffer {cfvn.replay_buffer.size()}')
    elif os.path.exists(p['state']):
        sys.exit(f'{args.out} already holds a run; pass --resume to continue it or pick another --out')
    if not os.path.exists(p['metrics']):
        with open(p['metrics'], 'w', newline='') as f:
            csv.writer(f).writerow(METRICS_COLUMNS)
    print(f'Run {args.out}: cycles {first_cycle}..{args.cycles - 1}, board {args.board}, s={args.expansions} c={args.c} '
          f'q_search={args.q_search}, targets={args.targets}, lr={args.lr} batch={args.batch_size} '
          f'net={args.layers}x{args.width}', flush=True)

    for cycle in range(first_cycle, args.cycles):
        t_cycle = time.time()
        row = dict.fromkeys(METRICS_COLUMNS, '')
        row['cycle'] = cycle
        row['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

        #
        # 1. Evaluate on the golden spot
        #
        if args.eval_every and cycle % args.eval_every == 0:
            t0 = time.time()
            row['net_strat_mae'], row['net_values_mae'] = eval_network(cfvn, eval_env.game, golden_strat, golden_values)
            if not args.no_eval_search:
                row['search_strat_mae'], row['search_values_mae'] = eval_search(
                    eval_solver, eval_env, golden_strat, golden_values, args.verbose)
            row['eval_time_s'] = round(time.time() - t0, 1)

        #
        # 2. Self-play: searches log queries onto cfvn.query_queue
        #
        t0 = time.time()
        boards, actions = [], []
        for _ in range(args.hands_per_cycle):
            with quiet(args.verbose):
                agent.self_play()
            boards.append(board_str(env.game.public_cards))
            actions.append(' '.join(action_names(env.game, n_prefix)))
        row['hand_time_s'] = round(time.time() - t0, 1)
        row['board'] = ' | '.join(boards)
        row['actions'] = ' | '.join(actions)

        #
        # 3. Solve the logged queries with fixed beliefs; package training targets
        #
        t0 = time.time()
        n_queries = 0
        while n_queries < args.max_queries_per_cycle:
            try:
                game, player_ranges = cfvn.query_queue.get(timeout=0.5)   # the queue's feeder thread may lag a put
            except queue.Empty:
                break
            with quiet(args.verbose):
                target_strat, target_values = query_solver.solve(game, input_player_ranges=player_ranges)
            query_solver.reset()
            cfvn.add_to_replay_buffer(cfvn.make_training_target(game, player_ranges, target_strat, target_values))
            n_queries += 1
        row['n_queries'] = n_queries
        row['query_time_s'] = round(time.time() - t0, 1)
        row['buffer_size'] = cfvn.replay_buffer.size()

        #
        # 4. Train
        #
        losses = []
        while len(losses) < args.updates_per_cycle and cfvn.replay_buffer.size() >= cfvn.batch_size:
            with quiet(args.verbose):
                losses.append(float(cfvn.batch_update()))
        row['n_updates'] = len(losses)
        row['mean_loss'] = round(float(np.mean(losses)), 6) if losses else ''

        #
        # 5. Checkpoint and log
        #
        row['cycle_time_s'] = round(time.time() - t_cycle, 1)
        save_checkpoint(args.out, cfvn, cycle + 1, args)
        with open(p['metrics'], 'a', newline='') as f:
            csv.writer(f).writerow([row[k] for k in METRICS_COLUMNS])
        print(f"[cycle {cycle}] board {row['board']} | actions {row['actions']} | "
              f"hand {row['hand_time_s']}s | queries {n_queries} ({row['query_time_s']}s) | "
              f"buffer {row['buffer_size']} | updates {row['n_updates']} loss {row['mean_loss']} | "
              f"net MAE strat {row['net_strat_mae']} values {row['net_values_mae']} | "
              f"search MAE strat {row['search_strat_mae']} values {row['search_values_mae']} | "
              f"cycle {row['cycle_time_s']}s", flush=True)

    #
    # Queries logged but not yet solved are dropped. Without this the queue's feeder
    # thread blocks interpreter exit while trying to flush them.
    #
    cfvn.query_queue.cancel_join_thread()
    print(f'Done. Checkpoint and metrics in {args.out}')


if __name__ == '__main__':
    main()
