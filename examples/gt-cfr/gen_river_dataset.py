# ===================================================================== #
#  River CFVN dataset generator (M0 plan, phase P2)                      #
# ===================================================================== #
#
# For each random river spot (spot_sampler): build the full public tree, solve it
# exactly with fixed beliefs (no gadget), evaluate it under the AVERAGE strategy
# (M0-8), and harvest a training target from every decision node both players
# reach (M0-2): about 20-70 targets per solve instead of one.
#
# Targets are written in shards of --shard-size examples (default 1024, about 45 minutes of
# solving on the laptop; when the disk is full the writer waits and retries instead of quitting):
#
#   shard-NNNNN.npz   X (n, 2709) float32   CFVN input: board one-hot, pot commitments,
#                                           actor one-hot, both NORMALIZED ranges
#                     S (n, A, 1326) float16 average strategy, rows in all_actions order,
#                                           zero rows for illegal actions
#                     V (n, 2, 1326) float16 values / pot at the node (M0-5), for normalized beliefs
#                     pot (n,) int32, depth (n,) int16, reach (n, 2) float32, spot (n,) int32
#   shard-NNNNN.json  per-spot metadata (board, pot, facing bet, range draws, node counts, time)
#   manifest.json     args, seed, spots done, targets written (updated after every shard)
#   progress.csv      one row per spot
#
# Resumable: rerun with the same --out and it continues after the last complete shard
# (spot indices are derived from the seed, so the dataset is reproducible).
#
# Example (one night):  python gen_river_dataset.py --out data\river-v1 --hours 11
#
import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import sys
import time
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from rlcard.agents.gt_cfr_agent import spot_sampler as sp
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRSolver

MIN_FREE_GB = 2.0


def parse_args():
    p = argparse.ArgumentParser(description='Generate river CFVN training targets from exact solves of random spots',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out', required=True, help='dataset directory (created; resumable)')
    p.add_argument('--hours', type=float, default=None, help='wall-clock budget; stop before a spot once exceeded')
    p.add_argument('--max-spots', type=int, default=None, help='stop after this many spots in total')
    p.add_argument('--shard-size', type=int, default=1024, help='targets per shard (about 45 min of solving; a crash loses at most one unwritten shard)')
    p.add_argument('--iters', type=int, default=1000, help='CFR iterations per exact solve')
    p.add_argument('--seed', type=int, default=0, help='dataset seed; spot i uses rng(seed, i)')
    p.add_argument('--chips', type=int, default=200, help='starting stack in chips (big blind = 2)')
    p.add_argument('--max-raises', type=int, default=2, help='raise cap per street')
    p.add_argument('--p-facing-bet', type=float, default=0.3, help='share of roots that face a bet')
    p.add_argument('--p-uniform', type=float, default=0.1, help='per-player chance of a uniform range')
    p.add_argument('--p-pair', type=float, default=0.5, help='share of spots drawn as aggressor/caller pairs')
    p.add_argument('--p-sparse', type=float, default=1 / 3, help='per-player chance of a sparse range')
    p.add_argument('--min-reach', type=float, default=0.01, help='harvest nodes where both players reach with mass above this (below about 0.01 CFR leaves the node unconverged: P2 check)')
    p.add_argument('--verbose', action='store_true', help='show solver prints')
    p.add_argument('--spots-per-process', type=int, default=100,
                   help='restart the worker process after this many spots (the worker leaks about 30 MB of host memory per spot; ~1190 spots '
                        'exhausted 16 GB RAM + pagefile on night 1); 0 = single process')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)   # internal: run one chunk, no supervisor
    return p.parse_args()


@contextlib.contextmanager
def quiet(enabled):
    if enabled:
        yield
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            yield


def atomic_write(path, write_fn):
    tmp = path + '.tmp'
    write_fn(tmp)
    os.replace(tmp, path)


def load_manifest(out, args):
    path = os.path.join(out, 'manifest.json')
    if os.path.exists(path):
        m = json.load(open(path))
        for key in ('seed', 'chips', 'iters', 'max_raises'):
            if m['args'][key] != getattr(args, key):
                sys.exit(f'--{key} differs from the dataset manifest ({m["args"][key]}); use a new --out')
        return m
    return {'args': vars(args), 'spots_done': 0, 'targets': 0, 'shards': 0, 'n_actions': None}


def supervise(args):
    """Run the generator in chunks of --spots-per-process spots, each in a fresh process, until the
    budget or --max-spots is reached. Each chunk resumes from the manifest."""
    t_run = time.time()
    while True:
        if args.hours is not None and time.time() - t_run > args.hours * 3600:
            print(f'Supervisor: wall-clock budget of {args.hours} h reached; stopping.', flush=True); return 0
        os.makedirs(args.out, exist_ok=True)
        manifest_path = os.path.join(args.out, 'manifest.json')
        done = json.load(open(manifest_path))['spots_done'] if os.path.exists(manifest_path) else 0
        if args.max_spots is not None and done >= args.max_spots:
            print(f'Supervisor: {done} spots done; --max-spots reached.', flush=True); return 0
        chunk_end = done + args.spots_per_process
        if args.max_spots is not None:
            chunk_end = min(chunk_end, args.max_spots)
        cmd = [sys.executable, os.path.abspath(__file__), '--worker', '--max-spots', str(chunk_end)]
        if args.hours is not None:
            cmd += ['--hours', f'{max(args.hours - (time.time() - t_run) / 3600, 0.01):.4f}']
        for key in ('out', 'shard_size', 'iters', 'seed', 'chips', 'max_raises', 'p_facing_bet', 'p_uniform',
                    'p_pair', 'p_sparse', 'min_reach'):
            cmd += ['--' + key.replace('_', '-'), str(getattr(args, key))]
        if args.verbose:
            cmd.append('--verbose')
        print(f'Supervisor: spots {done} -> {chunk_end} in a fresh process', flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f'Supervisor: worker exited with code {rc}; retrying in 60 s (shards are intact, it resumes)', flush=True)
            time.sleep(60)


def main():
    args = parse_args()
    if not args.worker and args.spots_per_process > 0:
        raise SystemExit(supervise(args))
    os.makedirs(args.out, exist_ok=True)
    manifest = load_manifest(args.out, args)
    progress_path = os.path.join(args.out, 'progress.csv')
    if not os.path.exists(progress_path):
        with open(progress_path, 'w', newline='') as f:
            csv.writer(f).writerow(['spot', 'timestamp', 'board', 'pot_start', 'pot', 'facing_bet', 'component',
                                    'nodes', 'harvested', 'solve_s', 'harvest_s'])

    #
    # A tiny CFVN instance only for its featurization (to_vect / make_training_target),
    # so shards use exactly the layout the real network is trained and queried with.
    #
    probe_env = sp.make_river_env(sp.sample_board(np.random.default_rng(0)), 40, args.chips, sp.RIVER_SET, args.max_raises)
    n_actions = probe_env.game.get_num_actions()
    if manifest['n_actions'] not in (None, n_actions):
        sys.exit(f'action count {n_actions} differs from the manifest ({manifest["n_actions"]})')
    manifest['n_actions'] = n_actions
    feat = CounterfactualValueNetwork(num_actions=n_actions, num_neurons_per_layer=8, num_layers=1,
                                      n_query_solvers=0, n_trainers=0, use_shared_weights=False)

    spot_idx = manifest['spots_done']
    shard_idx = manifest['shards']
    buf = {'X': [], 'S': [], 'V': [], 'pot': [], 'depth': [], 'reach': [], 'spot': []}
    spot_meta = []
    t_run = time.time()
    print(f'{args.out}: resuming at spot {spot_idx}, shard {shard_idx}, {manifest["targets"]} targets so far; '
          f'{n_actions} actions, {args.iters} iterations per solve', flush=True)

    def flush_shard():
        nonlocal shard_idx, buf, spot_meta
        n = len(buf['X'])
        if n == 0:
            return
        #
        # Disk full: wait rather than quit. The buffer holds up to an hour of solving and the
        # job runs unattended overnight; a stop would throw that work away and idle the GPU
        # until morning (river-v1 night 1 stopped at spot 401 this way).
        #
        waited = 0
        while shutil.disk_usage(args.out).free / 1e9 < MIN_FREE_GB:
            if waited % 600 == 0:
                print(f'WARNING: less than {MIN_FREE_GB:.0f} GB free; holding {n} targets in memory and '
                      f'retrying every minute (waited {waited // 60} min)', flush=True)
            time.sleep(60)
            waited += 60
            if args.hours is not None and time.time() - t_run > (args.hours + 1) * 3600:
                print('WARNING: disk still full one hour past the budget; attempting the write anyway', flush=True)
                break
        stem = os.path.join(args.out, f'shard-{shard_idx:05d}')
        arrays = dict(X=np.stack(buf['X']).astype(np.float32), S=np.stack(buf['S']).astype(np.float16),
                      V=np.stack(buf['V']).astype(np.float16), pot=np.asarray(buf['pot'], np.int32),
                      depth=np.asarray(buf['depth'], np.int16), reach=np.stack(buf['reach']).astype(np.float32),
                      spot=np.asarray(buf['spot'], np.int32))
        atomic_write(stem + '.npz', lambda t: np.savez(open(t, 'wb'), **arrays))
        atomic_write(stem + '.json', lambda t: json.dump(spot_meta, open(t, 'w')))
        manifest['targets'] += n
        manifest['shards'] = shard_idx + 1
        manifest['spots_done'] = spot_idx
        atomic_write(os.path.join(args.out, 'manifest.json'), lambda t: json.dump(manifest, open(t, 'w'), indent=2))
        print(f'  wrote {stem}.npz: {n} targets from {len(spot_meta)} spots; total {manifest["targets"]} targets, '
              f'{manifest["spots_done"]} spots, {(time.time() - t_run) / 3600:.2f} h', flush=True)
        shard_idx += 1
        buf = {k: [] for k in buf}
        spot_meta = []

    while True:
        if args.max_spots is not None and spot_idx >= args.max_spots:
            break
        if args.hours is not None and time.time() - t_run > args.hours * 3600:
            print(f'Wall-clock budget of {args.hours} h reached before spot {spot_idx}; stopping.', flush=True)
            break
        rng = np.random.default_rng([args.seed, spot_idx])
        env, ranges, meta = sp.sample_river_spot(rng, args.chips, sp.RIVER_SET, args.max_raises,
                                                 args.p_facing_bet, args.p_uniform, args.p_pair, args.p_sparse)
        game = env.game
        #
        # Exact fixed-belief solve, then the average-strategy evaluation pass, then harvest
        #
        t0 = time.time()
        solver = GTCFRSolver(full_solve=True, n_expansions_per_regret_update=1.0 / args.iters,
                             n_query_solvers=0, n_trainers=0, total_actions=n_actions)
        with quiet(args.verbose):
            solver.solve(game, input_player_ranges=ranges)
        tree = solver.tree
        tree.evaluate_average_strategy()
        t_solve = time.time() - t0
        t0 = time.time()
        targets = tree.harvest(args.min_reach)
        for t in targets:
            x, s, v = feat.make_training_target(t['game'], t['ranges'], t['strategy'], t['values'] / t['pot'])
            buf['X'].append(x); buf['S'].append(s); buf['V'].append(v)
            buf['pot'].append(t['pot']); buf['depth'].append(t['depth']); buf['reach'].append(t['reach_mass']); buf['spot'].append(spot_idx)
        t_harvest = time.time() - t0
        n_nodes = int(tree.n_nodes)
        solver.reset()
        meta.update({'spot': spot_idx, 'nodes': n_nodes, 'harvested': len(targets), 'solve_s': round(t_solve, 1)})
        spot_meta.append(meta)
        with open(progress_path, 'a', newline='') as f:
            csv.writer(f).writerow([spot_idx, time.strftime('%Y-%m-%d %H:%M:%S'), meta['board'], meta['pot_start'], meta['pot'],
                                    meta['facing_bet'], meta['component'], n_nodes, len(targets), round(t_solve, 1), round(t_harvest, 2)])
        print(f'[spot {spot_idx}] {meta["board"]} pot {meta["pot"]} facing {meta["facing_bet"]} {meta["component"]} | '
              f'{n_nodes} nodes -> {len(targets)} targets | solve {t_solve:.0f}s', flush=True)
        spot_idx += 1
        if len(buf['X']) >= args.shard_size:
            flush_shard()

    flush_shard()
    manifest['spots_done'] = spot_idx
    atomic_write(os.path.join(args.out, 'manifest.json'), lambda t: json.dump(manifest, open(t, 'w'), indent=2))
    print(f'Done: {manifest["targets"]} targets in {manifest["shards"]} shards from {spot_idx} spots '
          f'({(time.time() - t_run) / 3600:.2f} h)', flush=True)


if __name__ == '__main__':
    main()
