# ========================================================================= #
#                                                                           #
# River spot sampler: random public states and root ranges for the CFVN    #
# training-data generator (M0, plan: Repos/rlcard/plan/m0-river-network).  #
#                                                                           #
# Only the ROOT of a solve is sampled here. Every deeper node of the full   #
# solve inherits its ranges from equilibrium play, so the sampler's job is  #
# coverage of root situations with realistic range shapes, not realism at  #
# every node.                                                               #
#                                                                           #
# A range is drawn in four steps (decision 2026-10-10, "realism first"):    #
#   1. weight every hand by exp(beta * equity) for a random temperature     #
#      beta (high -> strong/polarized, ~0 -> wide/capped, negative -> weak) #
#   2. sometimes zero a random 50-90% of hands (a range already narrowed    #
#      by earlier folds and bets)                                           #
#   3. multiply by small log-normal per-hand noise (ragged, not a smooth    #
#      curve in equity)                                                     #
#   4. zero hands sharing a board card; normalize                           #
# Half of the spots draw the two players as a correlated pair (aggressor    #
# vs. capped caller); about one draw in ten is plain uniform (the fixed-    #
# belief anchor).                                                           #
#                                                                           #
# Everything is driven by one numpy Generator so a dataset is reproducible  #
# from its seed.                                                            #
#                                                                           #
# ========================================================================= #

from __future__ import annotations

import numpy as np

import rlcard
from rlcard.agents.gt_cfr_agent.utils import uniform_range
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action, bet_actions_set
from rlcard.utils.utils import init_standard_deck

#
# The product's river action set (M0-1) and the number of valid opponent hands
# every valid river hand faces: 47 unseen cards minus the hand's two = C(45, 2).
#
RIVER_SET = {Action.FOLD, Action.CHECK, Action.CALL, Action.ALL_IN,
             Action.BET_THIRD_POT, Action.BET_POT, Action.BET_2POT, Action.RAISE_2X}
N_OPP_HANDS_RIVER = 990

#
# Pot model in chips (the big blind is 2 chips): four buckets, log-uniform inside.
#
POT_BUCKETS = [((8, 30), 0.35), ((30, 80), 0.35), ((80, 240), 0.20), ((240, 360), 0.10)]

DECK = sorted(init_standard_deck(), key=lambda c: c.to_int())


# ------------------------------------------------------------------------- #
#  Equity
# ------------------------------------------------------------------------- #

def equity_vector(game) -> np.ndarray:
    """Equity of each of the 1326 hands against a uniformly random valid opponent hand
    on game.public_cards (ties count half). Zero on hands blocked by the board.
    Uses the solver's +1 / -1 / 0 payoff matrix, so it stays consistent with the tree."""
    from rlcard.agents.gt_cfr_agent.nodes import compute_payoff_matrix   # runtime import: nodes imports tensorflow
    payoffs = np.asarray(compute_payoff_matrix(game), dtype=np.float64)   # (1326, 1326)
    valid = uniform_range(game.public_cards) > 0
    n_opp = (valid.sum() - 1) - 2 * (52 - len(game.public_cards) - 2)       # valid hands minus those sharing a card with mine
    equity = 0.5 + 0.5 * payoffs.sum(axis=1) / max(n_opp, 1)
    return np.where(valid, equity, 0.0)


# ------------------------------------------------------------------------- #
#  Ranges
# ------------------------------------------------------------------------- #

def sample_range(rng: np.random.Generator, equity: np.ndarray, valid: np.ndarray,
                 beta: float, sparse_frac: float = None, noise_sigma: float = 0.0) -> np.ndarray:
    """One player's range: exp(beta * equity), optional sparsity, optional log-normal noise,
    blocked hands zeroed, normalized to sum to one. float32 (1326,)."""
    w = np.exp(beta * (equity - equity[valid].mean()))      # centred: keeps the exponent small for large |beta|
    if sparse_frac is not None:
        w = w * (rng.random(1326) >= sparse_frac)
    if noise_sigma > 0:
        w = w * rng.lognormal(0.0, noise_sigma, 1326)
    w = w * valid
    if w.sum() <= 0:                                         # a sparse draw emptied the range
        w = valid.astype(np.float64)
    return (w / w.sum()).astype(np.float32)


def _draw_beta(rng: np.random.Generator, kind: str) -> float:
    if kind == 'strong':   return float(rng.normal(4.0, 2.0))
    if kind == 'wide':     return float(rng.normal(0.0, 1.0))
    if kind == 'weak':     return float(rng.normal(-2.0, 1.0))
    if kind == 'aggressor': return float(rng.normal(5.0, 1.5))
    if kind == 'caller':   return float(rng.normal(0.5, 1.0))
    raise ValueError(kind)


def sample_range_pair(rng: np.random.Generator, equity: np.ndarray, valid: np.ndarray,
                      aggressor: int = None, p_uniform: float = 0.1, p_pair: float = 0.5,
                      p_sparse: float = 1 / 3) -> tuple[np.ndarray, dict]:
    """Both players' root ranges, (2, 1326) indexed by player id, plus the draw's metadata.
    `aggressor` is the player id that gets the strong side of a correlated pair (None: random)."""
    kinds = [None, None]
    if rng.random() < p_pair:
        agg = int(rng.integers(2)) if aggressor is None else aggressor
        kinds[agg], kinds[1 - agg] = 'aggressor', 'caller'
        component = 'pair'
    else:
        kinds = [str(rng.choice(['strong', 'wide', 'weak'])) for _ in range(2)]
        component = 'independent'
    ranges = np.zeros((2, 1326), dtype=np.float32)
    meta = {'component': component, 'players': []}
    for pid in range(2):
        if rng.random() < p_uniform:
            ranges[pid] = (valid / valid.sum()).astype(np.float32)
            meta['players'].append({'kind': 'uniform', 'beta': 0.0, 'sparse_frac': None, 'noise_sigma': 0.0})
            continue
        beta = _draw_beta(rng, kinds[pid])
        sparse = float(rng.uniform(0.5, 0.9)) if rng.random() < p_sparse else None
        sigma = float(rng.uniform(0.0, 0.7))
        ranges[pid] = sample_range(rng, equity, valid, beta, sparse, sigma)
        meta['players'].append({'kind': kinds[pid], 'beta': beta, 'sparse_frac': sparse, 'noise_sigma': sigma})
    return ranges, meta


# ------------------------------------------------------------------------- #
#  Spots
# ------------------------------------------------------------------------- #

def sample_board(rng: np.random.Generator) -> list[Card]:
    return [DECK[i] for i in rng.choice(52, 5, replace=False)]


def sample_pot(rng: np.random.Generator, chips: int = 200) -> int:
    """Total pot in chips at the river start: bucketed, log-uniform inside a bucket, even,
    leaving every player at least 20 chips behind."""
    probs = np.array([p for _, p in POT_BUCKETS]); probs /= probs.sum()
    lo, hi = POT_BUCKETS[int(rng.choice(len(POT_BUCKETS), p=probs))][0]
    pot = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
    pot = min(pot, 2 * chips - 40)
    return pot + (pot % 2)


def make_river_env(board: list[Card], pot: int, chips: int = 200, enabled=RIVER_SET, max_raises: int = 2,
                   dealer_id: int = 0):
    config = {'fixed_public_cards': list(board), 'starting_stage': 'river', 'starting_pot': int(pot),
              'chips_for_each': int(chips), 'max_raises_per_street': max_raises,
              'disabled_actions': set(Action) - set(enabled), 'dealer_id': dealer_id}
    env = rlcard.make('no-limit-holdem', config=config)
    env.reset()
    return env


def sample_river_spot(rng: np.random.Generator, chips: int = 200, enabled=RIVER_SET, max_raises: int = 2,
                      p_facing_bet: float = 0.3, p_uniform: float = 0.1, p_pair: float = 0.5,
                      p_sparse: float = 1 / 3):
    """A random river root: (env, ranges (2, 1326) by player id, meta). With probability
    p_facing_bet the out-of-position player has made one random legal bet, so the root is the
    in-position player facing it; the bettor is then the aggressor of a correlated pair."""
    board = sample_board(rng)
    pot = sample_pot(rng, chips)
    env = make_river_env(board, pot, chips, enabled, max_raises)
    game = env.game
    first_actor = game.game_pointer
    facing = None
    if rng.random() < p_facing_bet:
        bets = [a for a in game.get_legal_actions() if a in bet_actions_set]   # pot-fraction bets only: a shove into a small pot is a degenerate root
        if bets:
            facing = bets[int(rng.integers(len(bets)))]
            env.step(facing)
    aggressor = first_actor if facing is not None else None
    equity = equity_vector(game)
    valid = uniform_range(board) > 0
    ranges, meta = sample_range_pair(rng, equity, valid, aggressor, p_uniform, p_pair, p_sparse)
    meta.update({'board': ''.join(f'{c.rank}{c.suit.lower()}' for c in board), 'pot_start': pot,
                 'pot': int(game.dealer.pot), 'chips': chips, 'facing_bet': facing.name if facing else None,
                 'actor': int(game.game_pointer)})
    return env, ranges, meta
