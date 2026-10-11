"""Spot sampler invariants (M0 plan §7.1): equity, range shapes, spots, reproducibility.
Run from a scratch cwd with PYTHONPATH set, e.g.  python -m unittest tests.agents.test_gt_cfr_spot_sampler"""
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
import unittest
import numpy as np

from rlcard.agents.gt_cfr_agent import spot_sampler as sp
from rlcard.agents.gt_cfr_agent.utils import get_1d_coor, uniform_range
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action

BOARD = [Card('H', 'T'), Card('S', '8'), Card('S', '7'), Card('H', 'J'), Card('H', '8')]   # Th 8s 7s Jh 8h


def hand(c1: str, c2: str) -> int:
    ids = sorted(Card(c[1], c[0]).to_int() for c in (c1, c2))
    return get_1d_coor(ids[0], ids[1])


class TestEquity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = sp.make_river_env(BOARD, pot=40)
        cls.equity = sp.equity_vector(cls.env.game)
        cls.valid = uniform_range(BOARD) > 0

    def test_nuts_and_air(self):
        self.assertAlmostEqual(self.equity[hand('9H', 'QH')], 1.0, places=6)      # straight flush
        self.assertLess(self.equity[hand('2C', '3D')], 0.10)                        # no pair: only ties with other air
        self.assertTrue((self.equity[~self.valid] == 0).all())
        self.assertTrue((self.equity[self.valid] > 0).all() and (self.equity[self.valid] <= 1).all())

    def test_monotone_in_strength(self):
        self.assertGreater(self.equity[hand('8C', '8D')], self.equity[hand('TC', 'TD')])   # quads > full house
        self.assertGreater(self.equity[hand('TC', 'TD')], self.equity[hand('AC', 'KD')])   # full house > ace high


class TestRanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = sp.make_river_env(BOARD, pot=40)
        cls.equity = sp.equity_vector(cls.env.game)
        cls.valid = uniform_range(BOARD) > 0
        cls.rng = np.random.default_rng(0)

    def assert_valid_range(self, r):
        self.assertEqual(r.shape, (1326,)); self.assertEqual(r.dtype, np.float32)
        self.assertAlmostEqual(float(r.sum()), 1.0, places=5)
        self.assertTrue((r[~self.valid] == 0).all())
        self.assertGreaterEqual(int((r > 0).sum()), 10)

    def test_shapes_and_support(self):
        for beta in (-3.0, 0.0, 6.0):
            self.assert_valid_range(sp.sample_range(self.rng, self.equity, self.valid, beta))
            self.assert_valid_range(sp.sample_range(self.rng, self.equity, self.valid, beta, sparse_frac=0.8, noise_sigma=0.5))

    def test_sparse_has_fewer_hands(self):
        dense = sp.sample_range(np.random.default_rng(1), self.equity, self.valid, 2.0)
        sparse = sp.sample_range(np.random.default_rng(1), self.equity, self.valid, 2.0, sparse_frac=0.7)
        self.assertLess(int((sparse > 0).sum()), int((dense > 0).sum()))

    def test_strong_beta_concentrates_on_top_hands(self):
        top = self.equity >= np.quantile(self.equity[self.valid], 0.9)
        strong = sp.sample_range(self.rng, self.equity, self.valid, 6.0)
        weak = sp.sample_range(self.rng, self.equity, self.valid, -2.0)
        uniform = self.valid / self.valid.sum()
        self.assertGreater(strong[top].sum(), uniform[top].sum() * 2)
        self.assertLess(weak[top].sum(), uniform[top].sum())

    def test_pair_and_meta(self):
        for _ in range(20):
            ranges, meta = sp.sample_range_pair(self.rng, self.equity, self.valid)
            self.assertEqual(ranges.shape, (2, 1326))
            for pid in range(2):
                self.assert_valid_range(ranges[pid])
            self.assertIn(meta['component'], ('pair', 'independent'))
            self.assertEqual(len(meta['players']), 2)


class TestSpots(unittest.TestCase):
    def test_pot_model(self):
        rng = np.random.default_rng(3)
        pots = [sp.sample_pot(rng) for _ in range(500)]
        self.assertTrue(all(p % 2 == 0 and 8 <= p <= 360 for p in pots))
        self.assertGreater(len(set(pots)), 50)

    def test_spot_is_a_river_root(self):
        rng = np.random.default_rng(5)
        seen_facing = seen_open = False
        for _ in range(12):
            env, ranges, meta = sp.sample_river_spot(rng)
            g = env.game
            self.assertEqual(len(g.public_cards), 5)
            self.assertEqual(g.stage.name, 'RIVER')
            self.assertFalse(env.is_over())
            self.assertEqual(ranges.shape, (2, 1326))
            legal = g.get_legal_actions()
            if meta['facing_bet'] is None:
                seen_open = True
                self.assertEqual(g.dealer.pot, meta['pot_start'])
                self.assertIn(Action.CHECK, legal)
            else:
                seen_facing = True
                self.assertGreater(g.dealer.pot, meta['pot_start'])
                self.assertIn(Action.FOLD, legal); self.assertIn(Action.CALL, legal)
                self.assertEqual(meta['actor'], g.game_pointer)
        self.assertTrue(seen_facing and seen_open)

    def test_reproducible(self):
        a = sp.sample_river_spot(np.random.default_rng(11)); b = sp.sample_river_spot(np.random.default_rng(11))
        self.assertEqual(a[2], b[2])
        np.testing.assert_array_equal(a[1], b[1])


if __name__ == '__main__':
    unittest.main()
