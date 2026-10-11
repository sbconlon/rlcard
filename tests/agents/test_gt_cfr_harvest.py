"""Harvest invariants (M0 plan §7.1): targets read from a full solve after the average-strategy
evaluation pass. Fast checks run always (one 45-node solve, ~15 s on the laptop); the direct
fixed-belief comparison of interior nodes is slow and runs only with GTCFR_SLOW_TESTS=1.
Run with PYTHONPATH set:  python -m unittest tests.agents.test_gt_cfr_harvest"""
import contextlib
import copy
import io
import os
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
import unittest
import numpy as np

from rlcard.agents.gt_cfr_agent import spot_sampler as sp
from rlcard.agents.gt_cfr_agent.cfvn import CounterfactualValueNetwork
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRSolver
from rlcard.agents.gt_cfr_agent.utils import uniform_range
from rlcard.games.base import Card

BOARD = [Card('H', 'T'), Card('S', '8'), Card('S', '7'), Card('H', 'J'), Card('H', '8')]


def full_solve(game, ranges, iters):
    s = GTCFRSolver(full_solve=True, n_expansions_per_regret_update=1.0 / iters, n_query_solvers=0, n_trainers=0,
                    total_actions=game.get_num_actions())
    with contextlib.redirect_stdout(io.StringIO()):
        strat, vals = s.solve(game, input_player_ranges=ranges)
    return s, strat, np.asarray(vals)


class TestHarvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.env = sp.make_river_env(BOARD, pot=120)                       # 45-node tree: fast
        game = cls.env.game
        equity = sp.equity_vector(game)
        valid = uniform_range(BOARD) > 0
        cls.ranges, _ = sp.sample_range_pair(rng, equity, valid)
        cls.solver, cls.root_strat, cls.root_vals = full_solve(game, cls.ranges, 300)
        cls.tree = cls.solver.tree
        cls.tree.evaluate_average_strategy()
        cls.targets = cls.tree.harvest(0.01)
        cls.n_actions = game.get_num_actions()

    def test_targets_exist_and_root_is_first(self):
        self.assertGreater(len(self.targets), 5)
        self.assertEqual(self.targets[0]['node'], 0)
        np.testing.assert_allclose(self.targets[0]['ranges'], self.ranges, atol=1e-6)
        np.testing.assert_allclose(self.targets[0]['reach_mass'], [1.0, 1.0], atol=1e-6)

    def test_ranges_normalized_and_strategies_rows_sum_to_one(self):
        for t in self.targets:
            np.testing.assert_allclose(t['ranges'].sum(axis=1), [1.0, 1.0], atol=1e-5)
            legal = t['game'].get_legal_actions()
            self.assertEqual(t['strategy'].shape, (len(legal), 1326))
            sums = t['strategy'].sum(axis=0)
            live = t['ranges'][t['game'].game_pointer] > 0
            np.testing.assert_allclose(sums[live], 1.0, atol=1e-4)

    def test_zero_sum_under_reach(self):
        # values are linear in the opponent's reach mass; with normalized ranges the
        # reach-weighted values of the two players cancel at every node
        for t in self.targets:
            r, v = t['ranges'], t['values']
            total = float((r[0] * v[0]).sum() + (r[1] * v[1]).sum())
            self.assertLess(abs(total), 1e-3 * t['pot'])

    def test_root_matches_solver_output_under_average_strategy(self):
        root = self.targets[0]
        np.testing.assert_allclose(root['strategy'], self.root_strat, atol=1e-6)      # average strategy is unchanged by the pass
        np.testing.assert_array_equal(root['values'], self.tree.get_values(0).numpy())

    def test_training_target_layout(self):
        feat = CounterfactualValueNetwork(num_actions=self.n_actions, num_neurons_per_layer=8, num_layers=1,
                                          n_query_solvers=0, n_trainers=0, use_shared_weights=False)
        t = self.targets[-1]
        x, s, v = feat.make_training_target(t['game'], t['ranges'], t['strategy'], t['values'] / t['pot'])
        self.assertEqual(x.shape, (feat.INPUT_DIM,)); self.assertEqual(s.shape, (self.n_actions, 1326)); self.assertEqual(v.shape, (2, 1326))
        np.testing.assert_allclose(x[-2 * 1326:], t['ranges'].reshape(-1), atol=1e-6)   # normalized ranges in the input
        self.assertLess(np.abs(v).max(), 6.0)                                             # pot-normalized values stay small

    @unittest.skipUnless(os.environ.get('GTCFR_SLOW_TESTS') == '1', 'set GTCFR_SLOW_TESTS=1 (several minutes)')
    def test_interior_nodes_match_direct_solves(self):
        # The fixture solve uses 300 iterations for speed; this check needs the generator's
        # default (1000) on both sides, as in the P0 spike (mean 0.5 % of pot, worst 1.2 %).
        solver, _, _ = full_solve(self.env.game, self.ranges, 1000)
        solver.tree.evaluate_average_strategy()
        targets = solver.tree.harvest(0.01)
        picks = [t for t in targets if t['node'] != 0][:6]
        for t in picks:
            _, s_d, v_d = full_solve(copy.deepcopy(t['game']), t['ranges'], 1000)
            valid = t['ranges'] > 0
            err = np.abs(t['values'] - v_d)[valid].mean() / t['pot']
            self.assertLess(err, 0.02, f'node {t["node"]}: value error {err:.4f} of pot')


if __name__ == '__main__':
    unittest.main()
