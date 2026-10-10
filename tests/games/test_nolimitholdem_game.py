import unittest

from rlcard.games.limitholdem.player import PlayerStatus
from rlcard.games.nolimitholdem.game import NolimitholdemGame as Game, Stage
from rlcard.games.nolimitholdem.round import Action


def make_game(num_players=2, init_chips=None, dealer_id=0):
    '''Build a game with fixed positions.

    With dealer_id=0 and two players: player 0 is the button, posts the small
    blind and acts first preflop; player 1 posts the big blind, closes the
    preflop action and acts first on every later street.
    '''
    game = Game(num_players=num_players)
    game.dealer_id = dealer_id
    if init_chips is not None:
        game.init_chips = init_chips
    return game


class TestNolimitholdemMethods(unittest.TestCase):

    def test_init_game(self):
        game = make_game()
        state, player_id = game.init_game()
        self.assertEqual(game.get_player_id(), player_id)
        self.assertEqual(0, player_id)  # the button (small blind) acts first preflop

    def test_step(self):
        game = make_game()

        # call
        game.init_game()
        init_not_raise_num = game.round.not_raise_num
        game.step(Action.CALL)
        self.assertEqual(init_not_raise_num + 1, game.round.not_raise_num)

        # fold
        _, player_id = game.init_game()
        game.step(Action.FOLD)
        self.assertEqual(PlayerStatus.FOLDED, game.players[player_id].status)

        # check: the big blind closes a limped preflop round
        game.init_game()
        game.step(Action.CALL)
        game.step(Action.CHECK)
        self.assertEqual(Stage.FLOP, game.stage)

    def test_bet_more_than_chips(self):
        game = make_game()
        game.init_game()
        player = game.players[0]
        in_chips = player.in_chips
        player.bet(50)
        self.assertEqual(50 + in_chips, player.in_chips)

        player.bet(150)
        self.assertEqual(100, player.in_chips)

    def test_step_2(self):
        game = make_game()
        game.init_game()
        self.assertEqual(Stage.PREFLOP, game.stage)
        game.step(Action.CALL)       # small blind limps
        game.step(Action.RAISE_2X)   # big blind raises to 4
        self.assertListEqual([2, 4], game.round.raised)
        game.step(Action.CALL)

        self.assertEqual(Stage.FLOP, game.stage)
        game.step(Action.CHECK)
        game.step(Action.CHECK)

        self.assertEqual(Stage.TURN, game.stage)
        game.step(Action.CHECK)
        game.step(Action.CHECK)

        self.assertEqual(Stage.RIVER, game.stage)

    def test_step_3_players(self):
        game = make_game(num_players=3)

        _, first_player_id = game.init_game()
        self.assertEqual(Stage.PREFLOP, game.stage)
        game.step(Action.CALL)       # under the gun limps
        game.step(Action.CALL)       # small blind completes
        game.step(Action.RAISE_2X)   # big blind raises its option to 4
        game.step(Action.FOLD)       # under the gun folds
        game.step(Action.CALL)       # small blind calls

        self.assertEqual(Stage.FLOP, game.stage)
        self.assertEqual((first_player_id - 2) % 3, game.round.game_pointer)
        game.step(Action.CHECK)
        game.step(Action.BET_POT)
        self.assertListEqual([0, 0, 10], game.round.raised)
        game.step(Action.CALL)

        self.assertEqual(Stage.TURN, game.stage)
        self.assertEqual((first_player_id - 2) % 3, game.round.game_pointer)
        game.step(Action.CHECK)
        game.step(Action.CHECK)

        self.assertEqual(Stage.RIVER, game.stage)

    def test_auto_step(self):
        game = make_game()
        game.init_game()
        self.assertEqual(Stage.PREFLOP, game.stage)
        game.step(Action.ALL_IN)
        game.step(Action.CALL)

        # Both players are all in, so the remaining streets are dealt at once
        self.assertEqual(Stage.RIVER, game.stage)
        self.assertEqual(4, game.round_counter)
        self.assertEqual(200, game.dealer.pot)
        self.assertTrue(game.is_over())

    def test_all_in(self):
        game = make_game()
        _, player_id = game.init_game()
        game.step(Action.ALL_IN)
        self.assertEqual(100, game.round.raised[player_id])
        self.assertEqual(100, game.players[player_id].in_chips)
        self.assertEqual(0, game.players[player_id].remained_chips)
        # Facing a shove that covers the stack, only fold or call remain
        self.assertListEqual([Action.FOLD, Action.CALL], game.get_legal_actions())

    def test_all_in_rounds(self):
        game = make_game()
        game.init_game()
        game.step(Action.CALL)
        game.step(Action.CHECK)
        self.assertEqual(1, game.round_counter)

        game.step(Action.CHECK)
        game.step(Action.ALL_IN)
        self.assertListEqual([Action.FOLD, Action.CALL], game.get_legal_actions())
        game.step(Action.CALL)
        self.assertEqual(4, game.round_counter)
        self.assertEqual(200, game.dealer.pot)

    def test_raise_chain(self):
        game = make_game()
        _, player_id = game.init_game()
        game.step(Action.RAISE_2X)   # raise to 2 * 2
        self.assertEqual(4, game.round.raised[player_id])

        player_id = game.round.game_pointer
        game.step(Action.RAISE_2X)   # raise to 2 * 4
        self.assertEqual(8, game.round.raised[player_id])

        player_id = game.round.game_pointer
        game.step(Action.RAISE_2X)   # raise to 2 * 8
        self.assertEqual(16, game.round.raised[player_id])

        game.step(Action.CALL)
        self.assertEqual(Stage.FLOP, game.stage)
        player_id = game.round.game_pointer
        game.step(Action.BET_POT)    # pot is 32 on the flop
        self.assertEqual(32, game.round.raised[player_id])

    def test_bet_half_pot(self):
        game = make_game()
        game.init_game()
        # Preflop the blinds are live bets, so no bet action is offered
        for action in game.get_legal_actions():
            self.assertNotIn(action, {Action.BET_HALF_POT, Action.BET_POT, Action.BET_2POT, Action.BET_5POT})
        game.step(Action.CALL)
        game.step(Action.CHECK)

        player_id = game.round.game_pointer
        game.step(Action.BET_HALF_POT)   # half of a 4-chip pot
        self.assertEqual(2, game.round.raised[player_id])

        player_id = game.round.game_pointer
        game.step(Action.RAISE_2X)       # raise to 2 * 2
        self.assertEqual(4, game.round.raised[player_id])

    def test_big_blind_option(self):
        game = make_game()
        game.init_game()
        game.step(Action.CALL)
        # Limped pot: the big blind may check or raise, never fold or bet
        self.assertListEqual(
            [Action.CHECK, Action.ALL_IN, Action.RAISE_2X, Action.RAISE_3X, Action.RAISE_5X],
            game.get_legal_actions())

    def test_raise_needs_true_cost(self):
        game = make_game(init_chips=[10, 10])
        game.init_game()
        game.step(Action.CALL)
        # A 5x raise would cost 8 of the big blind's remaining 8 chips: that is an all-in, not a raise
        self.assertListEqual(
            [Action.CHECK, Action.ALL_IN, Action.RAISE_2X, Action.RAISE_3X],
            game.get_legal_actions())
        game.step(Action.RAISE_3X)       # raise to 6
        self.assertListEqual([2, 6], game.round.raised)
        # The small blind has 8 left and faces 4; every raise would cost at least 10
        self.assertListEqual([Action.FOLD, Action.CALL, Action.ALL_IN], game.get_legal_actions())

    def test_payoffs_1(self):
        game = make_game()
        game.init_game()
        game.step(Action.CALL)
        game.step(Action.RAISE_2X)
        game.step(Action.FOLD)
        self.assertTrue(game.is_over())
        self.assertListEqual([-2, 2], game.get_payoffs())

    def test_payoffs_2(self):
        game = make_game()
        game.init_game()
        game.step(Action.CALL)
        game.step(Action.RAISE_2X)
        game.step(Action.ALL_IN)
        game.step(Action.FOLD)
        self.assertTrue(game.is_over())
        self.assertListEqual([4, -4], game.get_payoffs())

    def test_all_in_to_call(self):
        game = make_game(init_chips=[50, 100])
        game.init_game()
        game.step(Action.CALL)
        game.step(Action.ALL_IN)
        game.step(Action.CALL)
        self.assertTrue(game.is_over())
        # The 100-chip stack's uncalled 50 stays in its in_chips; the judger returns it
        self.assertEqual(150, game.dealer.pot)
        self.assertListEqual([-50, 50], sorted(game.get_payoffs()))  # the showdown winner depends on the cards


if __name__ == '__main__':
    unittest.main()
