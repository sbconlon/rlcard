# ====================== #
#  Train a GT-CFR Agent  #
# ====================== #

# Debug + Testing imports
import cProfile
import ipdb

# External imports
import numpy as np

# Internal imports
from print_profiler_stats import print_rlcard_function_stats
import rlcard
from rlcard.agents.gt_cfr_agent.gt_cfr_agent import GTCFRAgent, GTCFRSolver
from rlcard.games.base import Card
from rlcard.games.nolimitholdem.round import Action
from rlcard.utils.utils import init_standard_deck




def main():
    #
    # Training parameters
    #
    #num_episodes = 1

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
                'dealer_id': 0 # Fix player 1 as the dealer
    }
    env = rlcard.make('no-limit-holdem', config=config)

    #
    # Initialize the GT-CFR Agent
    #
    # NOTE - query queue is owned by the CFVN
    #        and both the agent's solver and the query solver
    #        share the same CFVN.
    #
    agent = GTCFRAgent(
        env,
        prob_query_solve=0.003,
        n_expansions=3,
        n_expansions_per_regret_update=0.003, # 333 updates per expansion
        n_query_solvers=0, 
        n_trainers=0
    )
    query_solver = GTCFRSolver(
        input_cfvn=agent.solver.cfvn,
        n_expansions=3,
        n_expansions_per_regret_update=0.001, 
        prob_query_solve=0.001
    )
    query_solver.cfvn.init_replay_buffer()
    
    #
    # Load truth data
    #
    true_strat = np.load('strat-fs-7s8s8hThJh.npy')
    true_values = np.load('values-fs-7s8s8hThJh.npy')

    #
    # Patch true strat
    #
    # NOTE - fix this in the future
    #
    tril_idxs = np.tril_indices(52)
    true_strat[:, tril_idxs[0], tril_idxs[1]] == 0.

    #
    # Int to card map
    #
    int_to_card = {c.to_int(): c for c in init_standard_deck()}

    #
    # Run self-play until the predicted strategy and values converge to
    # the true values.
    #
    epsilon = 1e-3
    niters = 0
    while True:
        #
        # Test the network quality
        #
        print()
        print('=====================================================================')
        print(f'Iteration #{niters}')
        agent.solver.reset()
        env.reset()
        print()
        print('Starting test solve')
        pred_strat, pred_values = agent.solver.solve(env.game)
        mse_strat, mse_values = np.square(pred_strat - true_strat), np.square(pred_values - true_values)
        print(f'Strategy MSE = {mse_strat.mean()}')
        print(f'Values MSE   = {mse_values.mean()}')
        print()
        worst_strat_hand = np.unravel_index(np.argmax(np.sum(mse_strat, axis=0)), mse_strat.shape[1:])
        worst_values_hand = np.unravel_index(np.argmax(np.sum(mse_values, axis=0)), mse_values.shape[1:])
        print(f'Worst Strategy')
        print(f'{[str(int_to_card[worst_strat_hand[0]]), str(int_to_card[worst_strat_hand[1]])]}')
        print(f'Expected = {true_strat[:, worst_strat_hand[0], worst_strat_hand[1]]}')
        print(f'Got      = {pred_strat[:, worst_strat_hand[0], worst_strat_hand[1]]}')
        print()
        print(f'Worst Values')
        print(f'{[str(int_to_card[worst_values_hand[0]]), str(int_to_card[worst_values_hand[1]])]}')
        print(f'Expected = {true_values[:, worst_values_hand[0], worst_values_hand[1]]}')
        print(f'Got      = {pred_values[:, worst_values_hand[0], worst_values_hand[1]]}')
        print()
        if (np.all(np.isclose(pred_strat, true_strat, atol=epsilon)) and 
            np.all(np.isclose(pred_values, true_values, atol=epsilon))):
            print('SUCCESS')
            break
        #
        # Run a self play episode
        #
        agent.self_play()
        #
        # The query queue should now have a bunch of entries on it.
        #
        # Process all of these querries
        #
        while not agent.solver.cfvn.query_queue.empty():
            print('-')
            print(f'Query queue size = {agent.solver.cfvn.query_queue.qsize()}')
            # Get a query off the queue
            query = agent.solver.cfvn.query_queue.get()
            # Solve it with the query solver
            print('Starting query solve')
            print(f'decision point trajectory: {query[0].trajectory + query[3]}')
            target_strat, target_values = query_solver.solve(*query)
            print('Finished query solve')
            # Get the input CFVN vector for the given query
            node = query_solver.decision_point
            input_vect = query_solver.cfvn.to_vect(node.game, node.player_ranges, False)
            # Reshape the target strategy and values to match the network output dimensions
            triu_indices = np.triu_indices(52, k=1)
            target_strat = target_strat[:, triu_indices[0], triu_indices[1]]
            target_values = target_values[:, triu_indices[0], triu_indices[1]]
            # Pad the target strategy with zeros for invalid actions
            target_strat_padded = np.zeros((query_solver.cfvn.num_actions, 1326))
            all_actions = node.game.round.get_all_actions()
            valid_action_idxs = [all_actions.index(action) for action in node.actions]
            for idx, padded_idx in enumerate(valid_action_idxs):
                target_strat_padded[padded_idx] = target_strat[idx]
            # Place the solved query onto the replay buffer queue
            query_solver.cfvn.add_to_replay_buffer((input_vect, target_strat_padded, target_values))
            # Reset the solver game tree after solving
            query_solver.reset()
        print('-')
        print()
        #
        # Now there should be a set of training targets on the replay buffer,
        # perform batch updates until the replay buffer is exhausted.
        #
        n_updates = 0
        while query_solver.cfvn.replay_buffer.size() >= query_solver.cfvn.batch_size:
            print('-')
            print(f'Replay buffer size= {query_solver.cfvn.replay_buffer.size()}')
            query_solver.cfvn.train(0, niters=1) # perform a single batch update
            n_updates += 1
            print(f'Batch update {n_updates}')
        niters += 1
    print('--> SUCCESSFUL CONVERGENCE!!!')


if __name__ == '__main__': # Needed for multiprocessing
    main()
