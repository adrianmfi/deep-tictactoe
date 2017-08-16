import os
import argparse

import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.autograd as autograd

from agent import Agent, Policy1
from tttoe import TicTacToe

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--batches', type=int, default=500, metavar='N',
                    help='How many batches to train for')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='How many samples to train with')

parser.add_argument('--gpb', type=int, default=100, metavar='N',
                    help='How many games to play per batch')


def save_checkpoint(state):
    model_fname = os.path.join(os.path.dirname(
        __file__), 'agent_checkpoints', 'checkpoint.pth.tar')
    torch.save(state, model_fname)


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    policy1 = Policy1()
    policy2 = Policy1()
    optimizer1 = optim.Adam(policy1.parameters(), lr=1e-2)
    optimizer2 = optim.Adam(policy2.parameters(), lr=1e-2)

    agents = [Agent(policy1), Agent(policy1)]
    wins_per_agent = [0, 0]
    for batch_id in range(args.batches):
        saved_batch1 = []
        saved_batch2 = []

        for game_id in range(args.gpb):
            game = TicTacToe()
            saved_game = [[], []]
            while not game.check_finished():
                game_state = [field for row in game.board for field in row]
                moved = False
                while not moved:
                    current_player = game.current_player
                    current_agent = agents[current_player - 1]
                    action = current_agent.select_action(game_state)
                    row, col = current_agent.action_to_row_col(action)
                    try:
                        game.place(row, col)
                    except (ValueError, IndexError):
                        pass
                    else:
                        moved = True
                        saved_game[current_player -
                                   1].append([game_state, action])
            if game.winner is not None:
                wins_per_agent[game.winner - 1] += 1
            for state, action in saved_game[0]:
                if game.winner == 1:
                    saved_batch1.append([state, action, 1])
                elif game.winner == 2:
                    saved_batch1.append([state, action, -1])
                else:
                    saved_batch1.append([state, action, 0])
            for state, action in saved_game[1]:
                if game.winner == 1:
                    saved_batch2.append([state, action, -1])
                elif game.winner == 2:
                    saved_batch2.append([state, action, 1])
                else:
                    saved_batch2.append([state, action, 0])
        policy1.train()
        for state, action, reward in saved_batch1:
            reward_tensor = torch.FloatTensor(1, 1)
            reward_tensor[0] = reward
            #reward = torch.unsqueeze(reward, 0)
           # print(reward_tensor)
            action.reinforce(reward_tensor)
        optimizer1.zero_grad()
        np_batch = np.array(saved_batch1)
        autograd.backward(np_batch[:, 1], [None for _ in np_batch[:, 1]])
        optimizer1.step()
        state1 = {
            'state_dict': policy1.state_dict(),
        }
        save_checkpoint(state1)
        print(wins_per_agent)


if __name__ == '__main__':
    main()
