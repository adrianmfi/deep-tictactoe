import argparse
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

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


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(9, 128)
        self.affine2 = nn.Linear(128, 9)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores)


class Agent():
    def __init__(self, policy, optimizer):
        self.policy = policy
        self.optimizer = optimizer

    def select_action(self, state):
        self.policy.eval()
        state = torch.FloatTensor(state)
        state = torch.unsqueeze(state, 0)
        state = Variable(state)
        probs = self.policy(state)
        action = probs.multinomial()
        return action


def main():
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    policy1 = Policy()
    policy2 = Policy()
    optimizer1 = optim.Adam(policy1.parameters(), lr=1e-2)
    optimizer2 = optim.Adam(policy2.parameters(), lr=1e-2)

    agents = [Agent(policy1, optimizer1), Agent(policy1, optimizer2)]
    winners = [0, 0]
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
                    row = action.data.numpy()[0][0] // 3
                    col = action.data.numpy()[0][0] % 3
                    try:
                        game.place(row, col)
                    except (ValueError, IndexError):
                        pass
                    else:
                        moved = True
                        saved_game[current_player -
                                   1].append([game_state, action])
            if game.winner is not None:
                winners[game.winner - 1] += 1
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

    print(winners)


'''
def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(Variable(state))
    action = probs.multinomial()
    policy.saved_actions.append(action)
    return action.data


def finish_episode():
    R = 0
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / \
        (rewards.std() + np.finfo(np.float32).eps)
    for action, r in zip(policy.saved_actions, rewards):
        action.reinforce(r)
    optimizer.zero_grad()
    autograd.backward(policy.saved_actions, [
                      None for _ in policy.saved_actions])
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_actions[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action[0, 0])
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > 200:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

'''
if __name__ == '__main__':
    main()
