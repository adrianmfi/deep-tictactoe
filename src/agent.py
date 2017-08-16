import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Policy1(nn.Module):
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


def policy1_pretrained():
    p = Policy1()
    rel_dir = os.path.dirname(__file__)
    checkpoint_path = os.path.join(
        rel_dir, 'agent_checkpoints', 'checkpoint.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    p.load_state_dict(checkpoint['state_dict'])
    return p


class Agent():
    def __init__(self, policy):
        self.policy = policy

    def select_action(self, state):
        self.policy.eval()
        state = torch.FloatTensor(state)
        state = torch.unsqueeze(state, 0)
        state = Variable(state)
        probs = self.policy(state)
        action = probs.multinomial()
        return action

    def action_to_row_col(self, action):
        row = action.data.numpy()[0][0] // 3
        col = action.data.numpy()[0][0] % 3
        return row, col


def play_vs_agent():
    from tttoe import TicTacToe
    game = TicTacToe()
    policy = policy1_pretrained()
    agent = Agent(policy)

    while not game.check_finished():
        print(game)
        print('Player {}\'s turn'.format(game.current_player))
        try:
            if game.current_player == 2:
                row, col = map(int, input('Enter row,col: ').split(','))
                game.place(row, col)
            else:
                game_state = [field for row in game.board for field in row]
                action = agent.select_action(game_state)
                row, col = agent.action_to_row_col(action)
                game.place(row, col)
        except (ValueError, IndexError) as e:
            print(e)
            continue
        print()
    print(game)
    if game.winner is not None:
        print("Winner:", game.winner)
    else:
        print("Draw")


def main():
    play_vs_agent()


if __name__ == '__main__':
    main()
