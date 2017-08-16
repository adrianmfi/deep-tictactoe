#!/usr/bin/env python3
'''
Game of Tic tac toe.
'''


def to_string(field):
    if field == 0:
        return ' '
    elif field == 1:
        return 'x'
    elif field == 2:
        return 'o'
    else:
        raise ValueError('Wrong field, expected 0,1 or 2')


class TicTacToe():
    def __init__(self):
        self.board = [[0 for col in range(3)] for row in range(3)]
        self.current_player = 1
        self.finished = False
        self.winner = None

    def __str__(self):
        board = [[to_string(field) for field in row]
                 for row in self.board]
        return str(board[0]) + '\n' + str(board[1]) + '\n' + str(board[2])

    def place(self, row, col):
        if self.board[row][col] != 0:
            raise ValueError(
                "already placed at board[{}][{}]".format(row, col))

        self.board[row][col] = self.current_player

        if self.current_player == 1:
            self.current_player = 2
        else:
            self.current_player = 1

    def check_finished(self):
        if self.finished:
            return True
        if self.check_winner() is not None:
            return True

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return False
        self.finished = True
        return True

    def check_winner(self):
        if self.winner is not None:
            return self.winner
        elif self.finished:
            return None

        # check horizontally and vertically
        for i in range(3):
            if self.board[i][0] != 0 and self.board[i][0] == self.board[i][1] == self.board[i][2]:
                self.winner = self.board[i][1]
                self.finished = True
                return self.winner
            if self.board[0][i] != 0 and self.board[0][i] == self.board[1][i] == self.board[2][i]:
                self.winner = self.board[1][i]
                self.finished = True
                return self.winner
        # Check diagonally
        if self.board[0][0] != 0 and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            self.winner = self.board[0][0]
            self.finished = True
            return self.winner
        if self.board[0][2] != 0 and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            self.winner = self.board[0][2]
            self.finished = True
            return self.winner
        return None


def play_from_command_line():
    game = TicTacToe()

    while not game.check_finished():
        print(game)
        print('Player {}\'s turn'.format(game.current_player))
        try:
            row, col = map(int, input('Enter row,col: ').split(','))
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
    play_from_command_line()


if __name__ == '__main__':
    main()
