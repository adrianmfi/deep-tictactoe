#!/usr/bin/env python3
'''
Game of Tic tac toe.
'''

from image.toimage import to_image
P1 = 'x'
P2 = 'o'
EMPTY_FIELD = ' '


class TicTacToe():
    def __init__(self):
        self.board = [[' ' for col in range(3)] for row in range(3)]
        self.current_player = P1
        self.finished = False
        self.winner = None

    def __str__(self):
        return str(self.board[0]) + '\n' + str(self.board[1]) + '\n' + str(self.board[2])

    def place(self, row, col):
        if self.board[row][col] != EMPTY_FIELD:
            raise ValueError(
                "already placed at board[{}][{}]".format(row, col))

        self.board[row][col] = self.current_player

        if self.current_player == P1:
            self.current_player = P2
        else:
            self.current_player = P1

    def check_finished(self):
        if self.finished:
            return True
        if self.check_winner() is not None:
            return True

        for i in range(3):
            for j in range(3):
                if self.board[i][j] == EMPTY_FIELD:
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
            if self.board[i][0] != EMPTY_FIELD and self.board[i][0] == self.board[i][1] == self.board[i][2]:
                self.winner = self.board[i][1]
                self.finished = True
                return self.board[i][1]
            if self.board[0][i] != EMPTY_FIELD and self.board[0][i] == self.board[1][i] == self.board[2][i]:
                self.winner = self.board[i][1]
                self.finished = True
                return self.board[1][i]
        # Check diagonally
        if self.board[0][0] != EMPTY_FIELD and self.board[0][0] == self.board[1][1] == self.board[2][2]:
            self.winner = self.board[0][0]
            self.finished = True
            return self.winner
        if self.board[0][2] != EMPTY_FIELD and self.board[0][2] == self.board[1][1] == self.board[2][0]:
            self.winner = self.board[0][2]
            self.finished = True
            return self.winner
        return None


def play_with_command_line():
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
        to_image(game.board, "board")
        print()
    print(game)
    if game.winner is not None:
        print("Winner: " + game.winner)
    else:
        print("Draw")


def main():
    play_with_command_line()


if __name__ == '__main__':
    main()
