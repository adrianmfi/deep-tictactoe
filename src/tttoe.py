#Made with python version 2.7.12
from toimage import toImage
P1 = 'x'
P2 = 'o'
NONE = ' '
'''
Game of Tic tac toe.
To play see main function
'''
class TicTacToe():
	def __init__(self):
		self.board = [[' ' for _ in range(3)] for _ in range(3)]
		self.currentPlayer = P1

	def __str__(self):
		return str(self.board[0]) + '\n' + str(self.board[1]) + '\n'+str(self.board[2])

	def place(self,row,col):
		assert self.board[row][col] == NONE, "board[row][col] is nonempty"
		self.board[row][col] = self.currentPlayer
		if self.currentPlayer == P1:
			self.currentPlayer = P2
		else:
			self.currentPlayer = P1

	def isFinished(self):
		#Check if someone has won
		if self.getWinner() != NONE:
			return True
		#If not, check that there are possible moves
		for i in range(3):
			for j in range(3):
				if self.board[i][j] == NONE:
					return False
		return True

	def getWinner(self):
		#check horizontally and vertically
		for i in range(3):
			if self.board[i][0] != NONE and self.board[i][0] == self.board[i][1] == self.board[i][2]:
				return self.board[i][1]
			if self.board[0][i] != NONE and self.board[0][i] == self.board[1][i] == self.board[2][i]:
				return self.board[1][i]
		#Check diagonally
		if self.board[0][0] != NONE and self.board[0][0] == self.board[1][1] == self.board[2][2]:
			return self.board[1][1]
		if self.board[0][2] != NONE and self.board[0][2] == self.board[1][1] == self.board[2][0]:
			return self.board[1][1]
		return NONE

	def play(self):
		while not self.isFinished():
			print(self)
			print('Player '+ self.currentPlayer + ' turn')
			try:
				row,col = input('Enter row,col: ')
				self.place(row,col)
				toImage(self.board,"board")
			except:
				print ("Placing failed")
			print('\n')
		print(self)
		print ("Winner: " + self.getWinner())

def main():
	game = TicTacToe()
	game.play()

if __name__ == '__main__':
	main()