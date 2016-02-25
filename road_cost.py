import networkx as nx 
numRows = 3
numCols = 10
MAXVALUE = 9999

def index_to_rowcol(i):
	row = i/numCols
	col = i%numCols
	return row, col 

	



if __name__ == '__main__':
	with open('roads/costlist.txt','w') as f:
		size = numCols*numRows
		for i in range(size):
			orow, ocol = index_to_rowcol(i)
			for j in range(size):
				cost = 0
				drow, dcol = index_to_rowcol(j)
				if orow == drow:
					if orow == 0:
						cost = dcol - ocol if dcol-ocol>=0 else MAXVALUE
					elif orow == 2:
						cost = ocol - dcol if ocol - dcol >= 0 else MAXVALUE
				else:
					cost = MAXVALUE
				f.write('''%d,%d,%d,%d,%d\n''' % (orow,ocol,drow,dcol,cost))






