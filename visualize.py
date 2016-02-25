import csv
numCells = 40
def rowcol_to_index(row, col):
	return row * numCells + col

def index_to_rowcol(i):
	row = i/numCells
	col = i%numCells
	return row, col 

def manhanton(i, j):
	return abs(i[0]-j[0])+abs(i[1]-j[1])

def make_ODmatrix(file_name):
	size = numCells * numCells
	matrix = [[0 for i in range(size)] for j in range(size)]
	with open(file_name,'rb') as f:
		table = csv.reader(f)
		for row in table:
			from_index = rowcol_to_index(int(row[0]),int(row[1]))
			to_index = rowcol_to_index(int(row[2]),int(row[3]))
			matrix[from_index][to_index] += int(float(row[4]))
	return matrix

def read_groundtruth(file_name):
	matrix = []
	with open(file_name,'r') as f:
		table = csv.reader(f)
		for row in table:
			matrix.append([int(x) for x in row])
	return matrix

def check_outflow(targeted_cell,od_matrix):
	oindex = rowcol_to_index(*targeted_cell)
	for i in range(len(od_matrix)):
		cell = od_matrix[oindex][i]
		if cell != 0:
			print str(cell) +"-->" + str(index_to_rowcol(i))


def freq(matrix):
	freq_list = [0 for i in range(10)]
	size = len(matrix)
	for i in range(size):
		o = index_to_rowcol(i)
		for j in range(size):
			d = index_to_rowcol(j)
			step = manhanton(o,d)
			if step < 10:
				freq_list[step] += matrix[i][j]
	return freq_list


if __name__ == '__main__':
	targeted_cell = [14,25]
	predicted = "raster1001010020/odMatrix_without.csv"
	groundtruth = "raster1001010020/groundtruth.csv"
	print "According to Prediction: "
	check_outflow(targeted_cell,make_ODmatrix(predicted))
	print "According to ground truth: "
	g_matrix = read_groundtruth(groundtruth)
	check_outflow(targeted_cell,g_matrix)
	print "Ground truth frequency:"
	print freq(g_matrix)



