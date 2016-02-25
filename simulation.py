import random
import os

numCells = 20
simulation_folder = "simulation/"

def in_frame(row,col):
	return 0<=row<numCells and 0<=col<=numCells

def flush_raster(raster,file_name):
	with open(file_name,'w') as f:
		f.write("ncols\t%d\n" % (numCells))
		f.write("nrows\t%d\n" % (numCells))
		f.write("xllcorner\t0\n")
		f.write("yllcorner\t0\n")
		f.write("cellsize\t1\n")
		f.write("NODATA_value\t-99\n")
		for i in range(numCells):
			row = i
			for col in range(numCells):
				f.write(str(raster[row][col])+'\t')
			f.write('\n')

def flush_groundtruth(matrix,file_name):
	with open(file_name,'w') as f:
		for row in matrix:
			out_put = ''
			for cell in row:
				if out_put:
					out_put += ','
				out_put += str(cell)
			out_put += '\n'
			f.write(out_put)

def rowcol_to_index(row, col):
	return row * numCells + col

def index_to_rowcol(i):
	row = i/numCells
	col = i%numCells
	return row, col 


def flock_raster(num_objects,ocenter,dcenter):
	start_raster = [[0 for i in range(numCells)] for j in range(numCells)]
	end_raster = [[0 for i in range(numCells)] for j in range(numCells)]
	ODValues = [[0 for i in range(numCells*numCells)] for j in range(numCells*numCells)]
	for j in range(num_objects):
		s, e = random_movement(ocenter,dcenter)
		start_raster[s[0]][s[1]] += 1
		end_raster[e[0]][e[1]] += 1
		ODValues[rowcol_to_index(*s)][rowcol_to_index(*e)] += 1
	return start_raster, end_raster, ODValues

def folder_check(num_objects):
	folder_name = simulation_folder + str(num_objects)
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)
		return folder_name
	return False

def random_movement(ocenter, dcenter):
	return random_scatter(ocenter),random_scatter(dcenter)

def random_scatter(origin):
	row, col = origin
	row_diff = random.randint(-2,2)
	col_diff = random.randint(-2,2)
	cur_row = row + row_diff
	cur_col = col + col_diff
	while not in_frame(row,col):
		row_diff = random.randint(-2,2)
		col_diff = random.randint(-2,2)
		cur_row = row + row_diff
		cur_col = col + col_diff
	return cur_row, cur_col

def simulate_flock(num_objects):
	start_raster, end_raster, ODValues = flock_raster(num_objects, (2,2), (7,7))
	folder_name =  folder_check(num_objects)
	pcells = [[0 for i in range(numCells)] for j in range(numCells)]
	if folder_name:
		flush_raster(start_raster,folder_name+'/start.asc')
		flush_raster(end_raster,folder_name+'/end.asc')
		flush_raster(pcells,folder_name+'/pcells.asc')
		flush_groundtruth(ODValues,folder_name+'/groudtruth.csv')

def simulate_twoway_movement(num_objects):
	pass

def twoway_movement(num_objects, row1, row2):
	

if __name__ == '__main__':
	for i in range(5,25,5):
		simulate_flock(i)

