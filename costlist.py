import networkx as nx 
import psycopg2

# boundary_ul = (-0.1348, 51.5544)
boundary_ul = (-0.3549,51.5749)

boundary_size = 0.2
numCells = 40
maxspeed = 27.7
tablename = 'ecourier'
# tests_frames = [[[10,0],[10,1]]
# , [[10,0],[10,2]]
# , [[10,0],[10,3]]
# , [[10,0],[10,4]]]
tests_frames = [[[10,0,10],[10,0,20]],[[10,0,10],[10,0,40]]]



def folder_check(time_frame):
	folder_name = 'test40/raster'
	for time in time_frame:
		for timepart in time:
			folder_name += str(timepart)
	return folder_name

def manhanton(i, j):
	return abs(i[0]-j[0])+abs(i[1]-j[1])

def diagonalmanhantton(i,j):
	return max([abs(i[0]-j[0]),abs(i[1]-j[1])])

def db_connection():
	main_dt = {
	'db' : 'gis',
	'ip' : 'localhost',
	'port' : '5432',
	'password' : 'peach',
	'user' : 'postgres'
	}
	conn = psycopg2.connect('''host={ip} dbname={db} user={user} password={password} port={port}'''.format(**main_dt))
	return conn

def cell_connected(cell1,cell2,conn):
	step = boundary_size/numCells
	if cell1[0] == cell2[0]:
		latmin, latmax = boundary_ul[1] - (cell1[0]+1)*step , boundary_ul[1] - cell1[0]*step
		longitude = boundary_ul[0]+max(cell1[1],cell2[1])*step
		point1 = [longitude,latmin]
		point2 = [longitude,latmax]
	else:
		longmin, longmax = boundary_ul[0]+cell1[1]*step, boundary_ul[0]+(cell1[1]+1)*step
		latitude = boundary_ul[1] - max(cell1[0],cell2[0])*step
		point1 = [longmin, latitude]
		point2 = [longmax, latitude]

	# clazz <= 22
	with conn.cursor() as cur:
		cur.execute('''select * from hh_2po_4pgr where st_intersects(st_makeline(
			ST_SetSRID(ST_MakePoint(%s, %s),4326),
			ST_SetSRID(ST_MakePoint(%s, %s),4326)
			),geom_way)''' % (point1[0],point1[1],point2[0],point2[1]))
		if cur.fetchall():
			return True
		else:
			return False

def rowcol_to_index(row, col):
	return row * numCells + col

def index_to_rowcol(i):
	row = i/numCells
	col = i%numCells
	return row, col 

def diff_in_min(t1,t2):
	if len(t1) == 2:
		return (t2[0]-t1[0]) * 60 + (t2[1]-t1[1])
	else:
		return (t2[0]-t1[0]) * 60 + (t2[1]-t1[1]) + (t2[2]-t1[2])/60.0

def cal_cost_with_maxdist(t):
	conn = db_connection()	
	G = nx.grid_2d_graph(numCells,numCells)
	for e in G.edges_iter():
		if cell_connected(e[0],e[1],conn):
			G[e[0]][e[1]]['weight'] = 1
		else:
			G[e[0]][e[1]]['weight'] = 10000
			print e 

	maxdist = round(maxspeed * 60 / (float(boundary_size) * 111 * 1000 / numCells)  * diff_in_min(*t))


	matrix = [[0 for i in range(numCells*numCells)] for j in range(numCells*numCells)]
	for i in G.nodes_iter():
		row_index = rowcol_to_index(*i)
		for j in G.nodes_iter():
			col_index = rowcol_to_index(*j)
			if row_index<col_index:
				if manhanton(i,j) <= maxdist: 
					matrix[row_index][col_index] = nx.shortest_path_length(G,i,j,'weight')
				else:
					matrix[row_index][col_index] = 10000
		print row_index
	
	result_folder = folder_check(t)
	with open(result_folder+'/costlist.txt','w') as f:
		for i in range(numCells*numCells):
			orowcol = index_to_rowcol(i)
			for j in range(numCells*numCells):
				drowcol = index_to_rowcol(j)
				if i<j:
					cost = matrix[i][j]
				else:
					cost = matrix[j][i]
				f.write('''%d,%d,%d,%d,%d\n''' % (orowcol[0],orowcol[1],drowcol[0],drowcol[1],cost))
	conn.close()

if __name__ == '__main__':
	for t in tests_frames:
		cal_cost_with_maxdist(t)



