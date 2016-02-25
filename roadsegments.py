import psycopg2

upperRoad = [22761,18271]
lowerRoad = [1484, 22763]

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


def create_table(conn):
	cur  = conn.cursor()
	query = '''create table if not exists keyroads (
		id    serial PRIMARY KEY,
		direction varchar(6)
		);'''
	cur.execute(query)
	query = '''select AddGeometryColumn( 'keyroads', 'geom', 4326, 'LINESTRING', 2)'''
	cur.execute(query)
	conn.commit()

def add_road(start,end,direction,conn):
	cur = conn.cursor()
	query = '''insert into keyroads (direction,geom) values ('%s',(select st_linemerge(st_union(hh_2po_4pgr.geom_way)) from hh_2po_4pgr, 
		(select seq, id1 as node, id2 as edge, cost from pgr_dijkstra('select id, source, target, cost, reverse_cost from hh_2po_4pgr',%s,%s,true,true)) as roads
		where hh_2po_4pgr.id = roads.edge and roads.edge != -1) );'''
	cur.execute(query % (direction,start,end))
	# rid = cur.fetchone()[0]
	# query = '''update keyroads set direction = '%s' where id = %s ;'''
	# cur.execute(query % (direction,rid))
	conn.commit()

if __name__ == '__main__':
	conn = db_connection()
	# create_table(conn)
	add_road(upperRoad[0],upperRoad[1],"right",conn)
	add_road(lowerRoad[0],lowerRoad[1],"left",conn)

