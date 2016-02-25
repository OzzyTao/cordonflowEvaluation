import psycopg2
import csv
import sys
import os

id_list = []
tablename = 'ecourier'

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

def make_datetime(date_str,time_str):
	date_int = map(int,date_str.split('-'))
	time_int = map(int,time_str.split(':'))
	return datetime(*(date_int+time_int))


def fetch_id(typename, vehicleid):
	if typename + vehicleid in id_list:
		return id_list.index(typename+vehicleid)
	else:
		id_list.append(typename+vehicleid)
		return len(id_list)-1

def db_create_table(conn):
	sql_str = '''create table if not exists {table_name}(
		id serial PRIMARY KEY,
		v_id integer,
		longitude double precision,
		latitude double precision,
		time timestamp without time zone,
		geom geometry(Point, 4326)
		);'''
	cur = conn.cursor()
	cur.execute(sql_str.format(table_name=tablename))
	conn.commit()
	cur.close()

def populate_table(conn,tab_dict):
	sql_str = '''INSERT INTO {table_name} (v_id, longitude, latitude, time)
	VALUES ({v_id},{lon},{lat}, timestamp '{time}');'''
	cur = conn.cursor()
	try:
		# print tab_dict
		sql = sql_str.format(table_name=tablename,v_id=fetch_id(tab_dict['type'],tab_dict['vehicleid']),lon=tab_dict['longitude'],lat=tab_dict['latitude'],time=tab_dict['timestamp'])
		# print sql
		cur.execute(sql)
	except Exception, e:
		print e 
		conn.rollback()
	else:
		conn.commit()
	finally:
		cur.close()

def table_to_db(filename,conn):
	table = []
	with open(filename) as f:
		reader = csv.DictReader(f,delimiter='\t')
		for row in reader:
			populate_table(conn,row)

if __name__ == '__main__':
	if len(sys.argv) > 1:
		directory = sys.argv[1]
		conn = db_connection()
		db_create_table(conn)
		table_to_db(directory,conn)
		conn.close()
		print len(id_list) + ' vehicles in total.'
	else:
		print "need dir_name as parameter"

