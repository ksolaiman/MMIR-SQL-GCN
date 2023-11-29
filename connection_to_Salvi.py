import psycopg2

# set-up a postgres connection
conn = psycopg2.connect(database='Salvi', user='Salvi',password='sholock',
        host='localhost', port=5433)
dbcur = conn.cursor()
print("connection successful")
