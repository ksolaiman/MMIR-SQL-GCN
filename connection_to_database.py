import psycopg2
import json

# Load DB config
with open('config.json', 'r') as f:
    config = json.load(f)

which_db = config.get('db_host')
db_settings = config.get(which_db)

print(f"Connecting to {which_db} database at {db_settings['host']}:{db_settings['port']}")

# Connect using the selected settings
# conn = psycopg2.connect(
#     database=db_settings['database'],
#     user=db_settings['user'],
#     password=db_settings['password'],
#     host=db_settings['host'],
#     port=db_settings['port']
# )
# even shorter version below
conn = psycopg2.connect(**db_settings)
dbcur = conn.cursor()
print("connection successful")