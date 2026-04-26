import psycopg2
import sys
import socket

# Force IPv4
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# Connection string
conn_string = "postgresql://postgres:Flowera-099@db.yrkfcjksnuodzwidphjq.supabase.co:5432/postgres"

try:
    print("Connecting to Supabase...")
    conn = psycopg2.connect(conn_string)
    conn.autocommit = True
    cursor = conn.cursor()

    print("Reading SQL file...")
    with open('normalized_inventory_db.sql', 'r', encoding='utf-8') as f:
        sql_content = f.read()

    print("Executing SQL restore...")
    cursor.execute(sql_content)

    print("✓ Database restored successfully!")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
