import requests
import re

SUPABASE_URL = "https://yrkfcjksnuodzwidphjq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inlya2ZjamtzbnVvZHp3aWRwaGpxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDU2NzI0NzksImV4cCI6MjA2MTI0ODQ3OX0.L60pGO9-tYfB4Z-r8pXMNg_d7W2TMTX"

print("Reading SQL file...")
with open('normalized_inventory_db.sql', 'r', encoding='utf-8') as f:
    sql_content = f.read()

# Split into individual statements
statements = [s.strip() for s in re.split(r';\s*\n', sql_content) if s.strip() and not s.strip().startswith('--')]

print(f"Found {len(statements)} SQL statements")
print("Executing via Supabase REST API...")

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json"
}

failed = 0
for i, stmt in enumerate(statements, 1):
    if i % 100 == 0:
        print(f"Progress: {i}/{len(statements)}")

    try:
        response = requests.post(
            f"{SUPABASE_URL}/rest/v1/rpc/exec_sql",
            headers=headers,
            json={"query": stmt}
        )
        if response.status_code >= 400:
            failed += 1
            if failed <= 5:  # Show first 5 errors
                print(f"Error on statement {i}: {response.text[:100]}")
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"Exception on statement {i}: {str(e)[:100]}")

print(f"\nCompleted: {len(statements) - failed} succeeded, {failed} failed")
