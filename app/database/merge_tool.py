import sqlite3
import sys
import os

def merge_databases(input_db, output_db):
    # Connect to both databases
    input_conn = sqlite3.connect(input_db)
    output_conn = sqlite3.connect(output_db)
    
    input_cursor = input_conn.cursor()
    output_cursor = output_conn.cursor()

    # Get the list of tables from the input database
    input_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = input_cursor.fetchall()

    merge_summary = {}

    # Iterate through each table
    for table in tables:
        table_name = table[0]
        merge_summary[table_name] = {'total': 0, 'inserted': 0, 'duplicates': 0}
        
        # Get all rows from the input table
        input_cursor.execute(f"SELECT * FROM {table_name}")
        rows = input_cursor.fetchall()

        # Get column names
        input_cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in input_cursor.fetchall()]

        # Prepare the INSERT statement
        placeholders = ','.join(['?' for _ in columns])
        insert_query = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

        # Insert rows into the output database
        for row in rows:
            merge_summary[table_name]['total'] += 1
            try:
                output_cursor.execute(insert_query, row)
                merge_summary[table_name]['inserted'] += 1
            except sqlite3.IntegrityError:
                merge_summary[table_name]['duplicates'] += 1

    # Commit changes and close connections
    output_conn.commit()
    input_conn.close()
    output_conn.close()

    return merge_summary

def main():
    if len(sys.argv) != 3:
        print("Usage: python merge_tool.py <input_db> <output_db>")
        sys.exit(1)

    input_db = sys.argv[1]
    output_db = sys.argv[2]

    if not os.path.exists(input_db):
        print(f"Error: Input database '{input_db}' does not exist.")
        sys.exit(1)

    if not os.path.exists(output_db):
        print(f"Error: Output database '{output_db}' does not exist.")
        sys.exit(1)

    try:
        merge_summary = merge_databases(input_db, output_db)
        print(f"Merge operation completed. Summary:")
        for table, stats in merge_summary.items():
            print(f"Table '{table}':")
            print(f"  Total rows: {stats['total']}")
            print(f"  Inserted rows: {stats['inserted']}")
            print(f"  Duplicate rows: {stats['duplicates']}")
            print()
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()