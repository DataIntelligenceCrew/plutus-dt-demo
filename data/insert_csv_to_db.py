import argparse
import os
import pandas as pd
import psycopg2
import numpy as np
from psycopg2 import sql
from psycopg2.extensions import register_adapter, AsIs

# Register numpy.int64 type with psycopg2
def addapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)
register_adapter(np.int64, addapt_numpy_int64)

# Database connection string
DB_CONNECTION_STRING = "dbname=dtdemo user=jchang38"

def main(directory):
    # Create a database connection
    conn = psycopg2.connect(DB_CONNECTION_STRING)
    cur = conn.cursor()

    # Loop through all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            # Read the CSV file into a DataFrame
            df = pd.read_csv(os.path.join(directory, filename))

            # Remove the .csv from the filename to get the table name
            table_name = filename[:-4]

            # Delete the table if it already exists
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
            conn.commit()

            # Create a new table with columns based on the CSV file
            column_names = df.columns
            create_table_query = sql.SQL("CREATE TABLE {} ({});").format(
                sql.Identifier(table_name),
                sql.SQL(',').join(sql.SQL("{} float").format(sql.Identifier(column_name)) for column_name in column_names)
            )
            cur.execute(create_table_query)
            conn.commit()

            # Insert the data frame into the table
            df_columns = list(df)
            columns = ",".join(df_columns)
            values = "VALUES({})".format(",".join(["%s" for _ in df_columns]))
            insert_stmt = "INSERT INTO {} ({}) {}".format(table_name, columns, values)
            cur.executemany(sql.SQL(insert_stmt), df.values)
            conn.commit()

    # Close the database connection
    cur.close()
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('directory', type=str, help='Directory to parse CSV files from')

    args = parser.parse_args()
    main(args.directory)