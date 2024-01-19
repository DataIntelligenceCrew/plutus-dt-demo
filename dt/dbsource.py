import psycopg2
import pandas as pd
import numpy as np
import numpy.typing as npt
import typing as tp
import const
import utils

POSTGRES_DEFAULT_PORT = 5432

def get_query_result(task: str, idx: int) -> pd.DataFrame:
    """
    returns:
        The result of running the query in raw format. 
    """
    conn, cur = get_conn_cursor(task)
    cur.execute(construct_query_str(task, idx))
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=col_names)
    cur.close()
    conn.close()
    return utils.recode_db_to_raw(df, task)

def get_train(task: str) -> pd.DataFrame:
    conn, cur = get_conn_cursor(task)
    cur.execute(construct_train_query_str(task))
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=col_names)
    cur.close()
    conn.close()
    return utils.recode_db_to_raw(df, task)

def get_test(task: str) -> pd.DataFrame:
    conn, cur = get_conn_cursor(task)
    cur.execute(construct_test_query_str(task))
    rows = cur.fetchall()
    col_names = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=col_names)
    cur.close()
    conn.close()
    return utils.recode_db_to_raw(df, task)

def get_all_features(task: str) -> tp.List[str]:
    return get_x_features(task) + [const.Y_COLUMN[task]]

def get_x_features(task:str) -> tp.List[str]:
    return const.CATEGORICAL_FEATURES[task] + const.NUMERIC_FEATURES[task]

def construct_query_str(task: str, idx: int) -> str:
    query = "SELECT "
    # Select the correct features used for training
    x_features = get_x_features(task)
    for i, feature in enumerate(x_features):
        query += feature + ","
    query += const.Y_COLUMN[task]
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    # Table sample a small % of data
    query += " TABLESAMPLE SYSTEM(" + str(const.SOURCES[task]["sample_percent"]) + ")"
    # Select the correct source
    query += " WHERE " + const.SOURCES[task]["pivot"] + " = " \
        + str(const.SOURCES[task]["pivot_values"][idx])
    # Exclude train, test sets
    query += " AND train = false AND test = false "
    # Remove null y values
    query += " AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    # Randomly shuffle the tablesample to reduce bias
    query += " ORDER BY random() "
    query += " LIMIT " + str(const.SOURCES[task]["batch"])
    return query + ";"

def construct_count_query_str(task: str, idx: int, slice_: npt.NDArray) -> str:
    query = "SELECT COUNT(*)"
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    query += " WHERE "
    i = 0
    # Categorical bucket conditions
    cat_map = const.REVERSE_CATEGORICAL_MAPPINGS[task]
    for cat in const.CATEGORICAL_FEATURES[task]:
        if slice_[i] is not None:
            query += cat + " = '" + str(cat_map[cat][slice_[i]]) + "' AND "
        i += 1
    buckets = const.NUMERIC_BIN_BORDERS[task]
    # Numeric bucket conditions
    for num in const.NUMERIC_FEATURES[task]:
        if slice_[i] is not None:
            query += num + " >= " + str(buckets[num][slice_[i]]) + " AND "
            query += num + " <= " + str(buckets[num][slice_[i] + 1]) + " AND "
        i += 1
    # Exclude train, test sets
    query += " train = false AND test = false AND "
    # Select the correct source
    query += const.SOURCES[task]["pivot"] + " = " \
        + str(const.SOURCES[task]["pivot_values"][idx])
    # Remove null y values
    query += " AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + ";"

def construct_train_query_str(task: str) -> str:
    query = "SELECT "
    x_features = get_x_features(task)
    for i, feature in enumerate(x_features):
        query += feature + ","
    query += const.Y_COLUMN[task]
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    query += " WHERE train = true AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + ";"

def construct_test_query_str(task: str) -> str:
    query = "SELECT "
    x_features = get_x_features(task)
    for i, feature in enumerate(x_features):
        query += feature + ","
    query += const.Y_COLUMN[task]
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    query += " WHERE test = true AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + ";"

def get_conn_cursor(task: str):
    conn_details = const.CONN_DETAILS[task]
    conn = psycopg2.connect(
        host = conn_details["host"],
        database = conn_details["database"],
        user = conn_details["user"],
        password = conn_details["password"]
    )
    cur = conn.cursor()
    return conn, cur

# Gets the count of a slice in source specified by idx
def get_slice_count(task: str, idx: int, slice_: npt.NDArray) -> npt.NDArray:
    conn, cur = get_conn_cursor(task)
    cur.execute(construct_count_query_str(task, idx, slice_))
    rows = cur.fetchall()
    cnt = rows
    cur.close()
    conn.close()
    return rows[0][0]
    
# Returns the JDBC connection string
def jdbc_str(task: str):
    conn_details = const.CONN_DETAILS[task]
    conn = 'jdbc:postgresql://' + conn_details["host"] + ':' + POSTGRES_DEFAULT_PORT
    conn += '/' + conn_details["database"]
    return conn

# Testing methods
if __name__ == "__main__":
    print("X features:")
    print(get_x_features("flights-classify"))
    print()

    print("All features:")
    print(get_all_features("flights-classify"))
    print()

    print("Query string:")
    print(construct_query_str("flights-classify", 0))
    print("Query result:")
    print(get_query_result("flights-classify", 2))
    print()

    slice_ = [None, 1, None, None, None, None, None, None, None, None, None, None]
    print("Slice count query string with slice " + str(slice_))
    print(construct_count_query_str("flights-classify", 0, slice_))
    print(get_slice_count("flights-classify", 0, slice_))

    slice_ = [None, None, None, None, None, None, None, None, None, None, 2, None]
    print("Slice count query string with slice " + str(slice_))
    print(construct_count_query_str("flights-classify", 0, slice_))
    print(get_slice_count("flights-classify", 0, slice_))