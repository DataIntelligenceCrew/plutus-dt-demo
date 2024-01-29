import psycopg2
import pandas as pd
import numpy as np
import numpy.typing as npt
import typing as tp
from . import const
from . import utils

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
    query += " AND train2 = false AND test = false "
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
    query += " train2 = false AND test = false AND "
    # Select the correct source
    query += const.SOURCES[task]["pivot"] + " = " \
        + str(const.SOURCES[task]["pivot_values"][idx])
    # Remove null y values
    query += " AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + ";"

def construct_source_size_query_str(task: str, idx: int) -> str:
    query = "SELECT COUNT(*)"
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    query += " WHERE train2 = false AND test = false AND "
    query += const.SOURCES[task]["pivot"] + " = " \
        + str(const.SOURCES[task]["pivot_values"][idx])
    query += " AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + ";"

def construct_train_query_str(task: str) -> str:
    query = "SELECT "
    x_features = get_x_features(task)
    for i, feature in enumerate(x_features):
        query += feature + ","
    query += const.Y_COLUMN[task]
    query += " FROM " + const.CONN_DETAILS[task]["table"]
    query += " WHERE train2 = true AND " + const.Y_COLUMN[task] + " IS NOT NULL"
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

# Gets the size of a source
def get_source_size(task: str, idx: int) -> npt.NDArray:
    conn, cur = get_conn_cursor(task)
    cur.execute(construct_source_size_query_str(task, idx))
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

def construct_slice_cnt_select_query(task: str, idx: int) -> str:
    query = "SELECT "
    x_features = get_x_features(task)
    for i, feature in enumerate(x_features):
        query += feature + ","
    query += const.Y_COLUMN[task]
    query += " FROM " + const.CONN_DETAILS[task]['table']
    query += " TABLESAMPLE SYSTEM(1) "
    query += " WHERE train2 = false AND test = false AND "
    query += const.SOURCES[task]["pivot"] + " = " \
        + str(const.SOURCES[task]["pivot_values"][idx])
    query += " AND " + const.Y_COLUMN[task] + " IS NOT NULL"
    return query + " LIMIT 100000;"

def get_slice_cnts_in_source(slices: npt.NDArray, source: int, task: str) -> npt.NDArray:
    conn, cur = get_conn_cursor(task)
    cur.itersize = 1000
    query = construct_slice_cnt_select_query(task, source)
    cur.execute(query)
    slice_cnts = np.zeros((1,len(slices)), dtype=int)
    col_names = [desc[0] for desc in cur.description]
    while True:
        rows = cur.fetchmany(cur.itersize)
        if not rows:
            break
        rows_df = pd.DataFrame(rows, columns=col_names)
        rows_raw = utils.recode_db_to_raw(rows_df, task)
        rows_binned = utils.recode_raw_to_binned(rows_raw, task)
        rows_binned = rows_binned.drop(const.Y_COLUMN[task], axis=1)
        add_cnts = get_slice_cnts(rows_binned.to_numpy(), slices)
        slice_cnts += add_cnts
    cur.close()
    conn.close()
    return slice_cnts

def slice_ownership(
    binned_x: npt.NDArray, 
    slices: npt.NDArray
) -> npt.NDArray:
    """
    params:
        binned_x: X variables of data in binned encoding, as numpy matrix. 
        slices: Slices encoded as numpy matrix allowing None. 
    returns:
        A numpy boolean matrix where array[i][j] is set to True if the ith
        tuple belongs to the jth slice. 
    """
    n = len(binned_x)
    k = len(slices)
    ownership_mat = np.empty((n,k), dtype=bool)
    for i in range(n):
        for j in range(k):
            ownership_mat[i][j] = belongs_to_slice(binned_x[i], slices[j])
    return ownership_mat

def belongs_to_slice(
    x: npt.NDArray,
    slice_: npt.NDArray
) -> bool:
    """
    params: 
        x: d-length numpy array encoding one data point. 
        slice: d-length numpy array encoding one slice. 
    """
    # Filter out None columns
    not_none_mask = slice_ != None
    not_none_x = x[not_none_mask]
    # Test equality of not-None columns
    not_none_slice = slice_[not_none_mask]
    return np.array_equal(not_none_x, not_none_slice)

def get_slice_cnts(
    binned_x: npt.NDArray,
    slices: pd.DataFrame
) -> npt.NDArray:
    """
    params:
        binned_x: X variables of dataset in binned encoding. 
        slices: Top-k slices in integer dataframe format. 
    returns:
        The number of rows in binned_x that belong to the given slices. 
    """
    ownership_mat = slice_ownership(binned_x, slices)
    slice_cnts = np.sum(ownership_mat, axis=0)
    return slice_cnts

def construct_stats_table(slices: npt.NDArray, task: str):
    n = const.SOURCES[task]['n']
    m = len(slices)
    stat_tracker = np.empty((n, m), dtype=float)
    for source in range(n):
        stat_tracker[source,:] = get_slice_cnts_in_source(slices, source, task)
        source_cnt = const.SOURCES[task]['counts'][source]
        stat_tracker[source,:] /= source_cnt
    print("stat tracker", stat_tracker)
    return stat_tracker

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

