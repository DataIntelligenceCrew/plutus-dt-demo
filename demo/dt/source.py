from abc import abstractmethod
from random import random

import pandas as pd
import typing as tp
import psycopg2
from pandas._typing import npt

"""
An abstract class for data sources, which are used to acquire train, test, and additional data. 
"""


class AbstractSource:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_next_batch(self, batch_size: tp.Union[int, None]) -> tp.Tuple[tp.Union[pd.DataFrame, None], int]:
        """
        Return:
             batch of data queried directly from some database or file, or None if no fresh tuples remain
             the cost of querying the batch
        params:
            batch_size: number of tuples to query, or None to query all remaining tuples
        """
        pass

    @abstractmethod
    def has_next(self) -> bool:
        """
        Returns whether there is fresh data available for another batch of querying.
        """
        pass

    @abstractmethod
    def total_size(self) -> int:
        """
        Returns the total number of fresh tuples before any has been queried yet.
        """
        pass

    @abstractmethod
    def slice_count(self, slice_defn: dict) -> tp.Union[int, None]:
        """
        params:
            slice_defn: A dictionary mapping from column names to slice definitions. A slice definition is an iterable
                        collection of 2-tuples. Each tuple contains a string operator (e.g. '==', '>', '<') and a value.
                        For example, the slice [('>', 0), ('<', 10)] represents 'WHERE column > 0 AND column < 10'.
        Returns the total number of fresh tuples in the source that belongs to the slice, or None if it is uncountable.
        """
        pass

    def slices_count(self, slice_defs: tp.List[dict]) -> tp.List[int]:
        pass

    @abstractmethod
    def amortized_cost_per_tuple(self) -> int:
        pass


"""
A simple database-backed data source which can query for specified columns in batches from a single table. 
It cannot handle NULL values in tables, and will have undefined behavior if the table has any NULL values. 
"""


class SimpleDBSource(AbstractSource):
    @classmethod
    def new_from_config(cls, config: dict):
        """
        params:
            config: A dictionary mapping from parameter name to parameter value. Parameter names must include conn_str,
                    table, columns. batch_size can be omitted and will default to None if so.
        """
        conn_str = config['conn_str']
        table = config['table']
        columns = config['columns']
        return SimpleDBSource(conn_str, table, columns)

    def __init__(self, conn_str: str, table: str, columns: tp.List[str]):
        """
        params:
            conn_str: a libpq connection string to connect to host database
            columns: a list of column names to query from the database, including all X, y columns and no more
            batch_size: number of tuples to query per batch, or none if no such limits exist
        """
        # Copy relevant parameters
        self.table: str = table
        self.columns: tp.List[str] = columns
        # Acquire connection & cursor to table, then execute select query
        self.conn_str = conn_str
        self.conn = psycopg2.connect(conn_str)
        self.select_cur = self.conn.cursor(name='select_cur' + table)
        query = f"SELECT {','.join(self.columns)} FROM {self.table};"
        self.select_cur.execute(query)
        self.select_cur.fetchone()
        print(self.select_cur, query)
        self.return_column_names = [desc[0] for desc in self.select_cur.description]
        # Count the number of tuples in table
        self.initial_size: int = self._total_size()
        self.num_queried: int = 0  # Number of fresh tuples queried so far

    def close(self):
        if self.select_cur is not None:
            self.select_cur.close()
        self.conn.commit()
        self.conn.close()

    def reset(self):
        self.close()
        # Acquire connection & cursor to table, then execute select query
        self.conn = psycopg2.connect(self.conn_str)
        self.select_cur = self.conn.cursor(name='select_cur' + self.table)
        query = f"SELECT {','.join(self.columns)} FROM {self.table};"
        self.select_cur.execute(query)
        self.select_cur.fetchone()
        print(self.select_cur, query)
        self.return_column_names = [desc[0] for desc in self.select_cur.description]
        # Count the number of tuples in table
        self.initial_size: int = self._total_size()
        self.num_queried: int = 0  # Number of fresh tuples queried so far

    def get_next_batch(self, batch_size: tp.Union[int, None]) -> tp.Tuple[tp.Union[pd.DataFrame, None], int]:
        if not self.has_next():
            return None, 0
        # Fetch batch of rows from table then convert to DataFrame
        if batch_size is not None:  # Some integer batch size specified
            rows = self.select_cur.fetchmany(batch_size)
        else:  # No batch size limit
            rows = self.select_cur.fetchall()
        df = pd.DataFrame(rows, columns=self.return_column_names)
        # Close cursor & connection if no fresh tuples remain in table
        self.num_queried += len(rows)
        if not self.has_next():
            self.select_cur.close()
        return df, len(df)

    def has_next(self) -> bool:
        return self.initial_size > self.num_queried

    def total_size(self) -> int:
        return self.initial_size

    def slices_count(self, slice_defs: tp.List[dict]) -> tp.List[int]:
        count_cur = self.conn.cursor()
        values = []
        query_parts = []
        for idx, slice_ in enumerate(slice_defs):
            condition = ""
            for column, conditions in slice_.items():
                for op, val in conditions:
                    condition += f"{column} {op} %s AND "
                    if isinstance(val, float):
                        if val.is_integer():
                            val = int(val)
                    values.append(str(val))
            condition += "TRUE"
            query_parts.append(f"COUNT(CASE WHEN {condition} THEN 1 END) AS c{idx}")
        query = f"SELECT {', '.join(query_parts)} FROM {self.table};"
        print(query, values)
        count_cur.execute(query, tuple(values))
        ret = count_cur.fetchone()
        print("Slices count query result:", ret)
        count_cur.close()
        return list(ret)

    def _total_size(self) -> int:
        """
        returns the total number of fresh tuples in the table at the beginning
        """
        count_cur = self.conn.cursor()
        query = f"SELECT COUNT(*) FROM {self.table};"
        count_cur.execute(query)
        return count_cur.fetchone()[0]

    def amortized_cost_per_tuple(self) -> int:
        return 1
