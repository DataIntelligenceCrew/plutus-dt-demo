import sys
from pathlib import Path

# Add the root directory to the sys.path list
sys.path.append(str(Path(__file__).resolve().parent.parent))

import unittest as ut

import pandas as pd
import psycopg2

import dt


class TestSimpleDBSource(ut.TestCase):
    def setUp(self):
        self.conn_str = "dbname=test user=jchang38 port=5432"
        # Create a table in the test database
        self.conn = psycopg2.connect(self.conn_str)
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE table1 (id serial PRIMARY KEY, num integer, flt float, txt char(1), bool boolean, omit varchar);")
        cur.execute(
            "INSERT INTO table1 (num, flt, txt, bool, omit) VALUES (1, 1.0, 'a', TRUE, 'hello'), (-5, -1.0, 'b', TRUE, NULL), (3, 2.5, 'c', FALSE, 'world'), (1, 1.0, 'a', FALSE, NULL)")
        cur.close()
        # Create an instance of SimpleDBSource for this table
        self.source1 = SimpleDBSource(self.conn_str, "table1", ['txt', 'num', 'flt', 'bool'])

    def tearDown(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE table1")
        self.conn.close()

    def test_new_from_config(self):
        pass

    def test_get_next_batch(self):
        # Get a batch of 1 row, and also generate the expected (correct) dataframe
        batch1 = self.source1.get_next_batch(batch_size=1)
        expected_columns = ['txt', 'num', 'flt', 'bool']
        expected_data1 = [('a', 1, 1.0, True)]
        expected_batch1 = pd.DataFrame(data=expected_data1, columns=expected_columns)
        # Check equality between the returned dataframe and expected dataframe
        self.assertIsInstance(batch1, pd.DataFrame)  # Return type is dataframe
        pd.testing.assert_frame_equal(batch1, expected_batch1)

        # Get a batch of unlimited rows, which should give the remaining 2 rows
        batch2 = self.source1.get_next_batch(batch_size=None)
        expected_data2 = [('c', 3, 2.5, False), ('a', 1, 1.0, False)]
        expected_batch2 = pd.DataFrame(data=expected_data2, columns=expected_columns)
        self.assertIs(batch2, pd.DataFrame)  # Return type is dataframe
        pd.testing.assert_frame_equal(batch2, expected_batch2)

        # Get a batch of some rows, which should return None
        batch3 = self.source1.get_next_batch(batch_size=None)
        self.assertIsNone(batch3)
        batch4 = self.source1.get_next_batch(batch_size=1)
        self.assertIsNone(batch4)

    def test_has_next(self):
        self.fail()

    def test_total_size(self):
        self.fail()

    def test_slice_count(self):
        self.fail()

    def test__total_size(self):
        self.fail()


if __name__ == '__main__':
    ut.main()
