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
        cur.execute("DROP TABLE IF EXISTS table1;")
        cur.execute(
            "CREATE TABLE table1 (num integer, flt float, txt char(1), bool boolean, omit varchar);")
        cur.execute(
            "INSERT INTO table1 (num, flt, txt, bool, omit) VALUES (1, 1.0, 'a', TRUE, 'hello'), (-5, -1.0, 'b', TRUE, NULL), (3, 2.5, 'c', FALSE, 'world'), (1, 1.0, 'a', FALSE, NULL);")
        self.conn.commit()
        cur.close()
        # Create an instance of SimpleDBSource for this table
        self.source1 = dt.source.SimpleDBSource(self.conn_str, "table1", ['txt', 'num', 'flt', 'bool'])

    def tearDown(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE table1;")
        self.conn.commit()
        cur.close()
        self.conn.close()

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
        expected_data2 = [('b', -5, -1.0, True), ('c', 3, 2.5, False), ('a', 1, 1.0, False)]
        expected_batch2 = pd.DataFrame(data=expected_data2, columns=expected_columns)
        self.assertIsInstance(batch2, pd.DataFrame)  # Return type is dataframe
        pd.testing.assert_frame_equal(batch2, expected_batch2)

        # Get a batch of some rows, which should return None
        batch3 = self.source1.get_next_batch(batch_size=None)
        self.assertIsNone(batch3)
        batch4 = self.source1.get_next_batch(batch_size=1)
        self.assertIsNone(batch4)

        self.source1.close()
        batch5 = self.source1.get_next_batch(batch_size=None)
        self.assertIsNone(batch5)

    def test_has_next(self):
        self.assertEqual(self.source1.has_next(), True)
        self.source1.get_next_batch(batch_size=2)
        self.assertEqual(self.source1.has_next(), True)
        self.source1.get_next_batch(batch_size=2)
        self.assertEqual(self.source1.has_next(), False)
        self.source1.get_next_batch(batch_size=None)
        self.assertEqual(self.source1.has_next(), False)
        self.source1.close()

    def test_total_size(self):
        self.assertEqual(self.source1.total_size(), 4)
        self.source1.get_next_batch(batch_size=None)
        self.assertEqual(self.source1.total_size(), 4)
        self.source1.close()

    def test_slice_count(self):
        slice1 = {'bool': [('=', True)]}
        self.assertEqual(self.source1.slice_count(slice1), 2)
        slice2 = {'num': [('<', 0), ('>', -100)], 'flt': [('<', 0.0)]}
        self.assertEqual(self.source1.slice_count(slice2), 1)
        self.source1.close()


if __name__ == '__main__':
    ut.main()
