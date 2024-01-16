import psycopg2
import pandas as pd

POSTGRES_DEFAULT_PORT = 5432

# A class encapsulating all metadata and methods related to a data source
# simulated using postgres. 
class DBSource:
    def __init__(self, 
        host: str, 
        database: str, 
        user: str, 
        password: str, 
        query: str, 
        y: str):
        """
        params:
            host: Name of host, e.g. 'localhost'. 
            database: Name of postgres database, e.g. 'dtdemo'. 
            user, password: User and password of the database. 
            query: Query string that will be issued to the DB. 
            y: The name of the y variable in query result. 
        """
        self._host = host
        self._database = database
        self._user = user
        self._password = password
        self._query = query
        self.y = y

    # Runs the query and returns result as DF
    def get_query_result(self) -> pd.DataFrame:
        """
        returns:
            The result of running the query in raw "DB" format. 
        """
        # Create connetion & cursor
        conn = conn = psycopg2.connect(
            host = self._host, 
            database = self._database, 
            user = self._user, 
            password = self._password
        )
        cur = conn.cursor()
        # Attach query to cursor
        cur.execute(self._query)
        # Get query result
        rows = cur.fetchall()
        # Represent query result as dataframe
        col_names = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=col_names)
        # Close cursor & query
        cur.close()
        conn.close()
        # Return dataframe
        return df
    
    # Returns the JDBC connection string
    def jdbc_str(self):
        conn = 'jdbc:postgresql://' + self._host + ':' + POSTGRES_DEFAULT_PORT
        conn += '/' + self._database
        return conn

    def __str__(self):
        s = 'Connection: ' + self.jdbc_str()
        s += ', Query: ' + self._query
        s += ', Y: ' + self.y
        return s