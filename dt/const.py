# Defines all constants related to data sources. 
# Valid data source keys are:
# "flights-classify"
# "flights-regress"
# These variables ensure consistent encoding and binning, etc. 

CATEGORICAL_FEATURES = {
    "flights-classify": ['carrier_mkt', 'carrier_op', 'origin_state', 'dest_state'],
    "flights-regress": ['carrier_mkt', 'carrier_op', 'origin_state', 'dest_state']
}

NUMERIC_FEATURES = {
    "flights-classify": ['month', 'day', 'weekday', 'distance', 'origin_latitude', 'origin_longitude', 'dest_latitude', 'dest_longitude'],
    "flights-regress": ['month', 'day', 'weekday', 'distance', 'origin_latitude', 'origin_longitude', 'dest_latitude', 'dest_longitude']
}

# Subset of categorical features that will be dummy encoded
# All else will be numeric encoded using CATEGORICAL_MAPPINGS
# Keep these empty for fast testing
DUMMY_FEATURES = {
    "flights-classify": [],
    "flights-regress": []
}

Y_COLUMN = {
    "flights-classify": "arrival_code",
    "flights-regress": "arrival_delay"
}

Y_IS_CATEGORICAL = {
    "flights-classify": True,
    "flights-regress": False
}

CATEGORICAL_MAPPINGS = {
    "flights-classify": {
        "carrier_mkt": {
            "AA": 0, "AS": 1, "B6": 2, "DL": 3, "F9": 4, "G4": 5, "HA": 6, 
            "NK": 7, "UA": 8, "VX": 9, "WN": 10
        },
        "carrier_op": {
            "9E": 0, "9K": 1, "AA": 2, "AS": 3, "AX": 4, "B6": 5, "C5": 6, 
            "CP": 7, "DL": 8, "EM": 9, "EV": 10, "F9": 11, "G4": 12, "G7": 13, 
            "HA": 14, "KS": 15, "MQ": 16, "NK": 17, "OH": 18, "OO": 19, "PT": 20, 
            "QX": 21, "UA": 22, "VX": 23, "WN": 24, "YV": 25, "YX": 26, "ZW": 27
        }, 
        "origin_state": {
            "AK": 0, "AL": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5, "CT": 6, "DE": 7, 
            "FL": 8, "GA": 9, "HI": 10, "IA": 11, "ID": 12, "IL": 13, "IN": 14, 
            "KS": 15, "KY": 16, "LA": 17, "MA": 18, "MD": 19, "ME": 20, "MI": 21, 
            "MN": 22, "MO": 23, "MS": 24, "MT": 25, "NC": 26, "ND": 27, "NE": 28, 
            "NH": 29, "NJ": 30, "NM": 31, "NV": 32, "NY": 33, "OH": 34, "OK": 35, 
            "OR": 36, "PA": 37, "PR": 38, "RI": 39, "SC": 40, "SD": 41, "TN": 42, 
            "TT": 43, "TX": 44, "UT": 45, "VA": 46, "VI": 47, "VT": 48, "WA": 49, 
            "WI": 50, "WV": 51 , "WY": 52
        },
        "dest_state": {
            "AK": 0, "AL": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5, "CT": 6, "DE": 7, 
            "FL": 8, "GA": 9, "HI": 10, "IA": 11, "ID": 12, "IL": 13, "IN": 14, 
            "KS": 15, "KY": 16, "LA": 17, "MA": 18, "MD": 19, "ME": 20, "MI": 21, 
            "MN": 22, "MO": 23, "MS": 24, "MT": 25, "NC": 26, "ND": 27, "NE": 28, 
            "NH": 29, "NJ": 30, "NM": 31, "NV": 32, "NY": 33, "OH": 34, "OK": 35, 
            "OR": 36, "PA": 37, "PR": 38, "RI": 39, "SC": 40, "SD": 41, "TN": 42, 
            "TT": 43, "TX": 44, "UT": 45, "VA": 46, "VI": 47, "VT": 48, "WA": 49, 
            "WI": 50, "WV": 51, "WY": 52
        },
        "arrival_code": {
            "O": 0,
            "D": 1,
            "C": 2
        }
    },
    "flights-regress": {
        "carrier_mkt": {
            "AA": 0, "AS": 1, "B6": 2, "DL": 3, "F9": 4, "G4": 5, "HA": 6, 
            "NK": 7, "UA": 8, "VX": 9, "WN": 10
        },
        "carrier_op": {
            "9E": 0, "9K": 1, "AA": 2, "AS": 3, "AX": 4, "B6": 5, "C5": 6, 
            "CP": 7, "DL": 8, "EM": 9, "EV": 10, "F9": 11, "G4": 12, "G7": 13, 
            "HA": 14, "KS": 15, "MQ": 16, "NK": 17, "OH": 18, "OO": 19, "PT": 20, 
            "QX": 21, "UA": 22, "VX": 23, "WN": 24, "YV": 25, "YX": 26, "ZW": 27
        }, 
        "origin_state": {
            "AK": 0, "AL": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5, "CT": 6, "DE": 7, 
            "FL": 8, "GA": 9, "HI": 10, "IA": 11, "ID": 12, "IL": 13, "IN": 14, 
            "KS": 15, "KY": 16, "LA": 17, "MA": 18, "MD": 19, "ME": 20, "MI": 21, 
            "MN": 22, "MO": 23, "MS": 24, "MT": 25, "NC": 26, "ND": 27, "NE": 28, 
            "NH": 29, "NJ": 30, "NM": 31, "NV": 32, "NY": 33, "OH": 34, "OK": 35, 
            "OR": 36, "PA": 37, "PR": 38, "RI": 39, "SC": 40, "SD": 41, "TN": 42, 
            "TT": 43, "TX": 44, "UT": 45, "VA": 46, "VI": 47, "VT": 48, "WA": 49, 
            "WI": 50, "WV": 51, "WY": 52
        },
        "dest_state": {
            "AK": 0, "AL": 1, "AR": 2, "AZ": 3, "CA": 4, "CO": 5, "CT": 6, "DE": 7, 
            "FL": 8, "GA": 9, "HI": 10, "IA": 11, "ID": 12, "IL": 13, "IN": 14, 
            "KS": 15, "KY": 16, "LA": 17, "MA": 18, "MD": 19, "ME": 20, "MI": 21, 
            "MN": 22, "MO": 23, "MS": 24, "MT": 25, "NC": 26, "ND": 27, "NE": 28, 
            "NH": 29, "NJ": 30, "NM": 31, "NV": 32, "NY": 33, "OH": 34, "OK": 35, 
            "OR": 36, "PA": 37, "PR": 38, "RI": 39, "SC": 40, "SD": 41, "TN": 42, 
            "TT": 43, "TX": 44, "UT": 45, "VA": 46, "VI": 47, "VT": 48, "WA": 49, 
            "WI": 50, "WV": 51, "WY": 52
        }
    }
}

REVERSE_CATEGORICAL_MAPPINGS = {
    "flights-classify": {
        "carrier_mkt": [
            "AA", "AS", "B6", "DL", "F9", "G4", "HA", "NK", "UA", "VX", "WN"
        ],
        "carrier_op": [
            "9E", "9K", "AA", "AS", "AX", "B6", "C5", "CP", "DL", "EM", "EV",
            "F9", "G4", "G7", "HA", "KS", "MQ", "NK", "OH", "OO", "PT", "QX",
            "UA", "VX", "WN", "YV", "YX", "ZW"
        ],
        "origin_state": [
            "AK", "AL", "AR", "AZ", "CA", "CO", "CT", 'DE', "FL", "GA", "HI",
            "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI",
            "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
            "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TT",
            "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV", "WY"
        ],
        "dest_state": [
            "AK", "AL", "AR", "AZ", "CA", "CO", "CT", 'DE', "FL", "GA", "HI",
            "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI",
            "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
            "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TT",
            "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV", "WY"
        ],
        "arrival_code": [
            "O", "D", "C"
        ]
    },
    "flights-regress": {
        "carrier_mkt": [
            "AA", "AS", "B6", "DL", "F9", "G4", "HA", "NK", "UA", "VX", "WN"
        ],
        "carrier_op": [
            "9E", "9K", "AA", "AS", "AX", "B6", "C5", "CP", "DL", "EM", "EV",
            "F9", "G4", "G7", "HA", "KS", "MQ", "NK", "OH", "OO", "PT", "QX",
            "UA", "VX", "WN", "YV", "YX", "ZW"
        ],
        "origin_state": [
            "AK", "AL", "AR", "AZ", "CA", "CO", "CT", 'DE', "FL", "GA", "HI",
            "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI",
            "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
            "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TT",
            "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV"
        ],
        "dest_state": [
            "AK", "AL", "AR", "AZ", "CA", "CO", "CT", 'DE', "FL", "GA", "HI",
            "IA", "ID", "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI",
            "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV",
            "NY", "OH", "OK", "OR", "PA", "PR", "RI", "SC", "SD", "TN", "TT",
            "TX", "UT", "VA", "VI", "VT", "WA", "WI", "WV"
        ]
    }
}

# Since optimal binning is not the main point of the demo, predefined equi-count
# bins are fine. For the sake of interpretability, if a feature is roughly
# uniformly distributed, then equi-width binning is used. 
# If a feature is skewed, then equi-count binning is used instead. 
# The number of bins are roughly based on optbinning on a subset of data. 
NUMERIC_BIN_BORDERS = {
    "flights-classify": {
        "year": [2017.5, 2018.5, 2019.5, 2020.5, 2021.5, 2022.5, 2023.5], 
        "month": [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5], 
        "day": [0.0, 5.5, 10.5, 15.5, 20.5, 25.5, 32.0], 
        "weekday": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
        "distance": [0.0, 235.5, 356.5, 481.5, 626.5, 817.5, 1015.5, 1444.5, 6000], 
        "origin_latitude": [-15, 29.98333333, 33.43416667, 35.03527778, 37.61888889, 39.77444444, 40.78416667, 42.36305556, 72], 
        "origin_longitude": [-177, -118.40805556, -107.89472222, -96.85083333, -87.90666667, -84.42805556, -80.94916667, -76.67, 150],
        "dest_latitude": [-15, 29.98333333, 33.43416667, 35.03527778, 37.61888889, 39.77444444, 40.78416667, 42.36305556, 72],
        "dest_longitude": [-177, -118.40805556, -107.89472222, -96.85083333, -87.90666667, -84.42805556, -80.94916667, -76.67, 150]
    },
    "flights-regress": {
        "year": [2017.5, 2018.5, 2019.5, 2020.5, 2021.5, 2022.5, 2023.5], 
        "month": [0.5, 2.5, 4.5, 6.5, 8.5, 10.5, 12.5], 
        "day": [0.0, 5.5, 10.5, 15.5, 20.5, 25.5, 32.0], 
        "weekday": [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5], 
        "distance": [0.0, 235.5, 356.5, 481.5, 626.5, 817.5, 1015.5, 1444.5, float('inf')], 
        "origin_latitude": [-15, 29.98333333, 33.43416667, 35.03527778, 37.61888889, 39.77444444, 40.78416667, 42.36305556, 72], 
        "origin_longitude": [-177, -118.40805556, -107.89472222, -96.85083333, -87.90666667, -84.42805556, -80.94916667, -76.67, 150],
        "dest_latitude": [-15, 29.98333333, 33.43416667, 35.03527778, 37.61888889, 39.77444444, 40.78416667, 42.36305556, 72],
        "dest_longitude": [-177, -118.40805556, -107.89472222, -96.85083333, -87.90666667, -84.42805556, -80.94916667, -76.67, 150]
    }
}

CONN_DETAILS = {
    "flights-classify": {
        "host": "localhost",
        "database": "dtdemo",
        "user": "jwc",
        "password": "postgres",
        "table": "flights"
    },
    "flights-regress": {
        "host": "localhost",
        "database": "dtdemo",
        "user": "jwc",
        "password": "postgres",
        "table": "flights"
    }
}

SOURCES = {
    "flights-classify": {
        "pivot": "carrier_mkt",
        "n": 11,
        "pivot_values": ["'AA'", "'AS'", "'B6'", "'DL'", "'F9'", "'G4'", "'HA'", "'NK'", "'UA'", "'VX'", "'WN'"],
        "counts": [9866812, 2120489, 1387942, 8303596, 753758, 596051, 418714, 1114537, 7132813, 17180, 6899040],
        "costs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "batch": 100,
        "sample_percent": 0.1
    },
    "flights-regress": {
        "pivot": "carrier_mkt",
        "n": 11,
        "pivot_values": ["'AA'", "'AS'", "'B6'", "'DL'", "'F9'", "'G4'", "'HA'", "'NK'", "'UA'", "'VX'", "'WN'"],
        "counts": [9866812, 2120489, 1387942, 8303596, 753758, 596051, 418714, 1114537, 7132813, 17180, 6899040],
        "costs": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "batch": 100,
        "sample_percent": 0.1
    }
}

EPSILON_PROB = 0.000001

DATASET_DESCRIPTIONS = {
    "flights-classify": """
* Dataset: US Bureau of Transportation Statistics [On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK)\n
* n = 40mil\n
* Data sources: Split by marketing airline

**X variables**

| **Variable** | **Type** | **Description** |
|---|---|---|
| year | int | 2018 - 2023 |
| month | int | Month |
| day | int | Day of month |
| weekday | int | Day of week |
| scheduled departure time | int | 00:00 to 23:59 |
| marketing carrier | categorical | Airline that sold the tickets (9 total) |
| operating carrier | categorical | Airline that operated the airpline (21 total) |
| origin & destination location | float | longitude & latitude of origin & destination |
| origin & destination state | categorical | State of origin & destination (51 total) |
| distance | int | Travel distance (in miles) |

**y variable**

| **Variable**        | **Type**    | **Values**                  |
|---------------------|-------------|-----------------------------|
| arrival performance | categorical | on-time, delayed, or cancelled |
""",

    "census": """
TODO: write a description for the census data
""",

"flights-regress": """
* Dataset: US Bureau of Transportation Statistics [On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK)\n
* n = 40mil\n
* Data sources: Split by marketing airline

**X variables**

| **Variable** | **Type** | **Description** |
|---|---|---|
| year | int | 2018 - 2023 |
| month | int | Month |
| day | int | Day of month |
| weekday | int | Day of week |
| scheduled departure time | int | 00:00 to 23:59 |
| marketing carrier | categorical | Airline that sold the tickets (9 total) |
| operating carrier | categorical | Airline that operated the airpline (21 total) |
| origin & destination location | float | longitude & latitude of origin & destination |
| origin & destination state | categorical | State of origin & destination (51 total) |
| distance | int | Travel distance (in miles) |

**y variable**

| **Variable**        | **Type**    | **Values**                  |
|---------------------|-------------|-----------------------------|
| arrival performance | categorical | on-time, delayed, or cancelled |
""",

    "census": """
TODO: write a description for the census data
""",

"flights-classify": """
* Dataset: US Bureau of Transportation Statistics [On-Time Performance](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGK)\n
* n = 40mil\n
* Data sources: Split by marketing airline

**X variables**

| **Variable** | **Type** | **Description** |
|---|---|---|
| year | int | 2018 - 2023 |
| month | int | Month |
| day | int | Day of month |
| weekday | int | Day of week |
| scheduled departure time | int | 00:00 to 23:59 |
| marketing carrier | categorical | Airline that sold the tickets (9 total) |
| operating carrier | categorical | Airline that operated the airpline (21 total) |
| origin & destination location | float | longitude & latitude of origin & destination |
| origin & destination state | categorical | State of origin & destination (51 total) |
| distance | int | Travel distance (in miles) |

**y variable**

| **Variable**        | **Type**    | **Values**                  |
|---------------------|-------------|-----------------------------|
| arrival performance | int | arrival delay, in minutes |
"""
}