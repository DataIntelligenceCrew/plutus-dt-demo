import json
import os

from .source import *
from .utils import *

"""
A task in PLUTUS is defined as an instance of a concrete class that inherits AbstractTask. Several helper functions are
defined in AbstractTask such that, for most common tasks, defining a task amounts to dispatching the helper functions. 

Data is passed around as dataframes. Slices are passed around as numpy arrays. Since the pipeline passes
data between database, Dataframe, SystemDS, DT, and the dashboard, the whole pipeline requires recoding data to be
interpretable for humans, SystemDS, or the DT framework. This is done through several abstract methods defined below. 

If a task implementation deviates from the requirements, PLUTUS may exhibit undefined behavior. 
"""


class AbstractTask:
    @abstractmethod
    def __init__(self, train_source: AbstractSource, test_source: AbstractSource,
                 additional_sources: tp.List[AbstractSource], y_is_categorical: bool, name: str, description: str):
        self.initial_train, _ = train_source.get_next_batch(batch_size=None)
        self.test, _ = test_source.get_next_batch(batch_size=None)
        self.additional_sources = additional_sources
        self.y_is_categorical = y_is_categorical
        self.name = name
        self.description = description

    @abstractmethod
    def clean_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given some raw data that was returned directly from the various sources, and clean it into a DataFrame that is
        suitable for passing around and human-readable.
        """
        pass

    @abstractmethod
    def x_column_names(self) -> tp.List[str]:
        """
        Return the name of the x columns.
        """
        pass

    @abstractmethod
    def y_column_name(self) -> str:
        """
        Return the name of the y_column.
        """
        pass

    @abstractmethod
    def all_column_names(self) -> tp.List[str]:
        pass

    def get_train_set(self) -> tp.Optional[pd.DataFrame]:
        """
        Acquire and return the initial train set as a DataFrame that includes only the X and y variables.
        """
        return self.initial_train

    def get_train_xy(self) -> tp.Tuple[pd.DataFrame, npt.NDArray]:
        """
        Return the initial train set split into X (dataframe) and y (1-D ndarray).
        """
        if self.initial_train is None:
            raise ValueError("Missing train set.")
        return split_df_xy(self.initial_train, self.y_column_name())

    def get_test_set(self) -> tp.Optional[pd.DataFrame]:
        """
        Acquire and return the initial test set as a DataFrame that includes only the X and y variables.
        """
        return self.test

    def get_test_xy(self) -> tp.Tuple[pd.DataFrame, npt.NDArray]:
        if self.test is None:
            raise ValueError("Missing test set.")
        return split_df_xy(self.test, self.y_column_name())

    def get_additional_data(self, source_idx: int, batch_size: int) -> tp.Tuple[tp.Optional[pd.DataFrame], int]:
        """
        Return additional data from specified source, or None if no additional data is available.
        """
        source = self.additional_sources[source_idx]
        if source.has_next():
            return self.additional_sources[source_idx].get_next_batch(batch_size=batch_size)
        else:
            return None, 0

    def get_additional_data_xy(self, source_idx: int, batch_size: int) \
            -> tp.Union[tp.Tuple[pd.DataFrame, npt.NDArray, int], tp.Tuple[None, None, int]]:
        data, cost = self.get_additional_data(source_idx, batch_size)
        if data is None:
            return None, None, cost
        else:
            x, y = split_df_xy(data, self.y_column_name())
            return x, y, cost

    @abstractmethod
    def recode_for_model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Given some raw data that was returned directly from get_train_set, get_test_set or get_additional_data, recode
        it into another DataFrame that is suitable for model training and testing.
        """
        pass

    @abstractmethod
    def binning_for_sliceline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Given some raw data that was returned directly from get_train_set, get_test_set or get_additional_data, recode
        it into another DataFrame that is suitable for Sliceline. This requires every column to be integer-encoded
        with positive (no zero) values only.
        """
        pass

    @abstractmethod
    def recode_slice_to_human_readable(self, slice_: npt.NDArray) -> npt.NDArray:
        """
        In Sliceline, a slice is represented as a vector of integers. This is not human-interpretable. Recode the slice
        where each row is a human-readable slice with string representation of each slice.
        """
        pass

    @abstractmethod
    def recode_slice_to_sql_readable(self, slice_: npt.NDArray) -> tp.List[dict]:
        """
        A slice is SQL readable when it is a dictionary mapping from column name to list of (op, val) tuples. The full
        condition takes form col1 op1 val1 AND col1 op2 val2 AND col2 op3 val3 ...
        """
        pass

    @abstractmethod
    def slice_ownership(self, data: pd.DataFrame, slices: npt.NDArray) -> npt.NDArray:
        pass


"""
A simple boilerplate task that supports classifying or regressing a single y column. X columns are split into numeric
and categorical columns. Categorical columns are dummy-encoded for model training, and integer-encoded for Sliceline. 
The y column is left as is fort training, and is not used for binning. Supports equi-width, equi-count, and
predefined binning methods. 
Inheriting and overloading certain methods in SimpleTask is sufficient for many common tasks. 
"""


class SimpleTask(AbstractTask):
    @staticmethod
    def get_categorical_mapping_scheme(possible_values: tp.Set) -> tp.Tuple[tp.Dict, tp.Dict]:
        """
        Given a set of values that a categorical column can take, maps each possible value to a positive integer,
        starting from 1 up to n. Then, returns a mapping from integer to value, and value to integer.
        """
        list_form = list(possible_values)
        int_to_value_list = {(idx + 1): value for idx, value in enumerate(list_form)}
        value_to_int_dict = {value: idx + 1 for idx, value in enumerate(list_form)}
        return int_to_value_list, value_to_int_dict

    @staticmethod
    def get_equi_width_bin_borders(values: tp.List[float], num_bins: int) -> tp.List[float]:
        if len(values) <= 0:
            return []
        if num_bins <= 0:
            return []
        max_value, min_value = max(values), min(values)
        bin_width = (max_value - min_value) / num_bins
        bin_borders = [min_value + i * bin_width for i in range(num_bins + 1)]
        bin_borders[0] = float('-inf')
        bin_borders[-1] = float('inf')
        return bin_borders

    @staticmethod
    def get_equi_count_bin_borders(values: tp.List[float], num_bins: int) -> tp.List[float]:
        if len(values) <= 0:
            return []
        if num_bins <= 0:
            return []
        values.sort()
        quantiles = [i / num_bins for i in range(1, num_bins)]
        bin_borders = [np.quantile(values, q) for q in quantiles]
        return [float('-inf')] + bin_borders + [float('inf')]

    @staticmethod
    def load_from_directory(cls, directory: str):
        tasks = []
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                with open(os.path.join(directory, filename), 'r') as f:
                    config = json.load(f)
                    task = cls.new_from_config(config)
                    tasks.append(task)
        return tasks

    @staticmethod
    def new_from_config(config: dict):
        """
        config: A dictionary containing all config parameters. Must follow specific schema. Example:
        {
            'conn_str': 'libpq connection string...',
            'train_table': 'initial train set table name',
            'test_table': 'initial test set table name',
            'ext_tables': [],  // list of table names for additional data
            'numeric_x': [], // list of numeric x column names
            'categorical_x': [], // list of categorical x column names
            'y': str, // y column name
            'y_is_categorical': bool, // whether y column is a categorical column or a numeric column
            'binning': { // mapping from all numeric x column names to a binning method
                'col1': { 'method': 'equi-width', 'num_bins': 10 },
                'col2': { 'method': 'equi-count', 'num_bins': 5 },
                'col3': { 'method': 'predefined', 'num_bins': 3, 'borders': [-inf, 0.5, 1.5, inf] }
            }
        }
        """
        all_columns = config['numeric_x'] + config['categorical_x'] + [config['y']]
        train_source = SimpleDBSource.new_from_config({
            'conn_str': config['conn_str'], 'table': config['train_table'], 'columns': all_columns})
        test_source = SimpleDBSource.new_from_config({
            'conn_str': config['conn_str'], 'table': config['test_table'], 'columns': all_columns})
        additional_sources = [
            SimpleDBSource.new_from_config({
                'conn_str': config['conn_str'], 'table': table_name, 'columns': all_columns
            }) for table_name in config['ext_tables']]
        numeric_x_columns = set(config['numeric_x'])
        categorical_x_columns = set(config['categorical_x'])
        y_column = config['y']
        columns = list(numeric_x_columns) + list(categorical_x_columns) + [y_column]
        y_is_categorical = config['y_is_categorical']
        binning_method = config['binning']
        name = config['task_name']
        description = config['task_description']
        return SimpleTask(train_source, test_source, additional_sources, columns, numeric_x_columns,
                          categorical_x_columns, y_column, y_is_categorical, binning_method, name, description)

    def __init__(self, train_source: AbstractSource, test_source: AbstractSource,
                 additional_sources: tp.List[AbstractSource], columns: tp.List[str], numeric_x_columns: tp.Set[str],
                 categorical_x_columns: tp.Set[str], y_column: str, y_is_categorical: bool, binning: dict,
                 name: str = None, description: str = None):
        """
        params:
            train_source: An AbstractSource for acquiring initial train set.
            test_source: An AbstractSource for acquiring initial test set.
            additional_sources: A list of AbstractSources for acquiring additional data.
            columns: A list of all column names. This order defines the order of DataFrames and numpy arrays.
            numeric_x_columns: A set of column names for numeric X features.
            categorical_x_columns: A set of column names for categorical X features.
            y_column: Name of y column.
            y_is_categorical: Whether y column is categorical or numeric.
            binning_method: A mapping from numeric x column names to SimpleBinningMethod.
        """
        super().__init__(train_source, test_source, additional_sources, y_is_categorical, name, description)
        # Store all the column names
        self.numeric_x_columns = [column for column in columns if column in numeric_x_columns]
        self.categorical_x_columns = [column for column in columns if column in categorical_x_columns]
        self.x_columns = [column for column in columns if
                          column in numeric_x_columns or column in categorical_x_columns]
        self.y_column = y_column
        self.all_columns = columns
        # Store other parameters
        self.y_is_categorical = y_is_categorical
        # Compute numeric bin borders, which will stay fixed
        self.binning = binning
        self.bin_borders = dict()
        for x in self.numeric_x_columns:
            x_binning = self.binning[x]
            if x_binning['method'] == 'equi-width':
                borders = self.get_equi_width_bin_borders(self.initial_train[x].tolist(), x_binning['num_bins'])
            elif x_binning['method'] == 'equi-count':
                if x == 'total_bedrooms':
                    print(self.initial_train[x].tolist())
                borders = self.get_equi_count_bin_borders(self.initial_train[x].tolist(), x_binning['num_bins'])
            elif x_binning['method'] == 'predefined':
                borders = x_binning['borders']
            else:
                raise ValueError('Unknown binning method.')
            self.bin_borders.update({x: borders})
        # Pre-compute the string representation of each bin
        self.bin_names = dict()
        for x in self.numeric_x_columns:
            bin_names = ['']  # 1-indexing
            for idx in range(len(self.bin_borders[x])):
                bin_str = f'({format(self.bin_borders[x][idx], ".2f")}, {format(self.bin_borders[x][idx] + 1, ".2f")})'
                bin_names.append(bin_str)
            self.bin_names.update({x: bin_names})
        # Compute and store categorical mapping scheme
        self.categorical_mappings = dict()
        for x in self.categorical_x_columns:
            idx_to_label, label_to_idx = self.get_categorical_mapping_scheme(set(self.initial_train[x].tolist()))
            self.categorical_mappings.update({x: {'idx_to_label': idx_to_label, 'label_to_idx': label_to_idx}})

    def clean_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Project only the X and y columns
        df = data[list(self.all_columns)]
        return df

    def x_column_names(self) -> tp.List[str]:
        return self.x_columns

    def y_column_name(self) -> str:
        return self.y_column

    def all_column_names(self) -> tp.List[str]:
        return self.all_columns

    def recode_for_model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # This should dummy-encode, normalize, turn values numeric, etc., as necessary
        # Python XGBoost has support for categorical features OOTB, so we simply mark those columns as category
        for x in self.categorical_x_columns:
            dataset[x] = dataset[x].astype('category')
        return dataset

    def binning_for_sliceline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        print(self.bin_borders)
        for col in self.numeric_x_columns:
            dataset[col] = pd.cut(dataset[col], bins=self.bin_borders[col], labels=False) + 1
        for col in self.categorical_x_columns:
            dataset[col] = dataset[col].map(self.categorical_mappings[col]['label_to_idx'])
        return dataset

    def recode_slice_to_human_readable(self, slices: npt.NDArray) -> npt.NDArray:
        human_readable_slices = []
        for slice_ in slices:
            human_readable_slice = []
            for column_idx, bin_idx in enumerate(slice_):
                column_name = self.all_columns[column_idx]
                if column_name in self.numeric_x_columns:
                    human_readable_slice.append(self.bin_names[column_name][bin_idx])
                else:
                    label = self.categorical_mappings[column_name]['idx_to_label'][bin_idx]
                    human_readable_slice.append(label)
            human_readable_slices.append(human_readable_slice)
        return np.array(human_readable_slices)

    def recode_slice_to_sql_readable(self, slices: npt.NDArray) -> tp.List[dict]:
        sql_readable_slices = []
        for slice_ in slices:
            sql_readable_slice = dict()
            for column_idx, bin_idx in enumerate(slice_):
                if bin_idx is None or bin_idx == 0:  # This column does not matter for the slice
                    continue
                column_name = self.all_columns[column_idx]
                if column_name in self.numeric_x_columns:
                    column_conds = []
                    print(self.bin_borders[column_name])
                    lo = self.bin_borders[column_name][bin_idx - 1]
                    hi = self.bin_borders[column_name][bin_idx]
                    print(column_name, lo, hi)
                    if lo != float('-inf'):
                        column_conds.append(('>=', lo))
                    if hi != float('inf'):
                        column_conds.append(('<', hi))
                    sql_readable_slice.update({column_name: column_conds})
            sql_readable_slices.append(sql_readable_slice)
        print(sql_readable_slices)
        return sql_readable_slices

    def slice_ownership(self, data: pd.DataFrame, slices: npt.NDArray) -> npt.NDArray:
        data_binned = self.binning_for_sliceline(data).values
        slices_mask = slices > 0
        data_binned_expanded = data_binned[:, np.newaxis, :]
        slices_expanded = slices[np.newaxis, :, :]
        data_binned_masked = np.where(slices_mask, data_binned_expanded, 0)
        slices_masked = np.where(slices_mask, slices_expanded, 0)
        result = np.all(data_binned_masked == slices_masked , axis=-1)
        result = result.astype(int)
        return result
