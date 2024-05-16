import math
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
import pandas as pd
import typing as tp
from .source import AbstractSource

"""
A task in PLUTUS is defined as an instance of a concrete class that inherits AbstractTask. Several helper functions are
defined in AbstractTask such that, for most common tasks, defining a task amounts to dispatching the helper functions. 

Data is passed around as dataframes. Slices are passed around as 1-dimensional numpy arrays. Since the pipeline passes
data between database, Dataframe, SystemDS, DT, and the dashboard, the whole pipeline requires recoding data to be
interpretable for humans, SystemDS, or the DT framework. This is done through several abstract methods defined below. 

If a task implementation deviates from the requirements, PLUTUS may exhibit undefined behavior. 
"""


class AbstractTask:
    @abstractmethod
    def __init__(self, train_source: AbstractSource, test_source: AbstractSource,
                 additional_sources: tp.List[AbstractSource]):
        self.initial_train = train_source.get_next_batch(batch_size=None)
        self.test = test_source.get_next_batch(batch_size=None)
        self.additional_sources = additional_sources

    @abstractmethod
    def clean_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Given some raw data that was returned directly from the various sources, and clean it into a DataFrame that is
        suitable for passing around and human-readable.
        """
        pass

    @abstractmethod
    def x_column_names(self) -> tp.Set[str]:
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

    def get_train_set(self) -> tp.Union[pd.DataFrame, None]:
        """
        Acquire and return the initial train set as a DataFrame that includes only the X and y variables.
        """
        return self.initial_train

    def get_test_set(self) -> tp.Union[pd.DataFrame, None]:
        """
        Acquire and return the initial test set as a DataFrame that includes only the X and y variables.
        """
        return self.test

    def get_additional_data(self, source_idx: int, batch_size: int) -> tp.Union[pd.DataFrame, None]:
        """
        Return additional data from specified source, or None if no additional data is available.
        """
        source = self.additional_sources[source_idx]
        if source.has_next():
            return self.additional_sources[source_idx].get_next_batch(batch_size=batch_size)
        else:
            return None

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
    def recode_slice_to_human_readable(self, slice_: npt.NDArray) -> tp.List[str]:
        """
        In Sliceline, a slice is represented as a vector of integers. This is not human-interpretable. Recode the slice
        to a list of str such that it can be read and interpreted.
        """
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
    def get_categorical_mapping_scheme(possible_values: tp.Set) -> tp.Tuple[tp.List, tp.Dict]:
        """
        Given a set of values that a categorical column can take, maps each possible value to a positive integer,
        starting from 1 up to n. Then, returns a mapping from integer to value, and value to integer.
        """
        int_to_value_list = list(possible_values)
        value_to_int_dict = {value: idx for value, idx in enumerate(int_to_value_list)}
        return int_to_value_list, value_to_int_dict

    @staticmethod
    def get_equi_width_bin_borders(min_value: float, max_value: float, num_bins: int) -> tp.List[float]:
        bin_width = (max_value - min_value) / num_bins
        bin_borders = [min_value + i * bin_width for i in range(num_bins + 1)]
        return bin_borders

    @staticmethod
    def get_equi_count_bin_borders(values: tp.List[float], num_bins: int) -> tp.List[float]:
        if len(values) == 0:
            return []
        values.sort()
        quantiles = [i / num_bins for i in range(1, num_bins)]
        bin_borders = [np.quantile(values, q) for q in quantiles]
        return [min(values)] + bin_borders + [max(values)]

    def __init__(self, train_source: AbstractSource, test_source: AbstractSource,
                 additional_sources: tp.List[AbstractSource], numeric_x_columns: tp.Set[str],
                 categorical_x_columns: tp.Set[str], y_column: str, y_is_categorical: bool,
                 binning_method: tp.Union[str, tp.Dict[tp.List[float]]]):
        """
        params:
            train_source: An AbstractSource for acquiring initial train set.
            test_source: An AbstractSource for acquiring initial test set.
            additional_sources: A list of AbstractSources for acquiring additional data.
            numeric_x_columns: A set of column names for numeric X features.
            categorical_x_columns: A set of column names for categorical X features.
            y_column: Name of y column.
            y_is_categorical: Whether y column is categorical or numeric.
            binning_method: Either "equi-width", "equi-count", or a list of predefined bin borders, which is a mapping
                            from numeric column names to list of n+1 monotonically increasing floats that define the
                            bin borders for the specified column.
        """
        super().__init__(train_source, test_source, additional_sources)
        # Store all the column names
        self.numeric_x_columns = numeric_x_columns
        self.categorical_x_columns = categorical_x_columns
        self.x_columns = set.union(numeric_x_columns, categorical_x_columns)
        self.y_column = y_column
        self.all_columns = self.x_columns.add(self.y_column)
        # Store other parameters
        self.y_is_categorical = y_is_categorical
        # Compute numeric bin borders, which will stay fixed
        self.binning_method = binning_method
        if isinstance(binning_method, dict):  # Numeric bin borders were predefined
            self.bin_borders = binning_method  # Copy predefined numeric bin borders
        elif binning_method == 'equi-width':
            self.bin_borders = dict()
            for col in list(self.numeric_x_columns):
                min_value = self.initial_train[col].min()
                max_value = self.initial_train[col].max()
                self.bin_borders.update({col: self.get_equi_width_bin_borders(min_value, max_value, )})
        elif binning_method == 'equi-count':
            pass
        else:
            raise ValueError('Unknown binning method.')

    def clean_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Project only the X and y columns
        df = data[list(self.all_columns)]
        return df

    def x_column_names(self) -> tp.Set[str]:
        return self.x_columns

    def y_column_name(self) -> str:
        return self.y_column

    def recode_for_model(self, dataset: pd.DataFrame) -> pd.DataFrame:
        # Dummy encode categorical X features
        df = pd.get_dummies(dataset, columns=list(self.categorical_x_columns))
        return df

    def binning_for_sliceline(self, dataset: pd.DataFrame) -> pd.DataFrame:
        pass

    def recode_slice_to_human_readable(self, slice_: npt.NDArray) -> tp.List[str]:
        pass
