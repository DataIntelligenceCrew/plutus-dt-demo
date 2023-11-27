import pandas as pd
import numpy as np
import math
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
import statistics as sts
from sliceline.slicefinder import Slicefinder
import optbinning

train_df = pd.read_csv('data/housing0.csv')
train_y = train_df['median_house_value']
train_x = train_df.drop('median_house_value', axis=1)
train_x = pd.get_dummies(train_x, columns=['ocean_proximity'])

model = HistGradientBoostingRegressor(random_state=42)
model.fit(train_x, train_y)

test_df = pd.read_csv('data/housing1.csv')
test_y = train_df['median_house_value']
test_x = train_df.drop('median_house_value', axis=1)
test_x = pd.get_dummies(test_x, columns=['ocean_proximity'])

y_pred = model.predict(test_x)

training_errors = (test_y - y_pred)**2
print(training_errors)

mean_training_error = sts.mean(training_errors)
print(math.sqrt(mean_training_error))

optimal_binner = optbinning.ContinuousOptimalBinning(max_n_bins=5)

X_trans = pd.DataFrame(np.array(
    [
        optimal_binner.fit_transform(train_x[col], training_errors, metric="bins") for col in train_x.columns
    ]
).T, columns=train_x.columns)

# fitting sliceline
sf = Slicefinder(
    alpha = 0.95,
    k = 1,
    max_l = X_trans.shape[1],
    min_sup = 1,
    verbose = True
)

sf.fit(X_trans, training_errors)

slices_df = pd.DataFrame(sf.top_slices_, columns=sf.feature_names_in_, index=sf.get_feature_names_out())
print(slices_df)