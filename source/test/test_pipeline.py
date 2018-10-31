import unittest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from source.code.estimators.customclassifier import CustomClassifier
from source.code.transformers.customlabelencoder import CustomLabelEncoder
from source.code.transformers.customlabelbinarizer import CustomLabelBinarizer

numerical_features = [
    'age',
    'campaign',
    'cons.conf.idx',
    'cons.price.idx',
    'duration',
    'emp.var.rate',
    'euribor3m',
    'nr.employed',
    'pdays',
    'previous'
]

categorical_features = [
    'contact',
    'day_of_week',
    'default',
    'education',
    'housing',
    'job',
    'loan',
    'marital',
    'month',
    'poutcome'
]

target = 'y'

data = pd.read_csv('../../data/datasets/TS_Summer_2018/data.csv', sep=';')

columns_with_gaps = data.columns[:-1]

minimum = 0
maximum = 0.3

columns_with_gaps_dict = dict(
    zip(
        columns_with_gaps,
        np.random.uniform(
            minimum,
            maximum,
            len(columns_with_gaps)
        )
    )
)

data_with_gaps = data.copy()

for column in columns_with_gaps:
    if columns_with_gaps_dict[column] > 0:
        gaps_count = int(
            len(data_with_gaps) * columns_with_gaps_dict[column]
        )
        data_with_gaps[column].iloc[
            np.random.randint(
                0,
                len(data_with_gaps),
                gaps_count
            )
        ] = np.nan

X, y = data_with_gaps[numerical_features + categorical_features], data_with_gaps[target]

y = CustomLabelBinarizer().fit_transform(X, CustomLabelEncoder().fit_transform(X, y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


class TestPipeline(unittest.TestCase):

    def test_case_1(self):
        num_features_pipeline = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('scale', MinMaxScaler()),
            ('transform', QuantileTransformer(output_distribution='normal'))
        ])

        cat_features_pipeline = Pipeline([
            ('impute', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_features_pipeline, numerical_features),
                ('cat', cat_features_pipeline, categorical_features)
            ]
        )

        classifier_pipeline = Pipeline(
            steps=[
                ('preprocessing', preprocessor),
                ('classify', CustomClassifier(base=LogisticRegression()))
            ]
        )

        y_pred = classifier_pipeline.fit_predict(X_train, y_train)

        print(y_pred)


if __name__ == '__main__':
    unittest.main()
