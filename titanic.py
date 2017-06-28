
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

training_data = "titanic_data/train.csv"
test_data = "titanic_data/test.csv"

COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
            "Ticket", "Fare", "Cabin", "Embarked"]
TEST_COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch",
            "Ticket", "Fare", "Cabin", "Embarked"]
FEATURES = ["Pclass", "SibSp", "Parch", "Fare"]
SPARSED = ["Sex", "Embarked"]
LABEL = "Survived"

gender = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="Sex", hash_bucket_size=10)
embarked = tf.contrib.layers.sparse_column_with_hash_bucket(column_name="Embarked", hash_bucket_size=10)

deep_columns = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]

data_set = pd.read_csv(training_data, skiprows=1, names=COLUMNS)

training_set = data_set[0:800].dropna()
test_set = data_set[801:891].dropna()

feature_cols = [tf.contrib.layers.real_valued_column(k)
                  for k in FEATURES]+ [gender , embarked]

import tempfile
model_dir = tempfile.mkdtemp()

regressor = tf.contrib.learn.DNNLinearCombinedClassifier(linear_feature_columns=feature_cols,
                                            dnn_feature_columns=deep_columns,
                                            dnn_hidden_units=[100, 50],
                                            model_dir="titanic_model")

def input_fn(data_set):
  continuous_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data_set[k].size)],
        values=data_set[k].values,
        dense_shape=[data_set[k].size, 1])     
                    for k in SPARSED}
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())

  labels = tf.constant(data_set[LABEL].values)
  return feature_cols, labels

def input_fn_fn(data_set):
  continuous_cols = {k: tf.constant(data_set[k].values)
                  for k in FEATURES}
  categorical_cols = {k: tf.SparseTensor(
        indices=[[i, 0] for i in range(data_set[k].size)],
        values=data_set[k].values,
        dense_shape=[data_set[k].size, 1])     
                    for k in SPARSED}
  feature_cols = dict(continuous_cols.items() + categorical_cols.items())

  #labels = tf.constant(data_set[LABEL].values)
  return feature_cols #, labels

regressor.fit(input_fn=lambda: input_fn(training_set), steps=5000)

results = regressor.evaluate(input_fn=lambda: input_fn(test_set), steps=1)
loss_score = results["loss"]
print("Loss: {0:f}".format(loss_score))

new_samples = pd.read_csv(test_data, skiprows=1, names=TEST_COLUMNS)
y = list(regressor.predict(input_fn=lambda: input_fn_fn(new_samples), as_iterable=True))

print(y)