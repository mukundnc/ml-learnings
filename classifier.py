
import tensorflow as tf
import numpy as np

training_data = "data/iris_training.csv"
test_data = "data/iris_test.csv"

training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=training_data,
    target_dtype=np.int,
    features_dtype=np.float32)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
    filename=test_data,
    target_dtype=np.int,
    features_dtype=np.float32)

feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="iris_model")

classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1000)
accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)
print(accuracy_score)

new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
y = list(classifier.predict(new_samples, as_iterable=True))
print(y)




