
import tensorflow as tf
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
      '--train-files',
      help='GCS location to write checkpoints and export models',
      required=True
  )
parser.add_argument(
    '--eval-files',
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models',
    required=True
)
parser.add_argument(
    '--module-name',
    help='GCS location to write checkpoints and export models',
    required=False
)
parser.add_argument(
    '--package-path',
    help='GCS location to write checkpoints and export models',
    required=False
)
parser.add_argument(
    '--runtime-version',
    help='GCS location to write checkpoints and export models',
    required=False
)
args = parser.parse_args()
arguments = args.__dict__

training_data = arguments.pop('train_files')
test_data = arguments.pop('eval_files')
model_dir=arguments.pop('job_dir')
v = tf.Variable(0, name='my_variable')

sess = tf.Session()
    
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
                                            model_dir=model_dir)


classifier.fit(x=training_set.data,
            y=training_set.target,
            steps=1000)
accuracy_score = classifier.evaluate(x=test_set.data,y=test_set.target)
#print(accuracy_score)

#new_samples = np.array([[6.4, 3.2, 4.5, 1.5], [5.8, 3.1, 5.0, 1.7]], dtype=float)
#y = list(classifier.predict(new_samples, as_iterable=True))
#print(y)
tf.train.write_graph(sess.graph, model_dir, 'saved_model.pbtxt')




