from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf

CSV_COLUMNS = ['peso', 'edad_madre', 'edad_padre', 'semanas_gestacion', 'ganancia_peso', 'puntuacion_apgar']
CSV_COLUMN_DEFAULTS = [0, 0, 0, 0, 0, 0]
LABEL_COLUMN = 'peso'

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('edad_madre'),
    tf.feature_column.numeric_column('edad_padre'),
    tf.feature_column.numeric_column('semanas_gestacion'),
    tf.feature_column.numeric_column('ganancia_peso'),
    tf.feature_column.numeric_column('puntuacion_apgar')
]

def build_estimator(output_dir):
    return tf.contrib.learn.LinearRegressor(model_dir=output_dir, feature_columns=INPUT_COLUMNS)

def csv_serving_input_fn():
  csv_row = tf.placeholder(
      shape=[None],
      dtype=tf.float32
  )
  features = parse_csv(csv_row)
  features.pop(LABEL_COLUMN)
  return tf.contrib.learn.InputFnOps(features, None, {'csv_row': csv_row})

SERVING_FUNCTIONS = {
    'CSV': csv_serving_input_fn
}

def parse_csv(rows_tensor):
  row_columns = tf.expand_dims(rows_tensor, -1)
  columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
  features = dict(zip(CSV_COLUMNS, columns))

  return features

def generate_input_fn(filenames,
                      num_epochs=None,
                      skip_header_lines=1,
                      batch_size=200):
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=True)
    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)

    _, rows = reader.read_up_to(filename_queue, num_records=batch_size)

    # Parse the CSV File
    features = parse_csv(rows)

    features = tf.train.shuffle_batch(
        features,
        batch_size,
        min_after_dequeue=2 * batch_size + 1,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )

    label = features.pop(LABEL_COLUMN)
    return features, label

