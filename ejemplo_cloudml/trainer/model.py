#!/usr/bin/env python

# Copyright 2017 Mario Juez. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Creacion de un modelo de machine learning (regresion lineal)
# en Cloud ML Engine (TensorFlow)
#
# El modelo generado sirve para predecir el peso de un recien nacido
# en base a los siguientes valores:
#   * Edad de la madre
#   * Edad del padre
#   * Semanas de gestacion
#   * Ganancia de peso de la madre
#   * Puntuacion apgar
# 
# Los dataset de entrenamiento y test han sido obtenidos del conjunto
# de datos publico [publicdata:samples.natality] de BigQuery.
#
# Autor: Mario Juez <mario@mjuez.com>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing

import tensorflow as tf

# Columnas del fichero CSV.
CSV_COLUMNS = ['peso', 'edad_madre', 'edad_padre', 'semanas_gestacion', 'ganancia_peso', 'puntuacion_apgar']
# Valores por defecto.
CSV_COLUMN_DEFAULTS = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
# Etiqueta a predecir.
LABEL_COLUMN = 'peso'
# Trabajamos con caracteristicas numericas.
INPUT_COLUMNS = [
    tf.feature_column.numeric_column('edad_madre'),
    tf.feature_column.numeric_column('edad_padre'),
    tf.feature_column.numeric_column('semanas_gestacion'),
    tf.feature_column.numeric_column('ganancia_peso'),
    tf.feature_column.numeric_column('puntuacion_apgar')
]

def build_estimator(output_dir):
    """Construccion del estimador de regresion lineal"""
    return tf.contrib.learn.LinearRegressor(model_dir=output_dir, feature_columns=INPUT_COLUMNS)

def csv_serving_input_fn():
    """Definicion de la funcion de entrada"""
    csv_row = tf.placeholder(
        shape=[None],
        dtype=tf.string
    )
    features = parse_csv(csv_row)
    features.pop(LABEL_COLUMN)
    return tf.contrib.learn.InputFnOps(features, None, {'csv_row': csv_row})

# Solo ofrecemos la opcion de utilizar ficheros CSV
SERVING_FUNCTIONS = {
    'CSV': csv_serving_input_fn
}

def parse_csv(rows_tensor):
    """Lectura de caracteristicas desde CSV"""
    row_columns = tf.expand_dims(rows_tensor, -1)
    columns = tf.decode_csv(row_columns, record_defaults=CSV_COLUMN_DEFAULTS)
    features = dict(zip(CSV_COLUMNS, columns))
    return features

def generate_input_fn(filenames,
                      num_epochs=None,
                      skip_header_lines=1,
                      batch_size=200):
    """Creacion de la funcion de entrada"""
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=num_epochs, shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
    # Lectura de tantas filas como batch_size
    _, rows = reader.read_up_to(filename_queue, num_records=batch_size)
    # Obtencion de caracteristicas
    features = parse_csv(rows)
    features = tf.train.batch(
        features,
        batch_size,
        capacity=batch_size * 10,
        num_threads=multiprocessing.cpu_count(),
        enqueue_many=True,
        allow_smaller_final_batch=True
    )

    # Obtencion de etiqueta
    label = features.pop(LABEL_COLUMN)
    return features, label