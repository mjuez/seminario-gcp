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

# Creacion de una tarea de entrenamiento de Google Cloud ML.
#
# Tarea basada en el ejemplo Census de Cloud ML:
#   https://github.com/GoogleCloudPlatform/cloudml-samples/tree/master/census
# Autor: Mario Juez <mario@mjuez.com>

import argparse
import os

import model

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (
    saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam


def generate_experiment_fn(**experiment_args):
    """Creacion del experimento a correr en Cloud ML"""
    def _experiment_fn(run_config, hparams):
        """Definicion de experimento"""
        # Funcion de entrada de entrenamiento
        train_input = lambda: model.generate_input_fn(
            hparams.train_files,
            num_epochs=hparams.num_epochs,
            batch_size=hparams.train_batch_size,
        )
        # Funcion de entrada de evaluacion
        eval_input = lambda: model.generate_input_fn(
            hparams.eval_files,
            batch_size=hparams.eval_batch_size
        )
        # Experimento
        return tf.contrib.learn.Experiment(
            model.build_estimator(hparams.job_dir),
            train_input_fn=train_input,
            eval_input_fn=eval_input,
            **experiment_args
        )
    return _experiment_fn


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Argumentos de entrada
  parser.add_argument(
      '--train-files',
      help='Ficheros de entrenamiento.',
      nargs='+',
      required=True
  )
  parser.add_argument(
      '--num-epochs',
      help="""\
      Numero de veces que se recorre el dataset (epochs).\
      """,
      type=int,
      default=10
  )
  parser.add_argument(
      '--train-batch-size',
      help='Tamano de porciones del set de datos de entrenamiento.',
      type=int,
      default=800
  )
  parser.add_argument(
      '--eval-batch-size',
      help='Tamano de porciones del set de datos de evaluacion.',
      type=int,
      default=200
  )
  parser.add_argument(
      '--eval-files',
      help='Ficheros de evaluacion.',
      nargs='+',
      required=True
  )
  # Argumentos de entrenamiento
  parser.add_argument(
      '--job-dir',
      help='Ruta (GCS) donde escribir el modelo y los puntos de guardado.',
      required=True
  )

  # Argumento de logging
  parser.add_argument(
      '--verbosity',
      choices=[
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'
      ],
      default='INFO',
  )
  # Argumentos del experimento
  parser.add_argument(
      '--eval-delay-secs',
      help='Tiempo de espera hasta la primera evaluacion',
      default=10,
      type=int
  )
  parser.add_argument(
      '--min-eval-frequency',
      help='Numero minimo de pasos de entrenamiento entre evaluaciones',
      default=None,  # Por defecto 1000
      type=int
  )
  parser.add_argument(
      '--train-steps',
      help="""\
      Numero de pasos de entrenamiento.
      Se debe fijar si no se utiliza el numero de epochs.\
      """,
      type=int
  )
  parser.add_argument(
      '--eval-steps',
      help='Numero de pasos de evaluacion',
      default=100,
      type=int
  )
  parser.add_argument(
      '--export-format',
      help='Formato de exportacion.',
      choices=['CSV'],
      default='CSV'
  )

  args = parser.parse_args()

  # Configurando el nivel de logging.
  tf.logging.set_verbosity(args.verbosity)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # Ejecucion del entrenamiento.
  learn_runner.run(
      generate_experiment_fn(
          min_eval_frequency=args.min_eval_frequency,
          eval_delay_secs=args.eval_delay_secs,
          train_steps=args.train_steps,
          eval_steps=args.eval_steps,
          export_strategies=[saved_model_export_utils.make_export_strategy(
              model.SERVING_FUNCTIONS[args.export_format],
              exports_to_keep=1,
              default_output_alternative_key=None,
          )]
      ),
      run_config=run_config.RunConfig(model_dir=args.job_dir),
      hparams=hparam.HParams(**args.__dict__)
  )
