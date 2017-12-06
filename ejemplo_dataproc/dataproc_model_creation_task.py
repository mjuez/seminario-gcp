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

# Creacion de un modelo de machine learning (regresion linear)
# en Apache Spark ML.
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
# Este codigo se ejecuta como una tarea de Google Cloud Dataproc.
# 
# Este codigo es una adaptacion del siguiente ejemplo desarrollado
# por Google: 
#   
#   https://cloud.google.com/dataproc/docs/tutorials/bigquery-sparkml
# 
# Adaptacion realizada por Mario Juez <mario@mjuez.com>

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Dataset de entrenamiento
NATALITY_TRAIN_CSV = "gs://seminario-gcp/natality_train.csv"
# Dataset de test
NATALITY_TEST_CSV = "gs://seminario-gcp/natality_test.csv"

configuration = SparkConf().setAppName("natality-ml-regression")
spark = SparkSession.builder.config(conf=configuration).getOrCreate()

# Para evitar problemas, se eliminan todas aquellas filas
# que contengan valores nulos.
def clean_null(data):
  data.createOrReplaceTempView("data_view")
  query = """
    SELECT *
    FROM data_view
    WHERE peso is not null
          and edad_madre is not null
          and edad_padre is not null
          and semanas_gestacion is not null
  """
  clean_data = spark.sql(query)
  return clean_data

# Lectura del dataset desde un fichero CSV y conversion a dataframe.
def dataframe_from_csv(csv):
  readed_data = spark.read.csv(csv, header=True, mode="DROPMALFORMED", inferSchema=True)
  clean_data = clean_null(readed_data)
  df = clean_data.rdd.map(vector_from_inputs).toDF(["label","features"])
  df.cache()
  return df

# Separacion etiqueta-caracteristicas.
def vector_from_inputs(r):
  return (r["peso"], Vectors.dense(float(r["edad_madre"]),
                                            float(r["edad_padre"]),
                                            float(r["semanas_gestacion"]),
                                            float(r["ganancia_peso"]),
                                            float(r["puntuacion_apgar"])))

# Entrenamiento.
print ">>> Leyendo datos de entrenamiento..."
training_data = dataframe_from_csv(NATALITY_TRAIN_CSV)
print ">>> Comenzando entrenamiento..."
lr = LinearRegression(maxIter=10, regParam=0.2, solver="normal")
model = lr.fit(training_data)
print ">>> Entrenamiento terminado... RMSE:"+ str(model.summary.meanSquaredError)

# Prediccion del conjunto de test.
print ">>> Leyendo datos de test..."
testing_data = dataframe_from_csv(NATALITY_TEST_CSV)
print ">>> Realizando predicciones sobre conjunto de test..."
predictions = model.transform(testing_data)
print ">>> Predicciones terminadas..."
predictions.select("prediction", "label", "features").show(5)

# Evaluacion del modelo.
print ">>> Evaluando el modelo..."
evaluator = RegressionEvaluator()
rmse = evaluator.evaluate(predictions)
print ">>> Evaluacion terminada... RMSE:" + str(rmse)

# Almacenamiento del modelo para su uso posterior.
print ">>> Guardando el modelo..."
model.write().overwrite().save("gs://seminario-gcp/ml-models/pyspark-natality-lr-model")