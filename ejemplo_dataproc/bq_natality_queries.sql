/*
 * Autor: Mario Juez <mario@mjuez.com>
 *
 * Consultas para obtener los conjuntos de entrenamiento
 * y test desde la base de datos pública de BigQuery.
 * 
 * Consultas desarrolladas a partir de las mostradas en
 * el siguiente ejemplo de Google:
 * https://cloud.google.com/dataproc/docs/tutorials/bigquery-sparkml
 *
 */

-- Dataset de entrenamiento 8.000.000 filas (80%)
SELECT
  peso,
  edad_madre,
  edad_padre,
  semanas_gestacion,
  ganancia_peso,
  puntuacion_apgar
FROM(
  SELECT 
    weight_pounds AS peso, 
    mother_age AS edad_madre, 
    father_age AS edad_padre, 
    gestation_weeks AS semanas_gestacion,
    weight_gain_pounds AS ganancia_peso, 
    apgar_5min AS puntuacion_apgar,
    rand(100) as rand
  FROM 
    [publicdata:samples.natality]
  WHERE 
    weight_pounds is not null
    and mother_age is not null 
    and father_age is not null
    and gestation_weeks is not null
    and weight_gain_pounds is not null
    and apgar_5min is not null
  ORDER BY rand
  LIMIT 8000000)

-- Dataset de test 2.000.000 filas (20%)
SELECT
  peso,
  edad_madre,
  edad_padre,
  semanas_gestacion,
  ganancia_peso,
  puntuacion_apgar
FROM(
  SELECT 
    weight_pounds AS peso, 
    mother_age AS edad_madre, 
    father_age AS edad_padre, 
    gestation_weeks AS semanas_gestacion,
    weight_gain_pounds AS ganancia_peso, 
    apgar_5min AS puntuacion_apgar,
    rand(100) as rand
  FROM 
    [publicdata:samples.natality]
  WHERE 
    weight_pounds is not null
    and mother_age is not null 
    and father_age is not null
    and gestation_weeks is not null
    and weight_gain_pounds is not null
    and apgar_5min is not null
  ORDER BY rand
  LIMIT 2000000)