#!/bin/bash

# En este fichero se muestra como programar trabajos de entrenamiento
# así como desplegar los modelos entrenados o hacer predicciones con
# ellos en Cloud ML Engine.
#
# Autor: Mario Juez <mario@mjuez.com>

# Entrenamiento de un modelo de forma local 
# (recursos de la máquina donde se ejecuta el comando).
gcloud ml-engine local train \
    --job-dir "gs://seminario-gcp/ml-models/cloudml-natality-lr-model/$(date +"%Y%m%d_%H%M%S")" \
    --package-path ejemplo_cloudml/trainer/ \
    --module-name trainer.task \
    -- \
    --train-files "gs://seminario-gcp/natality_train.csv" \
    --eval-files "gs://seminario-gcp/natality_test.csv" \
    --num-epochs 10 \
    --eval-steps 10000

# Entrenamiento de un modelo de forma remota.
# Cloud ML se encarga de escalar con los recursos necesarios.
gcloud ml-engine jobs submit training "natality_regression_$(date +"%Y%m%d_%H%M%S")" \
    --job-dir "gs://seminario-gcp/ml-models/cloudml-natality-lr-model/$(date +"%Y%m%d_%H%M%S")" \
    --package-path ejemplo_cloudml/trainer/ \
    --module-name trainer.task \
    --region europe-west1 \
    --runtime-version 1.4 \
    -- \
    --train-files "gs://seminario-gcp/natality_train.csv" \
    --eval-files "gs://seminario-gcp/natality_test.csv" \
    --num-epochs 10 \
    --eval-steps 10000

# Creación de un modelo remoto. Alojará versiones de
# nuestro modelo para poder predecir remotamente.
gcloud ml-engine models create "natality_regression" \
    --regions="europe-west1-b" --enable-logging    
# Despliegue de una versión de nuestro modelo entrenado previamente.
gcloud ml-engine versions create v1 --model "natality_regression" \
    --origin "gs://seminario-gcp/ml-models/cloudml-natality-lr-model/20171208_154241/export/Servo/1512745231" \
    --runtime-version 1.4

# Predicción contra un modelo local (GCS).
gcloud ml-engine local predict \
    --model-dir="gs://seminario-gcp/ml-models/cloudml-natality-lr-model/20171208_154241/export/Servo/1512745231" \
    --text-instances="ejemplo_cloudml/tf_test_predict.csv"
# Predicción contra un modelo remoto (Cloud ML)
gcloud ml-engine predict --model "natality_regression" \
    --text-instances="ejemplo_cloudml/tf_test_predict.csv"

# Análisis del proceso de entrenamiento 
# con la interfaz gráfica tensorboard.
tensorboard --port 8080 --logdir gs://seminario-gcp/ml-models/cloudml-natality-lr-model/20171208_154241


