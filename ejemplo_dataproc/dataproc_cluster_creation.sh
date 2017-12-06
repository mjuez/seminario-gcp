#!/bin/bash

# Creación del cluster Dataproc necesario para la creación del
# modelo de regresión linear del ejemplo.
#
#   * 1 Nodo maestro n1-standard-1 (3.75GB RAM) + 10GB HDD
#   * 2 Nodos trabajadores n1-standard-1 (3.75GB RAM) + 10GB HDD
#   * 5 Nodos trabajadores prioritarios (preemtive)
#   Máximos recursos posibles: 21GB RAM + 80GB HDD
#   Mínimos recursos asegurados: 11.25GB RAM + 30GB HDD
#
# Autor: Mario Juez <mario@mjuez.com>

gcloud dataproc --region europe-west1 clusters create dp-cluster-seminario \
--subnet default --zone europe-west1-b \
--master-machine-type n1-standard-1 --master-boot-disk-size 10 \
--num-workers 2 --worker-machine-type n1-standard-1 --worker-boot-disk-size 10 \
--num-preemptible-workers 5 --project seminario-gcp