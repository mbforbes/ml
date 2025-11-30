#!/bin/bash
mkdir -p data/mnist/resplit/
head -n 50000 data/mnist/original/mnist_train.csv > data/mnist/resplit/mnist_train.csv
tail -n 10000 data/mnist/original/mnist_train.csv > data/mnist/resplit/mnist_val.csv
