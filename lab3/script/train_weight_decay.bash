#!/usr/bin/env bash

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model eegnet \
    --activation relu \
    --weight_decay 0.1 \
    --epochs 5000

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model eegnet \
    --activation leaky_relu \
    --weight_decay 0.1 \
    --epochs 5000

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model eegnet \
    --activation elu \
    --weight_decay 0.1 \
    --epochs 5000

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model deepconv \
    --activation relu \
    --weight_decay 0.1 \
    --epochs 5000

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model deepconv \
    --activation leaky_relu \
    --weight_decay 0.1 \
    --epochs 5000

python3 /home/user/nycu-deep-learning-2023-spring/lab3/src/main.py  \
    --model deepconv \
    --activation elu \
    --weight_decay 0.1 \
    --epochs 5000




