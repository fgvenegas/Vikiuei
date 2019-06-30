#!/bin/bash

python3 train.py --epochs 10 \
                 --batch_size 8 \
                 --print_every 10 \
                 --dataset miniCLEVR \
                 --val_questions_path /home/chrisams/Datasets/CLEVR_sample/questions/CLEVR_val_questions_sample.json \
                 --val_feats_dir /home/chrisams/Datasets/CLEVR_sample/regions-MiniCLEVR/regions-miniCLEVR-val \
                 --val_question_embeddings_path /home/chrisams/Datasets/CLEVR_sample/questions_val_glove_embeddings.npy \
                 --train_questions_path /home/chrisams/Datasets/CLEVR_sample/questions/CLEVR_train_questions_sample.json \
                 --train_feats_dir /home/chrisams/Datasets/CLEVR_sample/regions-MiniCLEVR/regions-miniCLEVR-train \
                 --train_question_embeddings_path /home/chrisams/Datasets/CLEVR_sample/questions_train_glove_embeddings.npy \