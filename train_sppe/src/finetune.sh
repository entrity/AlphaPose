#!/bin/bash

LR=${1:-1e-9}
python train.py \
  --dataset ucd \
  --trainBatch 8 --validBatch 8 \
  --loadModel ../../models/sppe/duc_se.pth \
  --snapshot 0 \
  --addDPG \
  --nClasses 33 \
  --LR $LR
