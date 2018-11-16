#!/bin/bash

SPREAD=$1

nohup python min_hole_3.py --act compile -n 0 -o spread${SPREAD} -s $SPREAD > nohup.spread${SPREAD} &