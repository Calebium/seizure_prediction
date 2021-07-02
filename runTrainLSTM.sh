#!/usr/bin/env bash

# bash runTrainLSTM.sh

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 0 -lr 0.001 -ep 1 -lg RodentSys -pw 1234 -fr RodentSys@gmail.com -to 4089300606@txt.att.net
