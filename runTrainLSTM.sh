#!/usr/bin/env bash

# bash runTrainLSTM.sh

#python scrTrainLSTM.py -dp '../EEG_Data/Kaggle_Prediction/Patient_1/TrainingData/' -fn 'Patient_1_*_segment_000[1|2].mat' -st 100000 -bs 3 -gpu 0 -hd 256 -nl 4 -os 2 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# NOTE: Large fluctuations in training loss and little convergence
#python scrTrainLSTM.py -dp '../EEG_Data/Kaggle_Prediction/CombinedTrainingData/' -fn 'Patient_1_*_segment_*.mat' -st 100000 -bs 3 -gpu 0 -hd 256 -nl 4 -os 2 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# RuntimeError: CUDA out of memory. Tried to allocate 88.10 GiB (GPU 0; 10.76 GiB total capacity; 8.00GiB already allocated; 1.94 GiB free; 4.51 MiB cached) -> Looks like if -bs > 3 nnd -hd > 256 will lead to out of memory error (should probably downsample)
#python scrTrainLSTM.py -csv './TrainingData.csv' -st 100000 -bs 10 -gpu 0 -hd 1024 -nl 4 -os 2 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Downsample to 500Hz and set subsequence duration to 20s to see if training using segments 0001 and 0002 give the same results as before
#python scrTrainLSTM.py -csv './TrainingData_0001-0002.csv' -rf 500 -du 20 -bs 3 -gpu 0 -hd 256 -nl 4 -os 2 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Downsample to 500Hz and set subsequence duration to 20s, and train using segments 0001/0002/0007/0008. Predicted using all the other segments in the same time period, and the prediction accuracy is slightly lower than if using 0001/0002 alone. Also, the training loss converges much quicker than the validation loss (which fluctuates without ever converging). This looks like overfitting
# Log file = runTrainLSTM_20190826-113444.log
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 20 -bs 3 -gpu 0 -hd 256 -nl 4 -os 2 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Based on the finding from the previous training, decrease the size of the network to increase regularization and decrease overfitting
# Log file = runTrainLSTM_20190826-175430.log
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 20 -bs 3 -gpu 0 -hd 256 -nl 2 -os 2 -dr 0.75 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Validation loss still fluctuating in the previous training (training loss converges). Decreasing the network size even further and increase the batch size
# Log file = runTrainLSTM_20190827-123210.log
# Log file = runTrainLSTM_20190827-154343.log (increased training set size)
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 20 -bs 5 -gpu 0 -hd 128 -nl 2 -os 2 -dr 0.75 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Validation loss still fluctuating in the previous trainings (training loss converges). Increasing batch size to 10
# Log file = runTrainLSTM_20190827-160019.log (now both training and validation losses fluctuate)
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 20 -bs 10 -gpu 0 -hd 128 -nl 2 -os 2 -dr 0.75 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Log file = runTrainLSTM_20190827-160918.log
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 1000 -du 30 -bs 5 -gpu 0 -hd 512 -nl 3 -os 2 -dr 0.5 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Log file = runTrainLSTM_20190827-173815.log
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 30 -bs 5 -gpu 0 -hd 256 -nl 3 -os 2 -dr 0.5 -lr 0.0005 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# Train using interictal + preictal 0001-0005 and 0007-0011 after fixing the bug regarding arrAllData[]'s type
# Log file = runTrainLSTM_20190830-040627.log
#python scrTrainLSTM.py -csv './TrainingData.csv' -rf 500 -du 20 -bs 6 -gpu 0 -hd 256 -nl 3 -os 2 -dr 0.5 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation
# Log file = runTrainLSTM_20191022-035911.log
# Log file = runTrainLSTM_20191022-032102.log
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 10 -bs 5 -gpu 0 -hd 256 -nl 3 -os 3 -dr 0.5 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (all seizure files
#          were used)
# Log file = runTrainLSTM_20191022-043608.log
# Result = Test accuracy > 0.99
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 10 -bs 10 -gpu 0 -hd 256 -nl 4 -os 3 -dr 0.5 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191022-165728.log
# Result = Test accuracy only about 0.5-0.6 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 10 -bs 10 -gpu 0 -hd 256 -nl 4 -os 3 -dr 0.5 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191023-010401.log
# Result = Test accuracy only about 0.5-0.6 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 10 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191023-012705.log
# Result = After changing -du to 1, test accuracy improved to ~0.87 when predicting seizures from
#          later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 1 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb03 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191023-034512.log
# Result = Test accuracy = ~0.76 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb03.csv' -rf -1 -du 1 -bs 32 -gpu 1 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb05 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191023-164324.log
# Result = Test accuracy = ~0.87 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb05.csv' -rf -1 -du 1 -bs 32 -gpu 1 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191023-172637.log
# Result = Test accuracy = ~0.78 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -rf -1 -du 1 -bs 32 -gpu 1 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191024-194526.log
# Result = Test accuracy = ~0.70 when predicting seizures from later times
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -rf -1 -du 1 -bs 32 -gpu 1 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20191027-092723.log
# Result = Test accuracy = ~0.50 when predicting seizures from later times (from training loss it
#          looks like the training has not yet converged)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -gpu 1 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200408-024910.log
# Result = Test accuracy ~0.53 using original annotation for chb24_21.edf. Improved to ~0.63 when
#          using the shorter seizure duration. The main reason for low accuracy is mostly due to
#          high false negatives
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200414-043046.log
# Result = Validation loss flucturates greatly, and is higher than training loss -> under-fitting?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200414-055244.log
# Result = Test accuracy ~0.48 even though both training and validation losses seem to converge
#          during the end of the training -> over-fitting?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 64 -gpu 0 -hd 512 -nl 3 -os 3 -dr 0.5 -lr 0.0005 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200415-041500.log
# Result = Test accuracy ~0.66. Training loss seem to fluctuate till the end while validation loss
#          appears to converge nicely. Very high false negative rate
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 1 -os 3 -dr 0.5 -lr 0.0005 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200415-055019.log
# Result = Trained with more data (from chb24_01.edf to chb24_20.edf) but forgot to increase the
#          number of epochs. Training and validation curves look unconverged
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200415-134216.log
# Result = Trained with more data (from chb24_01.edf to chb24_20.edf) and increased the number of
#          epochs from 1 to 5. However, training and validation curves still look unconverged
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
# Log file = runTrainLSTM_20200416-180313.log
# Result = Trained with more data (from chb24_01.edf to chb24_20.edf) and increased the number of
#          epochs from 1 to 10. However, validation loss keeps increasing as training loss
#          converges -> over-fitting? Some suggested maybe the validation set is too small
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -ss 80 -sw 0.3 -bs 32 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_09) and
#          chb24_10 -> chb24_12 for testing (also aggressively annotated)
#          Sliding window not used (but used sliding window function). Used a larger validation
#          set, dropout rate of 0.5, and lower learning rate
# Log file = runTrainLSTM_20200421-175424.log
# Result = Test accuracy ~0.77 if using chb24_10 -> chb24_12 (aggressively annotated) as test set.
#          Very low false positives very low. False positves still high (only half of the ictal
#          subseqs are predicted as ictal
#          Training and validation curves seemed to converge with small fluctuation in
#          validation losses. Training losses continue to decrease while validation losses are low
#          but consistently higher than training losses -> potential over-fitting?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -lr 0.0001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.75, and lower learning rate
# Log file = runTrainLSTM_20200423-023252.log
# Result = Test accuracy ~0.99 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.79 if using chb24_10 -> chb24_12 (aggressively annotated) as test set
#          Test accuracy ~0.62 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.0001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.75, and lower learning rate
#          Used AdamW as optimizer
# Log file = runTrainLSTM_20200423-033730.log
# Result = Test accuracy ~0.99 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.84 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 68, intNumFalseNegatives_NoAug = 12, intNumTruePositives_NoAug = 26, intNumTrueNegatives_NoAug_NoAug = 10672)
#          Test accuracy ~0.66 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 111, intNumFalseNegatives_NoAug = 128, intNumTruePositives_NoAug = 66, intNumTrueNegatives_NoAug_NoAug = 33148)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -lr 0.0001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.5, and lower learning rate
#          Used SGD as optimizer
# Log file = runTrainLSTM_20200423-051742.log
# Result = Training curves are not converged
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -lr 0.0005 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.5, and trained for 10 epochs
#          Used SGD as optimizer
# Log file = runTrainLSTM_20200423-055556.log
# Result = Training curves are starting to converge (showing elbow). Training losses are noisy
#          while validation losses seem smoother
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.5, and trained for 30 epochs
#          Used SGD as optimizer
# Log file = runTrainLSTM_20200423-120104.log
# Result = Training curves seem smoothly converged. However:
#          Test accuracy ~0.77 if using chb24_10 -> chb24_12 (aggressively annotated) as test set
#          (not as high as AdamW)
#          Test accuracy ~0.61 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (only 1/3 of ictal subseqs are predicted correctly)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -lr 0.001 -ep 30 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function). Used a larger batch size,
#          larger validation set, dropout rate of 0.5, and trained for 30 epochs
#          Used SGD as optimizer (added weight_decay = 0.001 for L2 regularization)
# Log file = runTrainLSTM_20200423-152340.log
# Result = Training curves seem smoothly converged. However:
#          Test accuracy ~0.77 if using chb24_10 -> chb24_12 (aggressively annotated) as test set
#          (not as high as AdamW) -> same result as not setting weight_decay
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 64 -vf 0.3 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -lr 0.001 -ep 30 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200501-192207.log
# Result = Test accuracy ~0.99 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.83 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 56, intNumFalseNegatives_NoAug = 13, intNumTruePositives_NoAug = 25, intNumTrueNegatives_NoAug_NoAug = 10698) -> similar to no scaling
#          Test accuracy ~0.85 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 3310, intNumFalseNegatives_NoAug = 36, intNumTruePositives_NoAug = 158, intNumTrueNegatives_NoAug_NoAug = 29954) -> ~20% improvement over no scaling!
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200502-023015.log
# Result = Test accuracy ~0.99 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.91 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 144, intNumFalseNegatives_NoAug = 7, intNumTruePositives_NoAug = 31, intNumTrueNegatives_NoAug_NoAug = 10611) -> slightly better than smod = 0
#          Test accuracy ~0.82 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 195, intNumFalseNegatives_NoAug = 68, intNumTruePositives_NoAug = 126, intNumTrueNegatives_NoAug_NoAug = 33069) -> slightly worse than smod = 0
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200502-030026.log
# Result = Test accuracy ~0.99 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.63 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 66, intNumFalseNegatives_NoAug = 28, intNumTruePositives_NoAug = 10, intNumTrueNegatives_NoAug_NoAug = 10691) -> worse than smod = 0 and 1 (very high false negatives)
#          Test accuracy ~0.73 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 511, intNumFalseNegatives_NoAug = 100, intNumTruePositives_NoAug = 94, intNumTrueNegatives_NoAug_NoAug = 32751) -> pretty high false negatives
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200502-035839.log
# Result = Wierd looking training and validation curves (like a step function)
#          Test accuracy ~0.97 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.67 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 6915, intNumFalseNegatives_NoAug = 0, intNumTruePositives_NoAug = 38, intNumTrueNegatives_NoAug_NoAug = 3837) -> high false positives, but 0 false negatives!!
#          Test accuracy ~0.83 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 3546, intNumFalseNegatives_NoAug = 46, intNumTruePositives_NoAug = 148, intNumTrueNegatives_NoAug_NoAug = 29718) -> less false neagtives than smod = 0
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 10 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used SGD as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200503-044707.log
# Result = Wierd looking training and validation curves (like a step function)
#          Test accuracy ~0.98 if using chb24_01 -> chb24_09 (aggressively annotated) as test set
#          Test accuracy ~0.79 if using chb24_10 -> chb24_12 (aggressively annotated) as test set (intNumFalsePositives_NoAug = 49, intNumFalseNegatives_NoAug = 16, intNumTruePositives_NoAug = 22, intNumTrueNegatives_NoAug_NoAug = 10703) -> similar to AdamW, smod = 0
#          Test accuracy ~0.80 if using chb24_13 -> chb24_22 (not aggressively annotated) as test
#          set (intNumFalsePositives_NoAug = 3238, intNumFalseNegatives_NoAug = 54, intNumTruePositives_NoAug = 140, intNumTrueNegatives_NoAug_NoAug = 30031)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 2 -lr 0.001 -ep 100 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used SGD as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200504-032635.log
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 2 -lr 0.001 -ep 100 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb24 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used aggressively trimmed ictal annotations for training (chn24_01 -> chb24_12)
#          Sliding window not used (use non-sliding window function)
#          Used SGD as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200506-101720.log
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb24.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 2 -lr 0.001 -ep 100 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb03 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200513-133344.log
# Result = Trainng and validation curves look like step function
#          Test accuracy ~0.94 (~0.76) if using chb03_20 -> chb03_38 (intNumFalsePositives_NoAug = 1571, intNumFalseNegatives_NoAug = 17, intNumTruePositives_NoAug = 147, intNumTrueNegatives_NoAug_NoAug = 66659)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb03.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb05 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200513-142952.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.97 (~0.87) if using chb05_17 -> chb05_39 (intNumFalsePositives_NoAug = 3749, intNumFalseNegatives_NoAug = 6, intNumTruePositives_NoAug = 231, intNumTrueNegatives_NoAug_NoAug = 78801)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb05.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200513-152951.log
# Result = Training and validation curves flat (test accuracy in training set ~0.49) -> something wrong?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb10 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200513-163709.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.94 (~0.8x -> ran by Jack) if using chb10_31 -> chb10_89 (intNumFalsePositives_NoAug = 128, intNumFalseNegatives_NoAug = 24, intNumTruePositives_NoAug = 195, intNumTrueNegatives_NoAug_NoAug = 21270)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb10.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200514-014724.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.65 (~0.9x -> ran by Jack) if using chb01_21 -> chb01_46 (intNumFalsePositives_NoAug = 369, intNumFalseNegatives_NoAug = 133, intNumTruePositives_NoAug = 61, intNumTrueNegatives_NoAug_NoAug = 74355)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb02 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200514-050859.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.98 (~0.7x -> ran by Jack) if using chb02_19 -> chb02_35 (intNumFalsePositives_NoAug = 1859, intNumFalseNegatives_NoAug = 0, intNumTruePositives_NoAug = 9, intNumTrueNegatives_NoAug_NoAug = 59319)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb02.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb06 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200514-063330.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb06_16 -> chb06_24 (intNumFalsePositives_NoAug = 1, intNumFalseNegatives_NoAug = 28, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 39729)
#          ***NO TRUE POSITIVES!
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb06.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200514-145601.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.80 (~0.5x -> ran by Jack) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 955, intNumFalseNegatives_NoAug = 57, intNumTruePositives_NoAug = 86, intNumTrueNegatives_NoAug_NoAug = 39758)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb23 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200514-173152.log
# Result = Training and validation curves look like step function and *not yet converged*
#          Test accuracy ~0.90 (~0.9x -> ran by Jack) if using chb23_09 -> chb23_20 (intNumFalsePositives_NoAug = 11720, intNumFalseNegatives_NoAug = 8, intNumTruePositives_NoAug = 236, intNumTrueNegatives_NoAug_NoAug = 63250)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb23.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200515-044839.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.65 (~0.6x -> ran by Jack) if using chb14_17 -> chb14_42 (intNumFalsePositives_NoAug = 422, intNumFalseNegatives_NoAug = 40, intNumTruePositives_NoAug = 18, intNumTrueNegatives_NoAug_NoAug = 53518)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200515-061448.log
# Result = Training and validation curves look flat (not yet converged)
#          Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (intNumFalsePositives_NoAug = 9, intNumFalseNegatives_NoAug = 74, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 52476)
#          ***NO TRUE POSITIVES!
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb21 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200516-000309.log
# Result = Training and validation curves look like square wave (converges twice!)
#          Test accuracy ~0.58 (~0.7x -> ran by Jack) if using chb21_22 -> chb21_33 (intNumFalsePositives_NoAug = 273, intNumFalseNegatives_NoAug = 10, intNumTruePositives_NoAug = 2, intNumTrueNegatives_NoAug_NoAug = 42907)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb21.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb22 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200516-061210.log
# Result = Test accuracy ~0.85 (~0.7x -> ran by Jack) if using chb22_28 -> chb22_77 (intNumFalsePositives_NoAug = 4027, intNumFalseNegatives_NoAug = 12, intNumTruePositives_NoAug = 60, intNumTrueNegatives_NoAug_NoAug = 21094)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb22.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200520-144105.log
# Result = Test accuracy ~0.73 (~0.9x -> ran by Jack) if using chb01_21 -> chb01_46 (intNumFalsePositives_NoAug = 758, intNumFalseNegatives_NoAug = 105, intNumTruePositives_NoAug = 89, intNumTrueNegatives_NoAug_NoAug = 73965)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200520-160648.log
# Result = Big spike in the elbow of the training and validation curves
#          Test accuracy ~0.81 (~0.9x -> ran by Jack) if using chb01_21 -> chb01_46 (intNumFalsePositives_NoAug = 728, intNumFalseNegatives_NoAug = 73, intNumTruePositives_NoAug = 121, intNumTrueNegatives_NoAug_NoAug = 73998)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb01 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200520-225201.log
# Result = Training and validation curves look like step function
#          Test accuracy ~0.91 (~0.9x -> ran by Jack) if using chb01_21 -> chb01_46 (intNumFalsePositives_NoAug = 643, intNumFalseNegatives_NoAug = 32, intNumTruePositives_NoAug = 162, intNumTrueNegatives_NoAug_NoAug = 74082)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb01.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb06 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200521-042124.log
# Result = Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb06_16 -> chb06_24 (intNumFalsePositives_NoAug = 10, intNumFalseNegatives_NoAug = 28, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 39719)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb06.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb06 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200521-064936.log
# Result = Test accuracy ~0.57 (~0.7x -> ran by Jack) if using chb06_16 -> chb06_24 (intNumFalsePositives_NoAug = 287, intNumFalseNegatives_NoAug = 25, intNumTruePositives_NoAug = 3, intNumTrueNegatives_NoAug_NoAug = 39442)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb06.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb06 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200521-163657.log
# Result = Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb06_16 -> chb06_24 (intNumFalsePositives_NoAug = 0, intNumFalseNegatives_NoAug = 28, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 39729)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb06.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200524-015721.log
# Results = Big bump in the middle of training and validation curves
#           Test accuracy ~0.63 (~0.6x -> ran by Jack) if using chb14_17 -> chb14_42 (intNumFalsePositives_NoAug = 2061, intNumFalseNegatives_NoAug = 41, intNumTruePositives_NoAug = 17, intNumTrueNegatives_NoAug_NoAug = 51878)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200524-022711.log
# Result = Test accuracy ~0.76 (~0.6x -> ran by Jack) if using chb14_17 -> chb14_42 (intNumFalsePositives_NoAug = 13836, intNumFalseNegatives_NoAug = 13, intNumTruePositives_NoAug = 45, intNumTrueNegatives_NoAug_NoAug = 40104)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200524-030446.log
# Result = Training and validation curves stayed flat for a long time, not yet fully converged. May need to train longer or use a higher learning rate
#          Test accuracy ~0.47 (~0.6x -> ran by Jack) if using chb14_16 -> chb14_42 (intNumFalsePositives_NoAug = 4776, intNumFalseNegatives_NoAug = 56, intNumTruePositives_NoAug = 2, intNumTrueNegatives_NoAug_NoAug = 49165)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200524-041106.log
# Result = Test accuracy ~0.53 (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (intNumFalsePositives_NoAug = 7423, intNumFalseNegatives_NoAug = 60, intNumTruePositives_NoAug = 14, intNumTrueNegatives_NoAug_NoAug = 45061)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200524-044617.log
# Result = Training and validation curves converging very slowly, and have not fully converged at the end of the training
#          Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (intNumFalsePositives_NoAug = 5824, intNumFalseNegatives_NoAug = 65, intNumTruePositives_NoAug = 9, intNumTrueNegatives_NoAug_NoAug = 46661)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200524-070109.log
# Result = Both training and validation curves are flat
#          Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (intNumFalsePositives_NoAug = 52484, intNumFalseNegatives_NoAug = 0, intNumTruePositives_NoAug = 74, intNumTrueNegatives_NoAug_NoAug = 1)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb21 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200525-004650.log
# Result = Both training and validation curves are quite flat
#          Test accuracy ~0.49 (~0.7x -> ran by Jack) if using chb21_22 -> chb21_33 (intNumFalsePositives_NoAug = 867, intNumFalseNegatives_NoAug = 12, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 42311)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb21.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb21 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200525-063334.log
# Result = Test accuracy ~0.92 (~0.7x -> ran by Jack) if using chb21_22 -> chb21_33 (intNumFalsePositives_NoAug = 3606, intNumFalseNegatives_NoAug = 1, intNumTruePositives_NoAug = 11, intNumTrueNegatives_NoAug_NoAug = 39570)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb21.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb21 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3
# Log file = runTrainLSTM_20200527-041839.log
# Result = One huge spike near the end of the validation curve
#          Test accuracy ~0.50 (~0.7x -> ran by Jack) if using chb21_22 -> chb21_33 (intNumFalsePositives_NoAug = 43034, intNumFalseNegatives_NoAug = 0, intNumTruePositives_NoAug = 12, intNumTrueNegatives_NoAug_NoAug = 146)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb21.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb04 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200531-052924.log
# Results = Test accuracy ~0.50 (~0.5x -> ran by Jack) if using chb04_22 -> chb04_43 (intNumFalsePositives_NoAug = 218, intNumFalseNegatives_NoAug = 217, intNumTruePositives_NoAug = 1, intNumTrueNegatives_NoAug_NoAug = 295332)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb04.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb11 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200601-032856.log
# Results = Test accuracy ~0.51 (~0.6x -> ran by Jack) if using chb11_17 -> chb11_99 (intNumFalsePositives_NoAug = 57855, intNumFalseNegatives_NoAug = 3, intNumTruePositives_NoAug = 749, intNumTrueNegatives_NoAug_NoAug = 1836) -> great ictal prediction, but very poor interictal prediction
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb09.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200601-045712.log
# Results = Completely flat training and validation curves, retry
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb16 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200601-072839.log
# Results = Test accuracy ~0.50 (~0.5x -> ran by Jack) if using chb16_17 -> chb16_17 (intNumFalsePositives_NoAug = 4, intNumFalseNegatives_NoAug = 31, intNumTruePositives_NoAug = 0, intNumTrueNegatives_NoAug_NoAug = 3550) -> high interictal accuracy, zero ictal accuracy!
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb16.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb17 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200601-172000.log
# Results = Test accuracy ~0.56 (~0.5x -> ran by Jack) if using chb17b_63 -> chb17c_08 (intNumFalsePositives_NoAug = 1254, intNumFalseNegatives_NoAug = 73, intNumTruePositives_NoAug = 15, intNumTrueNegatives_NoAug_NoAug = 38282) -> poor ictal prediction
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb17.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb18 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200601-183611.log
# Results = Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb18.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb19 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Sliding window not used (use non-sliding window function)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
# Log file = runTrainLSTM_20200602-011457.log
# Results = Test accuracy ~0.80 (~0.7x -> ran by Jack) if using chb19_30 -> chb19_30 (intNumFalsePositives_NoAug = 7, intNumFalseNegatives_NoAug = 33, intNumTruePositives_NoAug = 48, intNumTrueNegatives_NoAug_NoAug = 3241)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb19.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb21 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window (for testing) and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2
# Log file = runTrainLSTM_20200604-031051.log
# Result = Test accuracy ~0.95 (~0.7x -> ran by Jack) if using chb21_22 -> chb21_33 (intNumFalsePositives_NoAug = 4368, intNumFalseNegatives_NoAug = 1, intNumTruePositives_NoAug = 11, intNumTrueNegatives_NoAug_NoAug = 38820) -> Results very close to using non-sliding window function
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb21.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 2}
# Log file = runTrainLSTM_20200604-061121.log
# Result = Test accuracy ~0.72 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 22845, intNumFalseNegatives_NoAug = 1238, intNumTruePositives_NoAug = 17002, intNumTrueNegatives_NoAug_NoAug = 17947) -> much better ictal prediction accuracy, but interictal prediction accuracy decreased (maybe increase network capacity or drop some interictal subsequences?)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 32}
# Log file = runTrainLSTM_20200605-180432.log
# Result = Test accuracy ~0.87 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 1437, intNumFalseNegatives_NoAug = 270, intNumTruePositives_NoAug = 870, intNumTrueNegatives_NoAug_NoAug = 39295)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 32'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 16}
# Log file = runTrainLSTM_20200606-035551.log
# Result = Test accuracy ~0.89 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 1898, intNumFalseNegatives_NoAug = 375, intNumTruePositives_NoAug = 1905, intNumTrueNegatives_NoAug_NoAug = 38824)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 16'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1  {'ictal': 8}
# Log file = runTrainLSTM_20200606-065701.log
# Results = Low test accuracy during training (71%), training curves look like periodic step function (no convergence) -> test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 8'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 32}
#          For testing after code change involving normalization across training and test sets
# Log file = runTrainLSTM_20200606-144219.log
# Result = Test accuracy ~0.88 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 980, intNumFalseNegatives_NoAug = 236, intNumTruePositives_NoAug = 904, intNumTrueNegatives_NoAug_NoAug = 39752)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 32'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 4}
# Log file = runTrainLSTM_20200606-193415.log
# Result = Test accuracy ~0.85 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 5547, intNumFalseNegatives_NoAug = 1507, intNumTruePositives_NoAug = 7613, intNumTrueNegatives_NoAug_NoAug = 35199)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 4'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 64}
# Log file = runTrainLSTM_20200607-075317.log
# Result = Test accuracy ~0.88 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 1240, intNumFalseNegatives_NoAug = 119, intNumTruePositives_NoAug = 451, intNumTrueNegatives_NoAug_NoAug = 39485)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 64'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 128}
# Log file = runTrainLSTM_20200607-154348.log
# Result = Test accuracy ~0.87 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 2565, intNumFalseNegatives_NoAug = 57, intNumTruePositives_NoAug = 228, intNumTrueNegatives_NoAug_NoAug = 38158)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 128'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 {'ictal': 2}
#          Used larger network (-hd 512 -nl 3)
# Log file = runTrainLSTM_20200604-170545.log
# Result = Training and validation curves completely flat. Test set not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 512 -nl 3 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1
#          Set output size to 2 and changed dctSegStates[] to set ictal to 1 and preictal to 2 (code broken)
# Log file = 
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 2 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200608-045240.log
# Result = Test accuracy ~0.85 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 7, intNumFalseNegatives_NoAug = 44, intNumTruePositives_NoAug = 99, intNumTrueNegatives_NoAug_NoAug = 40707)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb22 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200609-011644.log
# Result = Test accuracy ~0.86 (~0.7x -> ran by Jack) if using chb22_28 -> chb22_77 (intNumFalsePositives_NoAug = 27, intNumFalseNegatives_NoAug = 21, intNumTruePositives_NoAug = 51, intNumTrueNegatives_NoAug_NoAug = 25093) -> very good interictal prediction, good ictal prediction (slightly less good than smod = 1 without tupScalingInfo)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb22.csv' -tcsv './DataCSVs/CHB-MIT/chb22_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb04 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file =  runTrainLSTM_20200609-051908.log
# Results = Test accuracy ~0.60 (~0.5x -> ran by Jack) if using chb04_22 -> chb04_43 (intNumFalsePositives_NoAug = 3942, intNumFalseNegatives_NoAug = 170, intNumTruePositives_NoAug = 48, intNumTrueNegatives_NoAug_NoAug = 291604)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb04.csv' -tcsv './DataCSVs/CHB-MIT/chb04_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200609-164419.log
# Result = Training and validation curves flat (test accuracy in training set ~0.51) -> same as in previous training (without tupScalingInfo). Definitely needs further investigation
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -tcsv './DataCSVs/CHB-MIT/chb08_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb09 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200609-171237.log
# Results = Test accuracy ~0.96 (~0.9x -> ran by Jack) if using chb09_10 -> chb09_19 (intNumFalsePositives_NoAug = 92, intNumFalseNegatives_NoAug = 5, intNumTruePositives_NoAug = 57, intNumTrueNegatives_NoAug_NoAug = 120940)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb09.csv' -tcsv './DataCSVs/CHB-MIT/chb09_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb11 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200609-230926.log
# Results = Convergence didn't happen until epoch 8, may need to train longer
#           Test accuracy ~0.96 (~0.6x -> ran by Jack) if using chb11_17 -> chb11_99 (intNumFalsePositives_NoAug = 2029, intNumFalseNegatives_NoAug = 30, intNumTruePositives_NoAug = 722, intNumTrueNegatives_NoAug_NoAug = 57663)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb11.csv' -tcsv './DataCSVs/CHB-MIT/chb11_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb14 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-013413.log
# Result = Test accuracy ~0.86 (~0.6x -> ran by Jack) if using chb14_17 -> chb14_42 (intNumFalsePositives_NoAug = 1730, intNumFalseNegatives_NoAug = 15, intNumTruePositives_NoAug = 43, intNumTrueNegatives_NoAug_NoAug = 52211)
#          Convergence started to happen at epoch 8 -> maybe train longer and try sliding window
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb14.csv' -tcsv './DataCSVs/CHB-MIT/chb14_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-020730.log
# Results = Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb16 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-031110.log
# Results = Test accuracy ~0.85 (~0.5x -> ran by Jack) if using chb16_17 -> chb16_17 (intNumFalsePositives_NoAug = 466, intNumFalseNegatives_NoAug = 5, intNumTruePositives_NoAug = 26, intNumTrueNegatives_NoAug_NoAug = 3093)
#           Very late convergence (end of epoch 8) -> train longer?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb16.csv' -tcsv './DataCSVs/CHB-MIT/chb16_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb17 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-040928.log
# Results = Test accuracy ~0. (~0.5x -> ran by Jack) if using chb17b_63 -> chb17c_08 (
#           Training curves flat. Test script not run -> try smod = 1 with tupScalingInfo next
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb17.csv' -tcsv './DataCSVs/CHB-MIT/chb17_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb17 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-050004.log
# Results = Test accuracy ~0.85 (~0.5x -> ran by Jack) if using chb17b_63 -> chb17c_08 (intNumFalsePositives_NoAug = 5815, intNumFalseNegatives_NoAug = 17, intNumTruePositives_NoAug = 71, intNumTrueNegatives_NoAug_NoAug = 33721)
#           Late convergenve of training curves, maybe train londer
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb17.csv' -tcsv './DataCSVs/CHB-MIT/chb17_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb18 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-052655.log
# Results = Incorrectly specified test CSV file as training file -> retrain
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb18.csv' -csv './DataCSVs/CHB-MIT/chb18_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb19 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200610-054004.log
# Results = Test accuracy ~0.91 (~0.7x -> ran by Jack) if using chb19_30 -> chb19_30 (intNumFalsePositives_NoAug = 47, intNumFalseNegatives_NoAug = 12, intNumTruePositives_NoAug = 69, intNumTrueNegatives_NoAug_NoAug = 3195)
#           Training curves appear kind of flat (not completely
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb19.csv' -tcsv './DataCSVs/CHB-MIT/chb19_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (tupScalingInfo)
# Log file = runTrainLSTM_20200610-125627.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (
#          Completely flat training curves. Retrain
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -tcsv './DataCSVs/CHB-MIT/chb20_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb04 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 64}
# Log file =  runTrainLSTM_20200610-183255.log
# Results = Test accuracy ~0. (~0.5x -> ran by Jack) if using chb04_22 -> chb04_43 (
#           Completely flat training curves, retrain
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb04.csv' -tcsv './DataCSVs/CHB-MIT/chb04_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 64'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 128}
# Log file = runTrainLSTM_20200610-230411.log
# Result = Test accuracy ~0.86 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 21, intNumFalseNegatives_NoAug = 81, intNumTruePositives_NoAug = 204, intNumTrueNegatives_NoAug_NoAug = 40705)
#          Big bump in epoch 7 after early convergence -> early stop?
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 128'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 32}
#          Early stop at 3 epochs
# Log file = runTrainLSTM_20200611-031856.log
# Result = Test accuracy ~0.88 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 52, intNumFalseNegatives_NoAug = 261, intNumTruePositives_NoAug = 879, intNumTrueNegatives_NoAug_NoAug = 40680)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 32'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 8}
#          Early stop at 3 epochs
# Log file = runTrainLSTM_20200611-150842.log
# Result = Test accuracy ~0.86 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 24, intNumFalseNegatives_NoAug = 1195, intNumTruePositives_NoAug = 3365, intNumTrueNegatives_NoAug_NoAug = 40714)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 8'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 3 epochs
# Log file = runTrainLSTM_20200611-160418.log
# Result = Test accuracy ~0. (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 1}
#          Early stop at 5 epochs
# Log file = runTrainLSTM_20200611-171903.log
# Result = Test accuracy ~0.84 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 34, intNumFalseNegatives_NoAug = 8954, intNumTruePositives_NoAug = 27526, intNumTrueNegatives_NoAug_NoAug = 40816)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 1'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (dropout = 0.1)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 5 epochs
# Log file = runTrainLSTM_20200612-043741.log
# Result = Test accuracy ~0.84 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 22, intNumFalseNegatives_NoAug = 5028, intNumTruePositives_NoAug = 13212, intNumTrueNegatives_NoAug_NoAug = 40760)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.1 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (dropout = 0.8)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 5 epochs
# Log file = runTrainLSTM_20200612-061526.log
# Result = Test accuracy ~0.87 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 48, intNumFalseNegatives_NoAug = 4063, intNumTruePositives_NoAug = 14177, intNumTrueNegatives_NoAug_NoAug = 40733)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.8 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (hidden dim = 128)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 5 epochs
# Log file = runTrainLSTM_20200612-075332.log
# Result = Test accuracy ~0.88 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 486, intNumFalseNegatives_NoAug = 3541, intNumTruePositives_NoAug = 14699, intNumTrueNegatives_NoAug_NoAug = 40293)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 128 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (hidden dim = 64)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 5 epochs
# Log file = runTrainLSTM_20200612-091934.log
# Result = Test accuracy ~0.86 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 16, intNumFalseNegatives_NoAug = 4612, intNumTruePositives_NoAug = 13628, intNumTrueNegatives_NoAug_NoAug = 40763)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 64 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (hidden dim = 32)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 5 epochs
# Log file = (Not run)
# Result = Test accuracy ~0. (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 32 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 3 epochs
# Log file = runTrainLSTM_20200613-042559.log
# Result = Test accuracy ~0.87 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 89, intNumFalseNegatives_NoAug = 3964, intNumTruePositives_NoAug = 14276, intNumTrueNegatives_NoAug_NoAug = 40689)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb07 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
#          Early stop at 3 epochs
# Log file = runTrainLSTM_20200613-071817.log
# Result = Test accuracy ~0.88 (~0.5x -> ran by Jack, ~0.80 -> non-sliding window) if using chb07_17 -> chb07_19 (intNumFalsePositives_NoAug = 1276, intNumFalseNegatives_NoAug = 3554, intNumTruePositives_NoAug = 14686, intNumTrueNegatives_NoAug_NoAug = 39506)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb07.csv' -tcsv './DataCSVs/CHB-MIT/chb07_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -cw 1 1 5 -opt 1 -lr 0.001 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb18 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (epoch = 5)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200614-044527.log
# Result = Test accuracy ~0.92 (~0.8x -> ran by Jack) if using chb18_33 -> chb18_36 (intNumFalsePositives_NoAug = 2138, intNumFalseNegatives_NoAug = 3, intNumTruePositives_NoAug = 111, intNumTrueNegatives_NoAug_NoAug = 12143)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb18.csv' -tcsv './DataCSVs/CHB-MIT/chb18_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb18 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (epoch = 5)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 2}
# Log file = runTrainLSTM_20200614-053457.log
# Results = Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb18.csv' -tcsv './DataCSVs/CHB-MIT/chb18_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 2'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 5 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb04 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used sliding window and AdamW as optimizer (lr = 0.005)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo) {'ictal': 32}
# Log file = runTrainLSTM_20200615-031949.log
# Results = Test accuracy ~0. (~0.5x -> ran by Jack) if using chb04_22 -> chb04_43 (
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb04.csv' -tcsv './DataCSVs/CHB-MIT/chb04_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -sss {'"ictal": 32'} -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.005 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (lr = 0.005)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-062102.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb08_13 -> chb08_29 (
#          Starting to converge at epoch 5, but returned to non-convergence level at epoch 7. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -tcsv './DataCSVs/CHB-MIT/chb08_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.005 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (lr - 0.005)
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-063902.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.005 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (lr = 0.005)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-073748.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -tcsv './DataCSVs/CHB-MIT/chb20_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.005 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-163458.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb08_13 -> chb08_29 (
#          Started to converge in epoch 4 but returned to non-convergence level at epoch 7. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -tcsv './DataCSVs/CHB-MIT/chb08_Test.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-165249.log
# Result = Test accuracy ~0.91 (~0.7x -> ran by Jack) if using chb08_13 -> chb08_29 (fltTruePositiveRate_NoAug = 0.8019, fltTrueNegativeRate_NoAug = 0.9149)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -tcsv './DataCSVs/CHB-MIT/chb08_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb08 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (lr = 0.002, epoch = 15)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200615-171044.log
# Result = Test accuracy ~0.92 (~0.7x -> ran by Jack) if using chb08_13 -> chb08_29 (fltTruePositiveRate_NoAug = 0.8090, fltTrueNegativeRate_NoAug = 0.9180)
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb08.csv' -tcsv './DataCSVs/CHB-MIT/chb08_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.002 -ep 15 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-025651.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (lr = 0.002, epoch = 15)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-035613.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.002 -ep 15 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-054536.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 2 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-172131.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (
#          Strange looking training curves (like diagonal lines). Poor ictal prediction, decent interictal prediction
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -tcsv './DataCSVs/CHB-MIT/chb20_Test.csv' -rf -1 -du 1 -bs 32 -smod 2 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 3 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-175455.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -tcsv './DataCSVs/CHB-MIT/chb20_Test.csv' -rf -1 -du 1 -bs 32 -smod 3 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 8 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb20 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200616-182759.log
# Result = Test accuracy ~0. (~0.7x -> ran by Jack) if using chb20_16 -> chb20_68 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb20.csv' -tcsv './DataCSVs/CHB-MIT/chb20_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.002 -ep 15 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (dropout = 0.75, epoch = 1)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200617-042914.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#          Flat training curves. Test script not run
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.75 -opt 1 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used AdamW as optimizer (epoch = 1)
#          Scaled data to (-1, 1) using argScalingMode = 0
# Log file = runTrainLSTM_20200617-044737.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
#python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -rf -1 -du 1 -bs 32 -smod 0 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 1 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used Adam as optimizer (epoch = 1)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = runTrainLSTM_20200617-045909.log
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 0 -lr 0.001 -ep 1 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net

# CHB-MIT: Train using interictal + ictal (chb15 subset) with ictal augmentation (use the earlier
#          seizures for training, reserved the later ones for testing)
#          Used SGD as optimizer (lr = 0.002, epoch = 3)
#          Scaled data to (-1, 1) using argScalingMode = 1 (with tupScalingInfo)
# Log file = 
# Result = Test accuracy ~0. (~0.8x -> ran by Jack) if using chb15_31 -> chb15_63 (
python scrTrainLSTM.py -csv './DataCSVs/CHB-MIT/chb15.csv' -tcsv './DataCSVs/CHB-MIT/chb15_Test.csv' -rf -1 -du 1 -bs 32 -smod 1 -smin -1 -smax 1 -vf 0.2 -tf 0.1 -gpu 0 -hd 256 -nl 2 -os 3 -dr 0.5 -opt 2 -lr 0.002 -ep 3 -lg RodentSys -pw M0useSys -fr RodentSys@gmail.com -to 4089300606@txt.att.net
