#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import relevant libraries and functions for this script
import sys, os
import numpy as np
import torch
import libDataIO as dio
import libModelLSTM as LSTM
import libUtils as utils

from operator import itemgetter


# In[ ]:


# Jupyter magic commands that should only be run when the code is running
# in Jupyter. Set to blnBatchMode to True when running in batch mode 
blnBatchMode = utils.fnIsBatchMode()

if (blnBatchMode):
    print('Running in BATCH mode...')
    
else:
    print('Running in INTERACTIVE mode...')
    # Automatically reload modules before code execution
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

    # Set plotting style
    get_ipython().run_line_magic('matplotlib', 'inline')
    get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[ ]:


# Set up script and training parameters from command line arguments (batch mode)
# or hard-coded values in the script
if (blnBatchMode):
    
    # BATCH MODE ONLY: This cell will execute in batch mode and parse the relevant
    #                  command line arguments
    
    import argparse

    # Construct argument parser
    objArgParse = argparse.ArgumentParser()

    # Add arguments to the parser
    objArgParse.add_argument('-md',   '--modeldir',          required = True,                   help = '')
    objArgParse.add_argument('-mn',   '--modelname',         required = True,                   help = '')
    objArgParse.add_argument('-tcsv', '--csvpath',           required = True,                   help = '')
    
    objArgParse.add_argument('-rf',   '--resamplingfreq',    required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-du',   '--subseqduration',    required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-ss',   '--stepsizetimepts',   required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-sss',  '--stepsizestates', type = json.loads, required = False, default = '{}', help = '')  # Example use: -sss '{"ictal": 128}'
    objArgParse.add_argument('-sw',   '--subwindowfraction', required = False, default = -1,    help = '')  # Will be over-written from model
    
    objArgParse.add_argument('-smod', '--scalingmode',       required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-smin', '--scaledmin',         required = False, default = -1,    help = '')  # Will be over-written from model
    objArgParse.add_argument('-smax', '--scaledmax',         required = False, default = 1,     help = '')  # Will be over-written from model
    
    objArgParse.add_argument('-sd',   '--shuffledata',       required = False, default = False, help = '')  # No real need to shuffle for testing
    
    objArgParse.add_argument('-gpu',  '--gpudevice',         required = False, default = -1,    help = '')
    objArgParse.add_argument('-sa',   '--saveanno',          required = False, default = False, help = '')

    dctArgs = vars(objArgParse.parse_args())

    # Convert parameters extract from arguments to their appropriate date types
    argModelDir          = dctArgs['modeldir']
    argModelName         = dctArgs['modelname']
    argCSVPath           = dctArgs['csvpath']
    
    argResamplingFreq    = int(dctArgs['resamplingfreq'])
    argSubSeqDuration    = int(dctArgs['subseqduration'])
    argStepSizeTimePts   = int(dctArgs['stepsizetimepts'])
    argStepSizeStates    = dctArgs['stepsizestates']
    argStepSizeStates    = {}     # Step size of sliding window for specific segment states (default = {}, use -ssv value)
    argSubWindowFraction = float(dctArgs['subwindowfraction'])
    
    argScalingMode       = int(dctArgs['scalingmode'])
    argScaledMin         = int(dctArgs['scaledmin'])
    argScaledMax         = int(dctArgs['scaledmax'])
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    argShuffleData       = dctArgs['shuffledata']
    argGPUDevice         = int(dctArgs['gpudevice'])
    argSaveAnno          = dctArgs['saveanno']
    
else:
    # Set all configurable parameters of the script as arguments. After
    # these parameters are set, the entire script can be run in one go

    # Specify the path where the trained models are saved
    argModelDir = './SavedModels/'

    # Specify the model name to use for testing
    argModelName = 'EEGLSTM_CHB-MIT_chb15_Epoch-1_TLoss-0.5327_VLoss-0.6197_20200617-044737.net'  # chb15 (AdamW, smod = 0), epoch = 1 -> hard to tell if there is convergence -> test accuracy from training = 0.82 although test loss = 0.5464. Almost zero ictal prediction, good interictal prediction
    
    # Specify the CSV file that lists the EEG segments to use for testing
    argCSVPath        = './DataCSVs/CHB-MIT/chb15_Test.csv'

    # Specify the resampling frequency and subsequence durations for the
    # EEG segments of the test set (most of the following arguments will
    # be over-written by those from the loaded model if the model contains
    # dctModelProperties{})
    argResamplingFreq    = -1
    argSubSeqDuration    = 1
    argStepSizeTimePts   = -1
    argStepSizeStates    = {}     # Step size of sliding window for specific segment states (default = {}, use -ssv value)
    argSubWindowFraction = 0.3
    
    argScalingMode       = 1
    argScaledMin         = -1
    argScaledMax         = 1
    
    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    # There is really no need to shuffle data when we're feeding in the test set
    argShuffleData       = True  # Shuffle data in DataLoader or not (should not affect test results)

    # Specify which GPU device to use
    argGPUDevice = 0
    
    # Specify whether to save test results to annotation files
    argSaveAnno = True


# In[ ]:


# Generate a timestamp that is unique to this run
strTimestamp = str(utils.fnGenTimestamp())
print('strTimestamp = {}'.format(strTimestamp))


# In[ ]:


# Create a log file only when in batch mode
if (blnBatchMode):
    # Log all output messages to a log file when it is in Batch mode
    strLogDir = './Logs/'  # TODO: Make this into an argument?

    # Create a new directory if it does not exist
    utils.fnOSMakeDir(strLogDir)

    # Saving the original stdout and stderr
    objStdout = sys.stdout
    objStderr = sys.stderr

    strLogFilename = 'runTestLSTM_' + strTimestamp + '_' + argModelName + '.log'
    print('strLogFilename = {}'.format(strLogFilename))

    # Open a new log file
    objLogFile = open(strLogDir + strLogFilename, 'w')

    # Replace stdout and stderr with log file so all print statements will
    # be redirected to the log file from this point on
    sys.stdout = objLogFile
    sys.stderr = objLogFile

    datScriptStart = utils.fnNow()
    print('Script started on {}'.format(utils.fnGetDatetime(datScriptStart)))
    print()


# In[ ]:


# Load a saved LSTM model from the file system

strModelDir = argModelDir
strModelName = argModelName

(intTrainNumChannels, intTrainSeqLen, intTrainNumSegments, objModelLSTM,
 intNumEpochs, intBatchSize, blnShuffleIndices, blnShuffleData, fltLearningRate, intPrintEvery, fltGradClip,
 lstTrainingStepLosses, lstValidationStepLosses, dctModelProperties) = LSTM.fnLoadLSTMModel(strModelDir, strModelName)

# Over-write arguments from the script with values saved in the model
if (dctModelProperties):
    lstTrainingChannels  = dctModelProperties['lstTrainingChannels']
    
    argResamplingFreq    = dctModelProperties['fltResamplingFreq']
    argSubSeqDuration    = dctModelProperties['fltSubSeqDuration']
    argStepSizeTimePts   = dctModelProperties['intStepSizeTimePts']
    argStepSizeStates    = utils.fnFindInDct(dctModelProperties, 'dctStepSizeStates', argStepSizeStates)  # May not exist in some models
    argSubWindowFraction = dctModelProperties['fltSubWindowFraction']

    argScalingMode       = dctModelProperties['intScalingMode']
    argScaledMin         = dctModelProperties['fltScaledMin']
    argScaledMax         = dctModelProperties['fltScaledMax']

    argScalingParams = () if (argScalingMode == -1) else (argScalingMode, (argScaledMin, argScaledMax))
    
    tupScalingInfo       = utils.fnFindInDct(dctModelProperties, 'tupScalingInfo', ())  # May not exist in some models
    
    intValPerEpoch       = dctModelProperties['intValPerEpoch']
    
    print('\n  dctModelProperties{} exists in saved model. Over-writting the following arguments with values saved in the model:')
    print()
    print('    argResamplingFreq = {}'.format(argResamplingFreq))
    print('    argSubSeqDuration = {}'.format(argSubSeqDuration))
    print('    argStepSizeTimePts = {}'.format(argStepSizeTimePts))
    print('    argStepSizeStates = {}'.format(argStepSizeStates))
    print('    argSubWindowFraction = {}'.format(argSubWindowFraction))
    print('    argScalingParams = {}'.format(argScalingParams))
    print()
    
    if ('strLogFilename' in dctModelProperties.keys()):
        print('    strLogFilename = {}'.format(dctModelProperties['strLogFilename']))
        print()
    
else:
    lstTrainingChaneels = []
    tupScalingInfo      = ()
    intValPerEpoch      = -1
    
    print('\n  WARNING: dctModelProperties{} not found in saved model. Using arguments specified in this training script')
    print()

print('lstTrainingChannels = {}'.format(lstTrainingChannels))
print('len(tupScalingInfo) = {}'.format(len(tupScalingInfo)))
print('intValPerEpoch = {}'.format(intValPerEpoch))
print()

utils.fnShowMemUsage()
print()


# In[ ]:


# Print out all specified arguments
print('argModelDir = {}'.format(argModelDir))
print('argModelName = {}'.format(argModelName))
print('argCSVPath = {}'.format(argCSVPath))

print('argResamplingFreq = {}'.format(argResamplingFreq))
print('argSubSeqDuration = {}'.format(argSubSeqDuration))
print('argStepSizeTimePts = {}'.format(argStepSizeTimePts))
print('argStepSizeStates = {}'.format(argStepSizeStates))
print('argSubWindowFraction = {}'.format(argSubWindowFraction))

print('argScalingParams = {}'.format(argScalingParams))

if (tupScalingInfo):
    print('  -> Scaling across training and test files')

print('argShuffleData = {}'.format(argShuffleData))

print('argGPUDevice = {}'.format(argGPUDevice))

print('argSaveAnno = {}'.format(argSaveAnno))

print()

utils.fnShowMemUsage()
print()


# In[ ]:


# Plot training loss and validation loss for the entire training if in interactive mode
if (not blnBatchMode):
    utils.fnPlotTrainValLosses(lstTrainingStepLosses, lstValidationStepLosses, intValPerEpoch, argXLim = (), argYLim = ())


# In[ ]:


# Read test data from files

strCSVPath = argCSVPath

fltResamplingFreq    = argResamplingFreq
fltSubSeqDuration    = argSubSeqDuration
intStepSizeTimePts   = argStepSizeTimePts
dctStepSizeStates    = argStepSizeStates
fltSubWindowFraction = argSubWindowFraction

tupScalingParams     = argScalingParams

blnShuffleData       = argShuffleData  # TODO: Should this follow the model or the argument?

# Read CHB-MIT training data from files using sliding window
lstLabeledTestFilenames, lstLabeledTestSegLabels, lstLabeledTestSegTypes, arrLabeledTestDataRaw, lstLabeledTestSegDurations, lstLabeledTestSamplingFreqs, lstLabeledTestChannels, lstLabeledTestSequences, lstLabeledTestSubSequences, lstLabeledTestSeizureDurations, arrLabeledTestStartEndTimesSec, _ = dio.fnReadCHBMITEDFFiles_SlidingWindow(
    argCSVPath = strCSVPath, argResamplingFreq = fltResamplingFreq, argSubSeqDuration = fltSubSeqDuration, argScalingParams = tupScalingParams, argScalingInfo = tupScalingInfo, argStepSizeTimePts = intStepSizeTimePts, argStepSizeStates = dctStepSizeStates, argSubWindowFraction = fltSubWindowFraction, argAnnoSuffix = 'annotation.txt', argDebug = False, argTestMode = False)

print()

utils.fnShowMemUsage()
print()


# In[ ]:


# Check whether the channels used for training are the same as the one used for testing
if (lstTrainingChannels):
    if (lstTrainingChannels != lstLabeledTestChannels[0]):
        print('Training channels:\n  {}'.format(lstTrainingChannels))
        print()
        print('Test channels:\n  {}'.format(lstLabeledTestChannels[0]))
        
        raise Exception('Training channels do not match test channels!')
        
# If tupScalingInfo exists (scaling across training and test sets, check whether the test
# files used in scaling are the same as the ones specified in this test script
if (tupScalingInfo):
    lstScalingTestFilenames = sorted(tupScalingInfo[0])
    lstScriptTestFilenames = sorted(set(lstLabeledTestFilenames))
    
    if (lstScalingTestFilenames != lstScriptTestFilenames):
        print('Test files used for scaling:\n {}'.format())
        print()
        print('Test files specified in test script:\n {}'.format())
        raise Exception('Test files used for scaling do not match test files specified in test script!')


# In[ ]:


# TEST: Read test data from files using the non-sliding window version of fnReadCHBMITEDFFiles()
lstLabeledTestFilenames_Old, lstLabeledTestSegLabels_Old, lstLabeledTestSegTypes_Old, arrLabeledTestDataRaw_Old, lstLabeledTestSegDurations_Old, lstLabeledTestSamplingFreqs_Old, lstLabeledTestChannels_Old, lstLabeledTestSequences_Old, lstLabeledTestSubSequences_Old, lstLabeledTestSeizureDurations_Old, arrLabeledTestStartEndTimesSec_Old = dio.fnReadCHBMITEDFFiles(
    argCSVPath = strCSVPath, argResamplingFreq = fltResamplingFreq, argSubSeqDuration = fltSubSeqDuration, argScalingParams = tupScalingParams, argAnnoSuffix = 'annotation.txt', argDebug = False)


# In[ ]:


# TEST: Compare results between fnReadCHBMITEDFFiles_SlidingWindow() and fnReadCHBMITEDFFiles().

print('Same') if (lstLabeledTestFilenames        == lstLabeledTestFilenames_Old)                      else print('Different')
print('Same') if (lstLabeledTestSegLabels        == lstLabeledTestSegLabels_Old)                      else print('Different')
print('Same') if (lstLabeledTestSegTypes         == lstLabeledTestSegTypes_Old)                       else print('Different')

#print('Same') if (np.array_equal(arrLabeledTestDataRaw, arrLabeledTestDataRaw_Old))                   else print('Different')
if (np.array_equal(arrLabeledTestDataRaw, arrLabeledTestDataRaw_Old)):
    print('Same')
else:
    print('Different (Difference = {})'.format(np.sum(arrLabeledTestDataRaw - arrLabeledTestDataRaw_Old)))

print('Same') if (lstLabeledTestSegDurations     == lstLabeledTestSegDurations_Old)                   else print('Different')
print('Same') if (lstLabeledTestSamplingFreqs    == lstLabeledTestSamplingFreqs_Old)                  else print('Different')
print('Same') if (lstLabeledTestChannels         == lstLabeledTestChannels_Old)                       else print('Different')
print('Same') if (lstLabeledTestSequences        == lstLabeledTestSequences_Old)                      else print('Different')
print('Same') if (lstLabeledTestSubSequences     == lstLabeledTestSubSequences_Old)                   else print('Different')
print('Same') if (lstLabeledTestSeizureDurations == lstLabeledTestSeizureDurations_Old)               else print('Different')
print('Same') if (np.array_equal(arrLabeledTestStartEndTimesSec, arrLabeledTestStartEndTimesSec_Old)) else print('Different')


# In[ ]:


# TEST: Take a quick snapshot of the data set
intStartIdx = 5000
intEndIdx = 6010
#intEndIdx = len(lstTrainingSegLabels)

print('Displaying a snapshot of the data set (from subsequence {} to {}):\n'.format(intStartIdx, intEndIdx))

for tupZip in zip(list(range(len(lstLabeledTestSegLabels[intStartIdx:intEndIdx]))),
                  lstLabeledTestFilenames[intStartIdx:intEndIdx],
                  lstLabeledTestSegLabels[intStartIdx:intEndIdx],
                  lstLabeledTestSegTypes[intStartIdx:intEndIdx],
                  lstLabeledTestSegDurations[intStartIdx:intEndIdx],
                  lstLabeledTestSamplingFreqs[intStartIdx:intEndIdx],
                  lstLabeledTestSequences[intStartIdx:intEndIdx],
                  lstLabeledTestSubSequences[intStartIdx:intEndIdx], 
                  arrLabeledTestStartEndTimesSec[intStartIdx:intEndIdx, :]
                 ):
    print(*tupZip, sep = '\t')
    
print()


# #### Relationships between UID, UUID, and EEG segments
# 
# Unique IDs are assigned to each subsequence in the test set in order to trace prediction results back to the original EEG subsequences. UIDs are not unique among subsequences that are replicated/augmented (e.g. preictal subsequences), while UUIDs are unique for each subsequence in the test set.
# 
# ```
# Type Seq SubSeq SegIdx UID UUID
# ---- --- ------ ------ --- ----
# Int   1   0      0      0   0
# Int   1   1      0      1   1
# Int   1   2      0      2   2
# 
# Ict0  1   0      1      3   3
# Ict0  1   1      1      4   4
# Ict0  1   2      1      5   5
# 
# Ict0  1   0      1      3   6
# Ict0  1   1      1      4   7
# Ict0  1   2      1      5   8
# 
# Int   1   0      2      6   9
# Int   1   1      2      7   10
# Int   1   2      2      8   11
# 
# Ict1  1   0      3      9   12
# Ict1  1   1      3      10  13
# Ict1  1   2      3      11  14
# 
# Ict1  1   0      3      9   15
# Ict1  1   1      3      10  16
# Ict1  1   2      3      11  17
# ```

# In[ ]:


# Assign sequential unique ID (UID) to pre-augmented data
print('Number of pre-augmentation subsequences = {}'.format(len(lstLabeledTestSegTypes)))
print('arrLabeledTestDataRaw.shape = {}'.format(arrLabeledTestDataRaw.shape))

lstLabeledTestUIDs = list(range(len(lstLabeledTestSegTypes)))
print('lstLabeledTestUIDs = {} ... {}'.format(np.min(lstLabeledTestUIDs), np.max(lstLabeledTestUIDs)))


# In[ ]:


# Perform non-random oversampling of the ictal data due to imbalanced
# classification between the amount of interictal data versus ictal data

blnDebug = False

# Get the subsequences that are labeled as ictal state
lstSeizureSeqIdx = [intSeqIdx for intSeqIdx, intSegType in enumerate(lstLabeledTestSegTypes) if intSegType == 2]

# Collect all related data for these ictal subsequences
lstSeizureFilenames     = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestFilenames))
lstSeizureSeqLabels     = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSegLabels))
lstSeizureSegTypes      = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSegTypes))
arrSeizureDataRaw       = arrLabeledTestDataRaw[:, :, lstSeizureSeqIdx]
lstSeizureSegDurations  = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSegDurations))
lstSeizureSamplingFreqs = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSamplingFreqs))
lstSeizureChannels      = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestChannels))
lstSeizureSequences     = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSequences))
lstSeizureSubSequences  = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestSubSequences))

arrSeizureStartEndTimesSec = arrLabeledTestStartEndTimesSec[lstSeizureSeqIdx, :]  # Start/end times (in seconds) for each subsequence

lstSeizureUIDs          = list(itemgetter(*lstSeizureSeqIdx)(lstLabeledTestUIDs))  # Get UIDs of ictal subsequences

print('arrSeizureDataRaw.shape = {}'.format(arrSeizureDataRaw.shape))
print()

if (blnDebug):
    # Print info on the collected ictal subsequences
    for tupSubSeq in zip(lstSeizureFilenames, lstSeizureSeqLabels, lstSeizureSegTypes, lstSeizureSegDurations, lstSeizureSamplingFreqs, lstSeizureSequences, lstSeizureSubSequences, lstSeizureUIDs):
        print('{}, {}, {}, {}, {}, {}, {}, {}'.format(*tupSubSeq))

    print()

# Calculate the ratio between the number of ictal and non-ictal subsequences,
# and multiply the number of ictal subsequences with this imbalance factor to
# increase the presence of ictal subsequences in the data set
intTotalSubSeqs = arrLabeledTestDataRaw.shape[2]
intNumSeizureSubSeqs = arrSeizureDataRaw.shape[2]

intImbalFactor = int(round((intTotalSubSeqs - intNumSeizureSubSeqs) / intNumSeizureSubSeqs))
print('intTotalSubSeqs = {}, intNumSeizureSubSeqs = {}: intImbalFactor = {}'.format(intTotalSubSeqs, intNumSeizureSubSeqs, intImbalFactor))

# Perform oversampling of ictal data only if intImbalFactor > 0
if (intImbalFactor > 0):
    print('Oversampling of ictal data performed using intImbalFactor = {}'.format(intImbalFactor))
    print()
    
    # Multiply the number of ictal subsequences by intImbalFactor
    lstSeizureFilenamesRep     = lstSeizureFilenames * intImbalFactor
    lstSeizureSeqLabelsRep     = lstSeizureSeqLabels * intImbalFactor
    lstSeizureSegTypesRep      = lstSeizureSegTypes * intImbalFactor
    arrSeizureDataRawRep       = np.tile(arrSeizureDataRaw, (1, 1, intImbalFactor))
    lstSeizureSegDurationsRep  = lstSeizureSegDurations * intImbalFactor
    lstSeizureSamplingFreqsRep = lstSeizureSamplingFreqs * intImbalFactor
    lstSeizureChannelsRep      = lstSeizureChannels * intImbalFactor
    lstSeizureSequencesRep     = lstSeizureSequences * intImbalFactor
    lstSeizureSubSequencesRep  = lstSeizureSubSequences * intImbalFactor

    arrSeizureStartEndTimesSecRep = np.tile(arrSeizureStartEndTimesSec, (intImbalFactor, 1))  # Start/end times (in seconds) for each subsequence

    lstSeizureUIDsRep  = lstSeizureUIDs * intImbalFactor

    print('arrSeizureDataRawRep.shape = {}, arrSeizureStartEndTimesSec = {}'.format(arrSeizureDataRawRep.shape, arrSeizureStartEndTimesSec.shape))
    print()

    #for tupSubSeqRep in zip(lstSeizureFilenamesRep, lstSeizureSeqLabelsRep, lstSeizureSegTypesRep, lstSeizureSegDurationsRep, lstSeizureSamplingFreqsRep, lstSeizureSequencesRep, lstSeizureSubSequencesRep):
    #    print('{}, {}, {}, {}, {}, {}, {}'.format(*tupSubSeqRep))

    #print()

    # Append the replicated ictal data to the end of the data set
    lstLabeledTestFilenamesAug     = lstLabeledTestFilenames + lstSeizureFilenamesRep
    lstLabeledTestSegLabelsAug     = lstLabeledTestSegLabels + lstSeizureSeqLabelsRep
    lstLabeledTestSegTypesAug      = lstLabeledTestSegTypes + lstSeizureSegTypesRep
    arrLabeledTestDataRawAug       = np.concatenate((arrLabeledTestDataRaw, arrSeizureDataRawRep), axis = 2)
    lstLabeledTestSegDurationsAug  = lstLabeledTestSegDurations + lstSeizureSegDurationsRep
    lstLabeledTestSamplingFreqsAug = lstLabeledTestSamplingFreqs + lstSeizureSamplingFreqsRep
    lstLabeledTestChannelsAug      = lstLabeledTestChannels + lstSeizureChannelsRep
    lstLabeledTestSequencesAug     = lstLabeledTestSequences + lstSeizureSequencesRep
    lstLabeledTestSubSequencesAug  = lstLabeledTestSubSequences + lstSeizureSubSequencesRep

    arrLabeledTestStartEndTimesSecAug  = np.concatenate((arrLabeledTestStartEndTimesSec, arrSeizureStartEndTimesSecRep), axis = 0)  # # Start/end times (in seconds) for each subsequence

    lstLabeledTestUIDsAug          = lstLabeledTestUIDs + lstSeizureUIDsRep

    print('arrLabeledTestDataRawAug.shape = {}'.format(arrLabeledTestDataRawAug.shape))

    # Append the replicated ictal data to the end of the data set
    lstLabeledTestFilenames     = lstLabeledTestFilenamesAug
    lstLabeledTestSegLabels     = lstLabeledTestSegLabelsAug
    lstLabeledTestSegTypes      = lstLabeledTestSegTypesAug
    arrLabeledTestDataRaw       = arrLabeledTestDataRawAug
    lstLabeledTestSegDurations  = lstLabeledTestSegDurationsAug
    lstLabeledTestSamplingFreqs = lstLabeledTestSamplingFreqsAug
    lstLabeledTestChannels      = lstLabeledTestChannelsAug
    lstLabeledTestSequences     = lstLabeledTestSequencesAug
    lstLabeledTestSubSequences  = lstLabeledTestSubSequencesAug

    arrLabeledTestStartEndTimesSec = arrLabeledTestStartEndTimesSecAug  # Start/end times (in seconds) for each subsequence

    lstLabeledTestUIDs          = lstLabeledTestUIDsAug

    print()

    print('Size of arrSeizureDataRawRep = {:.2f}Gb'.format(utils.fnByte2GB(arrSeizureDataRawRep.nbytes)))
    print('Size of arrTrainingDataRaw   = {:.2f}Gb'.format(utils.fnByte2GB(arrLabeledTestDataRaw.nbytes)))
    print()
    
else:
    print('Oversampling of ictal data not performed since intImbalFactor = {}'.format(intImbalFactor))
    print()
    
    print('Size of arrSeizureDataRaw = {:.2f}Gb'.format(utils.fnByte2GB(arrSeizureDataRaw.nbytes)))
    print('Size of arrTrainingDataRaw   = {:.2f}Gb'.format(utils.fnByte2GB(arrTrainingDataRaw.nbytes)))
    print()
    
utils.fnShowMemUsage()
print()


# In[ ]:


# Assign sequential unique unique ID (UUID) to post-augmented data
lstLabeledTestUUIDs = list(range(len(lstLabeledTestUIDs)))
print('lstLabeledTestUUIDs = {} ... {}'.format(np.min(lstLabeledTestUUIDs), np.max(lstLabeledTestUUIDs)))


# In[ ]:


# Input width = 15 (number of channels/features)
# Sequence length = 600 * 5000 = 3000000 (number of time points)
# Batch size = 1 (for each segment)

# Reshape arrAllData[] from [feature/channel size x segment length x batch/segment size]
# to batch_first [batch/segment size x segment length x feature/channel size]
intLabeledTestNumChannels, intLabeledTestSeqLen, intLabeledTestNumSegments = arrLabeledTestDataRaw.shape
print('intLabeledTestNumChannels, intLabeledTestSeqLen, intLabeledTestNumSegments = ({}, {}, {})'.format(intLabeledTestNumChannels, intLabeledTestSeqLen, intLabeledTestNumSegments))
arrLabeledTestDataBatchFirst = arrLabeledTestDataRaw.T.reshape(intLabeledTestNumSegments, intLabeledTestSeqLen, intLabeledTestNumChannels)
print('arrTestDataBatchFirst.shape = {}'.format(arrLabeledTestDataBatchFirst.shape))

# Convert the segment types into an np.array
arrLabeledTestSegTypes = np.array(lstLabeledTestSegTypes, dtype = int)
print('arrLabeledTestSegTypes = {}'.format(arrLabeledTestSegTypes))


# In[ ]:


arrLabeledTestData = arrLabeledTestDataBatchFirst
arrLabeledTestLabels = arrLabeledTestSegTypes
print('arrLabeledTestData.shape = {}, arrLabeledTestLabels.shape = {}'.format(arrLabeledTestData.shape, arrLabeledTestLabels.shape))


# In[ ]:


# Concatenate arrLabeledTestUUIDs[] to arrLabeledTestLabels[] so their entries are
# fed to the neural network in tandem during testing
arrLabeledTestUUIDs = np.array(lstLabeledTestUUIDs, dtype = int)  # Convert from list to np.array
print('arrLabeledTestLabels.shape = {}, arrLabeledTestUUIDs.shape = {}'.format(arrLabeledTestLabels.shape, arrLabeledTestUUIDs.shape))

# Concatenate the two arrays vertically
arrLabeledTestLabelsWithUUIDs = np.concatenate((arrLabeledTestLabels[:, np.newaxis], arrLabeledTestUUIDs[:, np.newaxis]), axis = 1)
print('arrLabeledTestLabelsWithUUIDs.shape = {}'.format(arrLabeledTestLabelsWithUUIDs.shape))


# In[ ]:


# Convert test data/labels into DataLoader objects so we can easily iterate through the
# data sets during testing

from torch.utils.data import TensorDataset, DataLoader

# Convert test data and labels from np.arrays into data set wrapping tensors for DataLoader
#objLabeledTestDataset = TensorDataset(torch.from_numpy(arrLabeledTestData), torch.from_numpy(arrLabeledTestLabels))
objLabeledTestDataset = TensorDataset(torch.from_numpy(arrLabeledTestData), torch.from_numpy(arrLabeledTestLabelsWithUUIDs))  # Use labels with UUIDs attached
objLabeledTestLoader  = DataLoader(objLabeledTestDataset, shuffle = blnShuffleData, batch_size = intBatchSize)


# In[ ]:


# Check if a GPU is available and if so, set a device to use

intGPUDevice = argGPUDevice

blnTrainOnGPU = torch.cuda.is_available()

if (blnTrainOnGPU):
    intNumGPUs = torch.cuda.device_count()
    print('Training on GPU ({} available):'.format(intNumGPUs))
    for intGPU in range(intNumGPUs):
        print('  Device {}: {}'.format(intGPU, torch.cuda.get_device_name(intGPU)))
    torch.cuda.set_device(intGPUDevice)
    print('Using GPU #{}'.format(intGPUDevice))
else:
    print('No GPU available, training on CPU')


# In[ ]:


# Define loss criterion

import torch.nn as nn

objCriterion = nn.CrossEntropyLoss()


# In[ ]:


# Evaluate model with test data set and record test losses & prediction accuracy

blnDebug = False
lstTestLosses = []  # Record test losses per batch/step
intNumCorrect = 0   # Number of correctly predicted sequences in a batch size of (intBatchSize)

# Initialize arrTestResults[] to collect UUIDs, labels, and prediction results
intBatchIdx = 0
intNumBatches = arrLabeledTestLabelsWithUUIDs.shape[0] // intBatchSize  # Exclude the last orphan batch
intTestSetSize = intNumBatches * intBatchSize
arrTestResults = np.zeros((intTestSetSize, 3), dtype = int)
if (blnDebug): print('arrTestResults.shape = {}'.format(arrTestResults.shape))

# Initialize hidden and cell states
arrHiddenState = objModelLSTM.initHidden(intBatchSize, blnTrainOnGPU, argDebug = False)  # Batch size defined above when creating DataLoader

# Move the model to the GPU if one is available
if (blnTrainOnGPU):
    objModelLSTM.cuda()
    
objModelLSTM.eval()

objTestLoader = objLabeledTestLoader

# Batch loop (each loop trains one batch of input data)
#for arrInputData, arrLabels in objTestLoader:
for arrInputData, arrLabelsWithUUIDs in objTestLoader:  # Batches of labels are attached with their corresponding UUIDs
    print('Feed forwarding new test batch (#{})...'.format(intBatchIdx + 1))
    
    # Separate labels from their UUIDs into different np.arrays
    arrLabels = arrLabelsWithUUIDs[:, 0]
    arrUUIDs   = arrLabelsWithUUIDs[:, 1]
    
    # If batch size allocated from DataLoader is smaller than intBatchSize
    # (which happens on the last batch when the data set is not divisible
    # by intBatchSize), break out of the loop
    
    # TODO: This is the strategy for now until we figure out what the best
    #       strategy is on how/whether to initialize the hidden state with
    #       a smaller batch size for the last orphan batch
    if (intBatchSize != arrInputData.shape[0]):
        arrUnusedUUIDs = utils.fnTensor2Array(arrUUIDs, blnTrainOnGPU)  # Convert tensor back to np.array
        print('Exiting validation loop (intBatchSize = {}, arrInputData.shape[0] = {})'.format(intBatchSize, arrInputData.shape[0]))
        break
    else:
        arrUnusedUUIDs = np.array([])
        
    if (blnTrainOnGPU):
        arrInputData, arrLabels = arrInputData.cuda(), arrLabels.cuda()
    
    # Extract new variables for the hidden and cell states to decouple states
    # from backprop history. Otherwise the gradient will be backpropagated
    # through the entire training history
    arrHiddenState = tuple([arrState.data for arrState in arrHiddenState])

    # Forward pass through the model and get the next hidden state and output
    # Output shape = (batch_size, 1), h shape = (n_layers, batch_size, hidden_dim)
    arrOutput, arrHiddenState = objModelLSTM.forward(arrInputData, arrHiddenState, argDebug = False)
    
    if (blnDebug):
        print('  arrLabels = {}'.format(arrLabels))
        print('  arrOutput = \n{}'.format(arrOutput))
    
    # Calculate test loss for this batch
    fltTestLoss = objCriterion(arrOutput, arrLabels)
    print('  fltTestLoss = {:.6f} ({})'.format(fltTestLoss, fltTestLoss.type()))
    
    # Record test loss for this batch
    lstTestLosses.append(fltTestLoss.item())
    
    # Convert output scores between classes to one-hot encoding
    # that indicate the predictions for each batch
    _, arrPredictions = torch.max(arrOutput, 1)
    
    # Compare the class predictions to the test set labels
    arrCorrect = arrPredictions.eq(arrLabels)
    arrCorrect = utils.fnTensor2Array(arrCorrect, blnTrainOnGPU)  # Convert tensor back to np.array
        
    intNumCorrect += np.sum(arrCorrect)  # Count the number of correctly predicted sequences
    
    # Convert tensors back to np.arrays for UUIDs, labels, and prediction results
    arrUUIDs = utils.fnTensor2Array(arrUUIDs, blnTrainOnGPU)
    arrLabels = utils.fnTensor2Array(arrLabels, blnTrainOnGPU)
    arrPredictions = utils.fnTensor2Array(arrPredictions, blnTrainOnGPU)
        
    arrBatchResults = np.concatenate((arrUUIDs[:, np.newaxis], arrLabels[:, np.newaxis], arrPredictions[:, np.newaxis]), axis = 1)
    if (blnDebug): print('  arrBatchResults = \n{}'.format(arrBatchResults))
    
    intStartIdx = intBatchIdx * intBatchSize
    intEndIdx   = intStartIdx + intBatchSize
    if (blnDebug): print('  intStartIdx = {}, intEndIdx = {}'.format(intStartIdx, intEndIdx))
    
    arrTestResults[intStartIdx:intEndIdx, :] = arrBatchResults  # Place batch results into arrTestResults[]
    
    intBatchIdx = intBatchIdx + 1
    
    # Clear the GPU cache regularly to avoid the following CUDA error:
    #
    #   RuntimeError: CUDA out of memory. Tried to allocate 6.61 GiB 
    #   (GPU 1; 10.76 GiB total capacity; 1.25 GiB already allocated; 
    #   2.39 GiB free; 6.38 GiB cached)
    torch.cuda.empty_cache()
    
# Print the mean test loss for the entire test set
print("Test Loss = {:.4f}".format(np.mean(lstTestLosses)))

# Print the test accuracy over all test data
fltTestAccuracy = intNumCorrect / intTestSetSize
print("Test Accuracy = {}/{} = {:.4f}".format(int(round(intNumCorrect)), intTestSetSize, fltTestAccuracy))


# In[ ]:


# Perform post-testing errror analyses (for augmented data set)

# Lists and arrays that may be helpful with the error analysis:
#
#  lstLabeledTestFilenames
#  lstLabeledTestSegLabels
#  lstLabeledTestSegTypes
#  lstLabeledTestSequences
#  lstLabeledTestSubSequences

#  lstLabeledTestUIDs
#  lstLabeledTestUUIDs

# Lists that are already converted to np.arrays:
#
#  arrLabeledTestSegTypes
#  arrLabeledTestUUIDs
#
#  arrLabeledTestStartEndTimesSec

intNumUnusedUUIDs = arrUnusedUUIDs.shape[0]

print('Number of unused UUIDs in orphan batch = {}'.format(intNumUnusedUUIDs))
if (blnDebug): print('Unused UUIDs in orphan batch (sorted):\n{}'.format(arrUnusedUUIDs[arrUnusedUUIDs.argsort()]))

# Convert lists into np.arrays
arrLabeledTestFilnames     = np.array(lstLabeledTestFilenames)
arrLabeledTestSegLabels    = np.array(lstLabeledTestSegLabels)
arrLabeledTestSequences    = np.array(lstLabeledTestSequences)
arrLabeledTestSubSequences = np.array(lstLabeledTestSubSequences)
arrLabeledTestUIDs         = np.array(lstLabeledTestUIDs)

if (intNumUnusedUUIDs):
    # Remove unused entries from orphan batch based on their UUIDs
    arrLabeledTestFilnames_Tested     = np.delete(arrLabeledTestFilnames, arrUnusedUUIDs)
    arrLabeledTestSegLabels_Tested    = np.delete(arrLabeledTestSegLabels, arrUnusedUUIDs)
    arrLabeledTestSegTypes_Tested     = np.delete(arrLabeledTestSegTypes, arrUnusedUUIDs)
    arrLabeledTestSequences_Tested    = np.delete(arrLabeledTestSequences, arrUnusedUUIDs)
    arrLabeledTestSubSequences_Tested = np.delete(arrLabeledTestSubSequences, arrUnusedUUIDs)
    arrLabeledTestUIDs_Tested         = np.delete(arrLabeledTestUIDs, arrUnusedUUIDs)
    arrLabeledTestUUIDs_Tested        = np.delete(arrLabeledTestUUIDs, arrUnusedUUIDs)

    arrLabeledTestStartEndTimesSec_Tested = np.delete(arrLabeledTestStartEndTimesSec, arrUnusedUUIDs, axis = 0)  # Start/end times (in seconds) for each subsequence

else:
    arrLabeledTestFilnames_Tested     = arrLabeledTestFilnames
    arrLabeledTestSegLabels_Tested    = arrLabeledTestSegLabels
    arrLabeledTestSegTypes_Tested     = arrLabeledTestSegTypes
    arrLabeledTestSequences_Tested    = arrLabeledTestSequences
    arrLabeledTestSubSequences_Tested = arrLabeledTestSubSequences
    arrLabeledTestUIDs_Tested         = arrLabeledTestUIDs
    arrLabeledTestUUIDs_Tested        = arrLabeledTestUUIDs

    arrLabeledTestStartEndTimesSec_Tested = arrLabeledTestStartEndTimesSec  # Start/end times (in seconds) for each subsequence
    
# Sort arrTestResults[] based on their UUIDs since the test entries are shuffled
# before being fed into the neural network
arrTestResults_Sorted = arrTestResults[arrTestResults[:, 0].argsort()]  # Columns = [UUID, label, predicition]
print('arrTestResults_Sorted.shape = {}'.format(arrTestResults_Sorted.shape))
print()

# Generate masks for various metrics (for augmented data set)
arrFalsePositivesMask = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['interictal'][1], arrTestResults_Sorted[:, 2] == dio.dctSegStates['ictal'][1])       # False positive (interictal (0) -> ictal (2))
arrFalseNegativesMask = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['ictal'][1], arrTestResults_Sorted[:, 2] == dio.dctSegStates['interictal'][1])       # False negative (ictal (2) -> interictal (0))
arrTruePositivesMask  = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['ictal'][1], arrTestResults_Sorted[:, 2] == dio.dctSegStates['ictal'][1])            # True positive (ictal (2) -> ictal (2))
arrTrueNegativesMask  = np.logical_and(
    arrTestResults_Sorted[:, 1] == dio.dctSegStates['interictal'][1], arrTestResults_Sorted[:, 2] == dio.dctSegStates['interictal'][1])  # True negative (interictal (0) -> interictal (0))

intTestSetSize = arrTestResults_Sorted.shape[0]

# Calculate various metrics (for augmented data set)
intNumFalsePositives = arrTestResults_Sorted[arrFalsePositivesMask].shape[0]
intNumFalseNegatives = arrTestResults_Sorted[arrFalseNegativesMask].shape[0]
intNumTruePositives  = arrTestResults_Sorted[arrTruePositivesMask].shape[0]
intNumTrueNegatives  = arrTestResults_Sorted[arrTrueNegativesMask].shape[0]

intNumCorrect = intTestSetSize - intNumFalsePositives - intNumFalseNegatives

print('intTestSetSize = {}, intNumFalsePositives = {}, intNumFalseNegatives = {}, intNumTruePositives = {}, intNumTrueNegatives = {}'
      .format(intTestSetSize, intNumFalsePositives, intNumFalseNegatives, intNumTruePositives, intNumTrueNegatives))

fltTruePositiveRate, fltTrueNegativeRate = utils.fnCalcPerfMetrics(
    intNumFalsePositives, intNumFalseNegatives, intNumTruePositives, intNumTrueNegatives)

# Print the test accuracy over all test data (sanity check) and other performance metrics
fltTestAccuracy = intNumCorrect / intTestSetSize
print('Test Accuracy = {}/{} = {:.4f}'.format(int(round(intNumCorrect)), intTestSetSize, fltTestAccuracy))
print('fltTruePositiveRate = {:.4f}, fltTrueNegativeRate = {:.4f}'.format(fltTruePositiveRate, fltTrueNegativeRate))
print()

# Visualize test results in a tabular format, including the other related lists/arrays (for augmented data set)

# NOTE: Since some of the fields (e.g. filenames and labels) are in string format, the
#       entire table is converted to string, including the sequence numbers and unique
#       ID. Therefore, arrCompleteTestResults[] is for visualization only. All other
#       types of operations should be performed on the uncombined lists/arrays
arrCompleteTestResults = np.concatenate((arrLabeledTestFilnames_Tested[:, np.newaxis], 
                                         arrLabeledTestSegLabels_Tested[:, np.newaxis], 
                                         arrLabeledTestSegTypes_Tested[:, np.newaxis], 
                                         arrLabeledTestSequences_Tested[:, np.newaxis], 
                                         arrLabeledTestSubSequences_Tested[:, np.newaxis], 
                                         arrLabeledTestUIDs_Tested[:, np.newaxis], 
                                         arrLabeledTestUUIDs_Tested[:, np.newaxis], 
                                         arrLabeledTestStartEndTimesSec_Tested, 
                                         arrTestResults_Sorted
                                        ), axis = 1)
print('arrCompleteTestResults.shape = {}'.format(arrCompleteTestResults.shape))
print()

print('Entries with false positives (interictal predicted as ictal):\n{}'.format(arrCompleteTestResults[arrFalsePositivesMask][0:10, :]))
print()

print('Entries with false negatives (ictal predicted as interictal):\n{}'.format(arrCompleteTestResults[arrFalseNegativesMask][0:10, :]))
print()


# In[ ]:


# Perform post-testing errror analyses (for non-augmented data set)

# Find the list of unique UID entries, which will exclude any augmented entries
arrLabeledTestUIDsOut_TestedNoAug, arrLabeledTestUIDsIdx_TestedNoAug, arrLabeledTestUIDsCount_TestedNoAug = np.unique(
    arrLabeledTestUIDs_Tested, return_index = True, return_inverse = False, return_counts = True, axis = 0)
print('np.unique() results = {}, {}, {}'.format(arrLabeledTestUIDsOut_TestedNoAug.shape, arrLabeledTestUIDsIdx_TestedNoAug.shape, arrLabeledTestUIDsCount_TestedNoAug.shape))
print()

# Remove replicated augmented entries based on their UUIDs by keeping only the unique entries
arrLabeledTestFilnames_TestedNoAug     = arrLabeledTestFilnames_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestSegLabels_TestedNoAug    = arrLabeledTestSegLabels_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestSegTypes_TestedNoAug     = arrLabeledTestSegTypes_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestSequences_TestedNoAug    = arrLabeledTestSequences_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestSubSequences_TestedNoAug = arrLabeledTestSubSequences_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestUIDs_TestedNoAug         = arrLabeledTestUIDs_Tested[arrLabeledTestUIDsIdx_TestedNoAug]
arrLabeledTestUUIDs_TestedNoAug        = arrLabeledTestUUIDs_Tested[arrLabeledTestUIDsIdx_TestedNoAug]

arrLabeledTestStartEndTimesSec_TestedNoAug = arrLabeledTestStartEndTimesSec_Tested[arrLabeledTestUIDsIdx_TestedNoAug, :]

arrTestResults_SortedNoAug = arrTestResults_Sorted[arrLabeledTestUIDsIdx_TestedNoAug, :]

print('Are arrLabeledTestUIDsOut_TestedNoAug[] and arrLabeledTestUIDs_TestedNoAug[] identical? -> {}'.format(
    np.array_equal(arrLabeledTestUIDsOut_TestedNoAug, arrLabeledTestUIDs_TestedNoAug)))
print()

# Generate masks for various metrics (for non-augmented data set)
arrFalsePositivesMask_NoAug = np.logical_and(
    arrTestResults_SortedNoAug[:, 1] == dio.dctSegStates['interictal'][1], arrTestResults_SortedNoAug[:, 2] == dio.dctSegStates['ictal'][1])       # False positive (interictal (0) -> ictal (2))
arrFalseNegativesMask_NoAug = np.logical_and(
    arrTestResults_SortedNoAug[:, 1] == dio.dctSegStates['ictal'][1], arrTestResults_SortedNoAug[:, 2] == dio.dctSegStates['interictal'][1])       # False negative (ictal (2) -> interictal (0))
arrTruePositivesMask_NoAug  = np.logical_and(
    arrTestResults_SortedNoAug[:, 1] == dio.dctSegStates['ictal'][1], arrTestResults_SortedNoAug[:, 2] == dio.dctSegStates['ictal'][1])            # True positive (ictal (2) -> ictal (2))
arrTrueNegativesMask_NoAug  = np.logical_and(
    arrTestResults_SortedNoAug[:, 1] == dio.dctSegStates['interictal'][1], arrTestResults_SortedNoAug[:, 2] == dio.dctSegStates['interictal'][1])  # True negative (interictal (0) -> interictal (0))

intTestSetSize_NoAug = arrTestResults_SortedNoAug.shape[0]

# Calculate various metrics (for non-augmented data set)
intNumFalsePositives_NoAug = arrTestResults_SortedNoAug[arrFalsePositivesMask_NoAug].shape[0]
intNumFalseNegatives_NoAug = arrTestResults_SortedNoAug[arrFalseNegativesMask_NoAug].shape[0]
intNumTruePositives_NoAug  = arrTestResults_SortedNoAug[arrTruePositivesMask_NoAug].shape[0]
intNumTrueNegatives_NoAug  = arrTestResults_SortedNoAug[arrTrueNegativesMask_NoAug].shape[0]

intNumCorrect_NoAug = intTestSetSize_NoAug - intNumFalsePositives_NoAug - intNumFalseNegatives_NoAug

print('intTestSetSize_NoAug = {}, intNumFalsePositives_NoAug = {}, intNumFalseNegatives_NoAug = {}, intNumTruePositives_NoAug = {}, intNumTrueNegatives_NoAug_NoAug = {}'
      .format(intTestSetSize_NoAug, intNumFalsePositives_NoAug, intNumFalseNegatives_NoAug, intNumTruePositives_NoAug, intNumTrueNegatives_NoAug))

fltTruePositiveRate_NoAug, fltTrueNegativeRate_NoAug = utils.fnCalcPerfMetrics(
    intNumFalsePositives_NoAug, intNumFalseNegatives_NoAug, intNumTruePositives_NoAug, intNumTrueNegatives_NoAug)

# Print the test accuracy over all test data (sanity check) and other performance metrics
fltTestAccuracy_NoAug = intNumCorrect_NoAug / intTestSetSize_NoAug
print('Test Accuracy (NoAug) = {}/{} = {:.4f}'.format(int(round(intNumCorrect_NoAug)), intTestSetSize_NoAug, fltTestAccuracy_NoAug))
print('fltTruePositiveRate_NoAug = {:.4f}, fltTrueNegativeRate_NoAug = {:.4f}'.format(fltTruePositiveRate_NoAug, fltTrueNegativeRate_NoAug))
print()

# Visualize test results in a tabular format, including the other related lists/arrays (for non-augmented data set)
arrCompleteTestResults_NoAug = np.concatenate((arrLabeledTestFilnames_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestSegLabels_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestSegTypes_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestSequences_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestSubSequences_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestUIDs_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestUUIDs_TestedNoAug[:, np.newaxis], 
                                               arrLabeledTestStartEndTimesSec_TestedNoAug, 
                                               arrTestResults_SortedNoAug
                                              ), axis = 1)
print('arrCompleteTestResults_NoAug.shape = {}'.format(arrCompleteTestResults_NoAug.shape))
print()

print('Entries with false positives (interictal predicted as ictal):\n{}'.format(arrCompleteTestResults_NoAug[arrFalsePositivesMask_NoAug, :][0:100, :]))
print()

print('Entries with false negatives (ictal predicted as interictal):\n{}'.format(arrCompleteTestResults_NoAug[arrFalseNegativesMask_NoAug, :][0:100, :]))
print()


# In[ ]:


if (argSaveAnno):
    # Save test results to annotation files for visualization in EDFbrowser
    strTestAnnoPath = './SavedAnno/' + strTimestamp + '/'
    print('Saving test result annotation files to: {}'.format(strTestAnnoPath))
    print()
    
    dio.fnWriteTestAnnoFiles(
        arrLabeledTestFilnames_TestedNoAug, arrTestResults_SortedNoAug, arrLabeledTestStartEndTimesSec_TestedNoAug, arrFalsePositivesMask_NoAug, arrFalseNegativesMask_NoAug, strTestAnnoPath)


# In[ ]:


if (blnBatchMode):
    # Close the log file and redirect output back to stdout and stderr
    datScriptEnd = utils.fnNow()
    print('Script ended on {}'.format(utils.fnGetDatetime(datScriptEnd)))

    datScriptDuration = datScriptEnd - datScriptStart
    print('datScriptDuration = {}'.format(datScriptDuration))

    objLogFile.close()
    sys.stdout = objStdout
    sys.stderr = objStderr

