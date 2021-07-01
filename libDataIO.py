#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv, os
import scipy.io as spio
from scipy import signal
import numpy as np
import statistics as stat
import re, math
import matplotlib.pyplot as plt
import pprint as pp
from operator import itemgetter

import libUtils as utils


# In[ ]:


# EEG segment state definition and associate functions

# Dictionary that defines state keys and values (tuple that contains
# the segment label and segment type)
dctSegStates = {
  'interictal': ('interictal', 0),
  'preictal': ('preictal', 1),
  'ictal': ('ictal', 2)
}

# Return the segmentation label given the segmentation type
def fnGetSegLabel(argSegType):
    lstSegValues = list(dctSegStates.values())  # Get the dictionary values as a list of tuples
    dctSegValues = dict(lstSegValues)  # Create a dictionary from a list of tuples
    
    # Create an inverted dictionary from dctSegValues{}
    dctInvertedSegValues = {intValue: strKey for strKey, intValue in dctSegValues.items()}
    
    return dctInvertedSegValues[argSegType]

# Return the segmentation type given the segmentation label
def fnGetSegType(argSegLabel):
    lstSegValues = list(dctSegStates.values())  # Get the dictionary values as a list of tuples
    dctSegValues = dict(lstSegValues)  # Create a dictionary from a list of tuples
    
    return dctSegValues[argSegLabel]

def fnIsSegState(argSegState, argSegType, argDebug = False):
    if (argSegState in dctSegStates):
        tupSegState = dctSegStates[argSegState]
        
        if (argDebug):
            print('argSegState = {}, argSegType = {}'.format(dctSegStates[argSegState], argSegType))
            
        if (tupSegState[1] == argSegType):
            blnResult = True
        else:
            blnResult = False
    else:
        raise Exception('Unknown argSegState ({})'.format(argSegState))
        
    return blnResult


# In[ ]:


# Get the data set name and training set name for each training based
# on the CSV file that specifies which EEG segment to use for training

# For example: if the .csv file is called, .../CHB-MIT/chb01.csv, we
# will extract CHB-MIT as the data set name and chb01 as the CSV name

def fnGetDataSetInfo(argCSVPath, argDebug = False):
    strParentPath, strFilename = os.path.split(argCSVPath)
    
    strDataSetName = os.path.basename(strParentPath)
    strCSVName, strFileExt = os.path.splitext(strFilename)
    
    return strDataSetName, strCSVName


# In[ ]:


# Read a CSV file that specifies which EEG segment(s) to use for training

# The file is expected to be comma-delimited. The first row is assumed
# to be the header, containing a list of column names for the different
# CSV fields (the first field is the full path filename of the EEG
# segment)

# Each remaining row will contain, in the first field, the full path of
# the EEG segment

# Comment lines (starting with #) and blank lines are allowed, and will
# be ignored

# TODO: Add an optional argFileExt filter to allow the filtering of file
#       type. If none is specified, then no filtering is applied

def fnReadDataFileListCSV(argCSVPath, argInfo = True, argDebug = False):
    print('Reading data from CSV file at {}'.format(argCSVPath))
    print()
    
    # Get the list of files from the CSV file. The files are not necessarily
    # ordered in any specific order
    lstMatchingFiles = []
    
    # Loop through each row in the CSV file
    with open(argCSVPath) as objCSVFile:
        objCSVReader = csv.reader(objCSVFile, delimiter = ',')
        intRow = 0

        for lstRow in objCSVReader:  # Read in the row from file
            # Skip the first row of column names
            if (intRow == 0):
                if (argInfo): print('Skipping the first row. Column names are: [{}]'.format(', '.join(lstRow)))
            else:
                if (argDebug): print('lstRow = [{}] (len(lstRow) = {}) (type(lstRow) = {})'.format(lstRow, len(lstRow), type(lstRow)))
                
                # Skip empty rows, which are lists with len() = 0
                if (len(lstRow) > 0):
                    strFullFilename = lstRow[0]  # Read in the first list item from lstRow[]

                    # Skip any rows that are comments (starting with a #)
                    if not(re.match(r'^#', strFullFilename, re.I)):  # r'' means raw string
                        lstMatchingFiles.append(strFullFilename)  # Add the filename to the list of machine files
                        if (argInfo): print('  {}'.format(strFullFilename))

            intRow += 1
            
    if (argInfo or argDebug): print()
    
    return sorted(lstMatchingFiles)


# In[ ]:


# Read in one or more EEG segments from the Kaggle prediction data set
# from files (.mat) based on which files are specified in argCSVPath

# The units of argResamplingFreq is in Hz and argSubSeqDuration is in
# seconds

'''
Within the .mat data structure:

data:               a matrix of EEG sample values arranged row x column as electrode x time
data_length_sec:    the time duration of each data row
sampling_frequency: the number of data samples representing 1 second of EEG data
channels:           a list of electrode names corresponding to the rows in the data field
sequence:           the index of the data segment within the one hour series of clips. For example, 
                    preictal_segment_6.mat has a sequence number of 6, and represents the iEEG data 
                    from 50 to 60 minutes into the preictal data
'''

def fnReadKagglePredMatFiles(argCSVPath, argResamplingFreq = -1, argSubSeqDuration = -1, argDebug = False):
    lstMatchingFiles = fnReadDataFileListCSV(argCSVPath)
    
    intNumMatchingFiles = len(lstMatchingFiles)  # Number of matching files
    if (argDebug): print('intNumMatchingFiles = {}'.format(intNumMatchingFiles))
        
    if (argDebug): print()
    
    intNumProcessedFiles = 0
    intAllDataIdx        = 0
    
    # Import the first .mat file to a Python dictionary to get the shape
    # of the data array
    matSegment = spio.loadmat(lstMatchingFiles[0])
    strSegLabel = sorted(matSegment.keys())[3]
    arrData = matSegment[strSegLabel]['data'].item()
    if (argDebug): print('arrData.shape = {} ({})'.format(arrData.shape, type(arrData[0][0])))
    if (argDebug): print('First matching file = {}'.format(lstMatchingFiles[0]))
    
    # Get the original segment duration and sampling frequency from the raw data
    intSegDuration = matSegment[strSegLabel]['data_length_sec'].item().item()  # TODO: Segment duration could be a float
    print('intSegDuration = {}s'.format(intSegDuration))
    intSamplingFreq = matSegment[strSegLabel]['sampling_frequency'].item().item()  # TODO: Sampling frequency could be a float
    print('intSamplingFreq = {}Hz'.format(intSamplingFreq))
    
    # Initialize data structures to store data for the entire data set
    lstAllSegLabels = []      # List of segment labels
    lstAllSegTypes = []       # List of segment types (preictal = 1 or interictal = 2)
    
    # Use * to unpack the shape tuple and create arrAllData[] using the same dtype as arrData[]
    # Resulting shape of data = [feature/channel size x sequence length x batch/segment size]
    #arrAllData = np.zeros((*arrData.shape, intNumMatchingFiles), dtype = type(arrData[0][0]))
    
    # Get the number of features/channels (rows) and time points (cols)
    # from the original segment stored in arrData[]
    intNumChannels, intNumTimePts = arrData.shape
    
    # TODO: Double check that intNumTimePts = intSegDuration * intSamplingFreq?
    
    # If argResamplingFreq = -1 (defautl) or argResamplingFreq > intSamplingFreq,
    # use the original sampling frequency (upsampling is not allowed for now)
    if ((argResamplingFreq == -1) or (argResamplingFreq > intSamplingFreq)):
        argResamplingFreq = intSamplingFreq
    print('argResamplingFreq = {}Hz'.format(argResamplingFreq))
    
    # Calculate the number of time points in each segment after the resampling
    intResampledTimePts = round((argResamplingFreq / intSamplingFreq) * intNumTimePts)
    print('intResampledTimePts = {}'.format(intResampledTimePts))
    
    # If argSubSeqDuration = -1 (default) or argSubSeqDuration > intSegDuration,
    # default back to the original segment duration
    if ((argSubSeqDuration == -1) or (argSubSeqDuration > intSegDuration)):
        argSubSeqDuration = intSegDuration  # Do not break segment into subsequences
    print('argSubSeqDuration = {}s'.format(argSubSeqDuration))
        
    # Calculate the number of time points in each segment after splitting up the segment
    # into shorter sequences
    intSubSeqTimePts = round((argSubSeqDuration / intSegDuration) * intResampledTimePts)
    print('intSubSeqTimePts = {}'.format(intSubSeqTimePts))
    
    # The default value for argSubSeqTimePts is -1, which is the number of time
    # points in the original segment, without breaking it up into sebsequences
    #if ((argSubSeqTimePts == -1) or (argSubSeqTimePts > intNumTimePts)):
    #    argSubSeqTimePts = intNumTimePts  # Do not break segment into subsequences
    
    # ***TODO: For now, if the number of subsequence time points specified results
    #          in the last subsequence not being completely filled up, we default
    #          back to not breaking up the segment into subsequences. We can think
    #          about how to deal with the last unfilled subsequence later if we do
    #          not like this policy
    if (intResampledTimePts % intSubSeqTimePts > 0):
        intSubSeqTimePts = intResampledTimePts  # Do not break segment into subsequences
        
        print('WARNING: Time points cannot be divided into complete subsequences')
        print('  Resetting intSubSeqTimePts to {}'.format(intSubSeqTimePts))
        
    # Analyze the number of subsequences to break down from the main segment, and
    # whether the time points can be equally divided among all the subsequences
    intNumSubSeqs = math.ceil(intResampledTimePts / intSubSeqTimePts)  # Total number of subsequences required
    intNumFullSubSeqs = intResampledTimePts // intSubSeqTimePts        # Number of completely filled subsequences
    intNumOrphanTimePts = intResampledTimePts % intSubSeqTimePts       # Number of orphan time points
    
    print()
    
    if (intNumSubSeqs > 1):
        print('Splitting each segment into {} subsequences based on intSubSeqTimePts = {}'.format(intNumSubSeqs, intSubSeqTimePts))
        print()
    
    if (argDebug):
        print('intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts = {} / {} / {} ({:.2f}%)'.format(intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts, intNumOrphanTimePts/intSubSeqTimePts))
        print()
    
    # Create an array to hold all data segments, taking into account that we may
    # have subsequences and multiple data files
    
    # NOTE: Depending on the max and min values in arrData[] for a particular segment,
    #       type(arrData[0][0]) could be int16, int32, or int64. Therefore, it is not
    #       a good idea to create arrAllData[] based on the first arrData[] read in.
    #       It may be safer to hardcode arrAllData[] to be of type int64
    intTotalBatchSize = intNumMatchingFiles * intNumSubSeqs  # Total number of sequences
    #arrAllData = np.zeros((intNumChannels, intSubSeqTimePts, intTotalBatchSize), dtype = type(arrData[0][0]))
    arrAllData = np.zeros((intNumChannels, intSubSeqTimePts, intTotalBatchSize), dtype = np.int64)  # TODO: Signal values could be a float
    if (argDebug): print('arrAllData.shape = {} ({})'.format(arrAllData.shape, type(arrAllData[0][0][0])))
    
    lstAllSegDurations = []   # List of segment durations (in seconds)
    lstAllSamplingFreqs = []  # List of sampling frequencies (in Hz)
    lstAllChannels = []       # List of lists (of channel names)
    lstAllSequences = []      # List of sequence indices
    lstAllSubSequences = []   # List of subsequence indices (if one sequence is broken up into subsequences)
    
    if (argDebug): print()
    
    lstBaseFilenames = []
    
    # Loop through each file in the target directory                                                
    for strFullFilename in lstMatchingFiles:
        # Process only .mat files
        if strFullFilename.endswith('.mat'):
            print('Processing {}...'.format(strFullFilename))
            
            # Extract the filename from the full path
            strPath, strFilename = os.path.split(strFullFilename)
            lstBaseFilenames.append(strFilename)
            
            # Import .mat file to a Python dictionary
            matSegment = spio.loadmat(strFullFilename)
            
            # The list of dictionary keys always contain ['__globals__',
            # '__header__', '__version__', 'segment_label'], so I assume
            # the 4th item contains the segment label after sorting
            strSegLabel = sorted(matSegment.keys())[3]
            if (argDebug): print('  strSegLabel = {}'.format(strSegLabel))
            
            # Classes: interictal = 0, preictal = 1
            if (re.match(r'interictal', strSegLabel, re.I)):  # r'' means raw string
                intSegType = '0'  # Interictal state
            elif (re.match(r'preictal', strSegLabel, re.I)):
                intSegType = '1'  # Preictal state
            else:
                intSegType = '-1'  # Unknown segment type
            
            # matSegment[strSegLabel] is a structured numpy array with
            # the data types 'data', 'data_length_sec', 'sampling_frequency',
            # 'channels', and 'sequence' (not for test segments)
            arrData = matSegment[strSegLabel]['data'].item()
            if (argDebug): print('  arrData.shape = {} ({})'.format(arrData.shape, type(arrData[0][0])))
            #if (argDebug): print(arrData[0:2, 0:10])
            
            arrDataMax  = np.max(arrData, axis = 1)
            arrDataMin  = np.min(arrData, axis = 1)
            arrDataMean = np.mean(arrData, axis = 1)
            
            intSegDuration = matSegment[strSegLabel]['data_length_sec'].item().item()
            if (argDebug): print('  intSegDuration = {}s'.format(intSegDuration))
            
            intSamplingFreq = matSegment[strSegLabel]['sampling_frequency'].item().item()
            if (argDebug): print('  intSamplingFreq = {}Hz'.format(intSamplingFreq))
            
            lstChannels = []
            arrChannels = np.squeeze(matSegment[strSegLabel]['channels'].item())
            for arrChannel in arrChannels:
                lstChannels.append(arrChannel.item())
            if (argDebug):
                print('  arrChannels.shape = {} ({})'.format(arrChannels.shape, type(arrChannels[0].item())))
                print('  arrChannels[0] = {}'.format(arrChannels[0].item()))
                print('  lstChannels = {}'.format(lstChannels))
            
            # See if matSegment[strSegLabel] has the 'sequence' data type.
            # Looks like using dtype.names and dtype.fields give the same
            # result -> what is the difference between the two?
            if ('sequence' in matSegment[strSegLabel].dtype.names):
                intSequence = matSegment[strSegLabel]['sequence'].item().item()
            else:
                intSequence = -1
            if (argDebug): print('  intSequence = {}'.format(intSequence))
            
            print('  {}\t{}\t{}s\t{}Hz\t{}\t{}\t'.format(strSegLabel, arrData.shape, intSegDuration,
                                                         intSamplingFreq, arrChannels.shape, intSequence))
            
            #for tupZip in zip(lstChannels, arrDataMax, arrDataMin, arrDataMean):
            #    print('    {}\t{}\t{}\t{:.4f}'.format(*tupZip))
            #
            #print()
            
            # Resample and round the results to the nearest integer since arrData[] is of type int16
            arrDataResampled = np.rint(signal.resample(arrData, intResampledTimePts, axis = 1))  # Round all values to integers
            if (argDebug): print('    arrDataResampled.shape = {} ({})'.format(arrDataResampled.shape, type(arrDataResampled[0][0])))
            arrDataResampled = arrDataResampled.astype(int)  # Cast all values to type int
            if (argDebug): print('    arrDataResampled.shape = {} ({})'.format(arrDataResampled.shape, type(arrDataResampled[0][0])))
            
            arrDataResampledMax  = np.max(arrDataResampled, axis = 1)
            arrDataResampledMin  = np.min(arrDataResampled, axis = 1)
            arrDataResampledMean = np.mean(arrDataResampled, axis = 1)
            
            for tupZip in zip(lstChannels, arrDataResampledMax, arrDataResampledMin, arrDataResampledMean):
                print('    {}\t{}\t{}\t{:.4f}'.format(*tupZip))
            
            # Reshape arrData[channels, time pts] into arrDataSplit[channels, subseq time pts, subsequences]
            # where the subsequences are split along the 3rd dimension
            arrDataSplit = arrDataResampled.reshape(intNumChannels, -1, intSubSeqTimePts)
            arrDataSplit = np.swapaxes(arrDataSplit, 1, 2)
            if (argDebug): print('    arrDataSplit.shape = {} ({})'.format(arrDataSplit.shape, type(arrDataSplit[0][0][0])))
            
            fltTotalDiff = 0
            
            # Loop through each subsequence and save the data and metadata into
            # the appropriate data structures
            for intSubSequence in np.arange(intNumSubSeqs):
                # Group all segments processed into groups of data structures
                lstAllSegLabels.append(strSegLabel)
                lstAllSegTypes.append(intSegType)
                #arrAllData[:, :, intNumProcessedFiles] = arrData
                arrAllData[:, :, intAllDataIdx] = arrDataSplit[:, :, intSubSequence]
                #lstAllSegDurations.append(intSegDuration)
                lstAllSegDurations.append(argSubSeqDuration)   # Record the duration after the split
                #lstAllSamplingFreqs.append(intSamplingFreq)
                lstAllSamplingFreqs.append(argResamplingFreq)  # Record the resampled frequency
                lstAllChannels.append(lstChannels)
                lstAllSequences.append(intSequence)
                lstAllSubSequences.append(intSubSequence)
                
                if (argDebug): print('    intSubSeg, intAllDataIdx = {}, {}'.format(intSubSequence, intAllDataIdx))
                
                # Test code to check that the data integrity is intact after the resampling and
                # after the data split
                if (argDebug):
                    intStartIdx  = intSubSequence * intSubSeqTimePts
                    intEndIdx    = (intSubSequence + 1) * intSubSeqTimePts
                    fltDiffSplit = sum(arrDataResampled[0, intStartIdx:intEndIdx] - arrDataSplit[0, 0:intSubSeqTimePts, intSubSequence])
                    fltDiffAll   = sum(arrDataSplit[0, 0:intSubSeqTimePts, intSubSequence] - arrAllData[0, 0:intSubSeqTimePts, intAllDataIdx])
                    fltTotalDiff = fltTotalDiff + fltDiffSplit + fltDiffAll
                
                if (argDebug): print('      intStartIdx = {}, intEndIdx = {}, fltDiffSplit = {}, fltDiffAll = {}'.format(intStartIdx, intEndIdx, fltDiffSplit, fltDiffAll))
                
                intAllDataIdx = intAllDataIdx + 1
            
            if (argDebug): print('    fltTotalDiff = {}'.format(fltTotalDiff))
            
            intNumProcessedFiles = intNumProcessedFiles + 1
            
    print('intNumProcessedFiles = {}'.format(intNumProcessedFiles))
    
    # Make sure that the number of files that matched the specified
    # extension is the same number of files that we actually processed
    if (intNumMatchingFiles != intNumProcessedFiles):
        print('WARNING: intNumMatchingFiles != intNumProcessedFiles!')
    
    print()
    
    print('len(lstAllSegLabels) = {}'.format(len(lstAllSegLabels)))
    print('len(lstAllSegTypes) = {}'.format(len(lstAllSegTypes)))
    print('arrAllData.shape = {} ({}) (features x sequence length x batch size)'.format(arrAllData.shape,  type(arrAllData[0][0][0])))
    print('len(lstAllSegDurations) = {}'.format(len(lstAllSegDurations)))
    print('len(lstAllSamplingFreqs) = {}'.format(len(lstAllSamplingFreqs)))
    print('len(lstAllChannels) = {}'.format(len(lstAllChannels)))
    print('len(lstAllSequences) = {}'.format(len(lstAllSequences)))
    print('len(lstAllSubSequences) = {}'.format(len(lstAllSubSequences)))

    if (argDebug):
        print('lstAllSegLabels = {}'.format(lstAllSegLabels))
        print('lstAllSegTypes = {}'.format(lstAllSegTypes))
        print('arrAllData = {}'.format(arrAllData))
        print('lstAllSegDurations = {}'.format(lstAllSegDurations))
        print('lstAllSamplingFreqs = {}'.format(lstAllSamplingFreqs))
        print('lstAllChannels = {}'.format(lstAllChannels))
        print('lstAllSequences = {}'.format(lstAllSequences))
        print('lstAllSubSequences = {}'.format(lstAllSubSequences))
    
    print()
    
    return lstBaseFilenames, lstAllSegLabels, lstAllSegTypes, arrAllData, lstAllSegDurations, lstAllSamplingFreqs, lstAllChannels, lstAllSequences, lstAllSubSequences


# In[ ]:


import pyedflib as pyedf

# Read a single EDF file using PyEDFlib
def fnReadEDFUsingPyEDFLib(argFullFilename, argPerformChecks = True, argReturnHeader = False, argNoData = False, argDebug = False):
    # There is no need to call close() when using 'with' to open
    # files
    with pyedf.EdfReader(argFullFilename) as fhEDFFile:
        
        # The following are EDF-specific fields
        objStartDatetime  = fhEDFFile.getStartdatetime()  # This is a datetime object
        dictHeader        = fhEDFFile.getHeader()  # The EDF header is returned as a dictionary
        
        # TODO: Fill this in as 'interictal' if segment is outside of
        #       the seizure annotation zone, and 'ictal' if within the
        #       zone. How to label for 'preictal' is yet to be determined
        
        # This variable currently returns a dummy empty value, and
        # the true segment label is determined by fnBreakCHBMITSegment()
        # since each EDF file can be broken into segments of multiple
        # EEG states based on whether there are seizure episodes
        strSegLabel       = ''
        
        lstChannels = []
        lstDiscardChannelID = []
        intNumChannels = 0
        intChannelOrigID = 0
        
        lstChannelsOrig   = fhEDFFile.getSignalLabels()
        
        # ***NOTE: Maybe we need to move this code one level up or put
        #          it under a switch so that the check is data set
        #          dependent
        
        # Exclude non-EEG channels
        # Not an exhausted list here
        # So far, we know those channels can be ECG, EKG and VNS
        for strChannelLabel in lstChannelsOrig:
            objInvalidCh = re.match(r'.*E[CK]G.*|.*VNS.*|\s*-\s*|\s*\.\s*', strChannelLabel)
            if (objInvalidCh):
                lstDiscardChannelID.append(intChannelOrigID)
                intChannelOrigID += 1
                continue
                
            lstChannels.append(strChannelLabel)
            intNumChannels   += 1
            intChannelOrigID += 1
            
        if (argDebug and len(lstDiscardChannelID) > 0):
            print('Discarded channels = {} (Indices = {})'.format(list(itemgetter(*lstDiscardChannelID)(lstChannelsOrig)), lstDiscardChannelID))
            print('lstChannels = {}'.format(lstChannels))
            print()
            
        lstChannelHeaders = fhEDFFile.getSignalHeaders()  # EDF-specific field (a list of dictionaries)
        lstChannelHeaders = np.delete(lstChannelHeaders, lstDiscardChannelID)  # Remove discarded channels
        
        intNumChannelsOrig = fhEDFFile.signals_in_file  # Number of signals in the EDF file
        
        arrNumTimePts      = fhEDFFile.getNSamples()
        arrNumTimePts      = np.delete(arrNumTimePts, lstDiscardChannelID)  # Remove discarded channels
        
        # Check for consistency of number of time points across all
        # channels within the EDF file
        intNumTimePts = 0
        if (utils.fnAllIdentical1D(arrNumTimePts)):
            intNumTimePts = arrNumTimePts[0]
        else:
            raise Exception('intNumTimePts are not identical across all channels!')
        
        arrData = np.zeros((intNumChannels, arrNumTimePts[0]), dtype = np.float64)
        
        # ***TODO: This loop is currently taking the most time to execute in this
        #          function. I added a loop so that we can choose not to get data
        #          from the EDF file. However, we may want to see if we can improve
        #          the performance anyway
        if (not argNoData):
            intChannel = 0
            # Loop through each channel and read the signal data into arrData[]
            for intChannelOrig in np.arange(intNumChannelsOrig):
                # if the channel is in the discard channel list, do not include its data
                if (intChannelOrig not in lstDiscardChannelID):
                    arrData[intChannel, :] = fhEDFFile.readSignal(intChannelOrig)
                    intChannel += 1
            
        fltSegDuration    = fhEDFFile.getFileDuration()
        
        arrSamplingFreqs  = fhEDFFile.getSampleFrequencies()
        arrSamplingFreqs  = np.delete(arrSamplingFreqs, lstDiscardChannelID)  # Remove discarded channels
        
        # Check for consistency of the sampling frequency across all
        # channels within the EDF file
        fltSamplingFreq = 0
        if (utils.fnAllIdentical1D(arrSamplingFreqs)):
            fltSamplingFreq = arrSamplingFreqs[0]
        else:
            raise Exception('fltSamplingFreqs are not identical across all channels!')
            
        # ***NOTE: With more data sets sharing this function, maybe we can expand
        #          argPerformCheck to take additional values, such as 0 = no check,
        #          1 = CHB-MIT, 2 = New_Data_2020, 3 = ...
        
        if (argPerformChecks):
            # Extract sequence number from filename
            objMatch = re.match(r'.+chb\d+[a-zA-Z]*_(\d+)[\+a-zA-Z]*\.edf', argFullFilename)  # ***BUG: Sequence number may be not alphanumeric?
            if (objMatch == None):
                raise Exception('argFullFilename does not match the required pattern!')

            intSequence = int(objMatch.group(1))
        else:
            intSequence = 0
        
        if (argDebug):
            print('objStartDatetime = {}\n'.format(objStartDatetime))
            print('dictHeader =\n{}\n'.format(dictHeader))
            print('intNumChannels = {}\n'.format(intNumChannels))
            print('arrNumTimePts =\n{}\n'.format(arrNumTimePts))
            print('intNumTimePts = {}\n'.format(intNumTimePts))
            print('arrData.shape = {}, type(arrData[0, 0]) = {}\n'.format(arrData.shape, type(arrData[0, 0])))
            print('fltSegDuration = {}s\n'.format(fltSegDuration))
            print('arrSamplingFreqs =\n{}\n'.format(arrSamplingFreqs))
            print('fltSamplingFreq = {}Hz\n'.format(fltSamplingFreq))
            print('lstChannels =\n{}\n'.format(lstChannels))
            print('len(lstChannelHeaders) = {}'.format(len(lstChannelHeaders)))
            print('lstChannelHeaders[0] =\n{}\n'.format(lstChannelHeaders[0]))
            print('intSequence = {}\n'.format(intSequence))
            print('arrData[0, 0:50] =\n{}\n'.format(arrData[0, 0:50]))
            
    if (argReturnHeader):
        return strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts, dictHeader
    else:
        return strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts        


# In[ ]:


def fnMatchEDFChannels(argFullEDFFiles, argDebug = False):

    lstAllChannels   = []
    blnMismatchFound = False
    lstMismatches    = ['   ']
    
    # Loop through each file in argFullEDFFiles
    for intFullFilenameIdx, strFullFilename in enumerate(argFullEDFFiles):
        #print(intFullFilenameIdx, strFullFilename)
        
        # Get the data from each EDF file
        strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts, dictEDFHeader = fnReadEDFUsingPyEDFLib(
            strFullFilename, argPerformChecks = True, argReturnHeader = True, argNoData = True, argDebug = False)
            
        lstAllChannels.append(lstChannels)
        
        if (intFullFilenameIdx > 0):
            if (lstAllChannels[intFullFilenameIdx] != lstAllChannels[intFullFilenameIdx - 1]):
                blnMismatchFound = True
                lstMismatches.append('***')
            else:
                lstMismatches.append('   ')
                
    if (blnMismatchFound or argDebug):
        for intFullFilenameIdx, strFullFilename in enumerate(argFullEDFFiles):
            print('In {}:\n{} Ch = {}'.format(strFullFilename, lstMismatches[intFullFilenameIdx], lstAllChannels[intFullFilenameIdx]))
            print()
    else:
        print('Channels match in all EDF files')
        print()
        
    if (blnMismatchFound):
        raise Exception('Channels do not match in EDF files!')


# In[ ]:


# Given a list of full EDF filenames, extract the statistics (min, max, and mean)
# from each file (N) and each channel (C), returning three arrays of statistics
# in C x N format. The statistics can be used for input data scaling
def fnGetCHBMITStats(argFullTrainingFiles, argFullTestFiles = [], argDebug = False):

    # Use the first EDF file in the list to get the number of channels. Assuming that
    # the files in the rest of the list have the same channel arrangements and sampling
    # rates
    strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts, dictEDFHeader = fnReadEDFUsingPyEDFLib(
        argFullTrainingFiles[0], argPerformChecks = True, argReturnHeader = True, argNoData = True, argDebug = False)
        
    intNumTrainingFiles = len(argFullTrainingFiles)
    
    # Assumption: if argFullTestFiles[] is specified, we assume that the number of
    #             channels and the list of channels are the same between the training
    #             and test files (which should be checked by the code that calls this
    #             function)
    if (argFullTestFiles):
        intNumTestFiles = len(argFullTestFiles)
    else:
        intNumTestFiles = 0
        
    intNumTotalFiles = intNumTrainingFiles + intNumTestFiles
    
    # Initialize the data structures for storing the statistics
    arrDataSetMin  = np.zeros((intNumChannels, intNumTotalFiles), dtype = np.float64)
    arrDataSetMax  = np.zeros_like(arrDataSetMin)
    arrDataSetMean = np.zeros_like(arrDataSetMin)
    
    argFullTotalFiles = argFullTrainingFiles + argFullTestFiles
    if (argDebug):
        print('len(argFullTotalFiles) = {}'.format(len(argFullTotalFiles)))
        print()
    
    # Loop through each file in argFullTrainingFiles[] (and argFullTestFiles[], if specified)
    for intFullFilenameIdx, strFullFilename in enumerate(argFullTotalFiles):
        
        # Get the data from each EDF file
        strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts, dictEDFHeader = fnReadEDFUsingPyEDFLib(
            strFullFilename, argPerformChecks = True, argReturnHeader = True, argNoData = False, argDebug = False)
            
        # Get the min, max, and mean for each file and each channel independently
        arrDataMin  = np.min(arrData, axis = 1)
        arrDataMax  = np.max(arrData, axis = 1)
        arrDataMean = np.mean(arrData, axis = 1)
        
        # Group statistics in C (channel) x N (EDF file) 
        arrDataSetMin[:, intFullFilenameIdx]  = arrDataMin[:, np.newaxis].T
        arrDataSetMax[:, intFullFilenameIdx]  = arrDataMax[:, np.newaxis].T
        arrDataSetMean[:, intFullFilenameIdx] = arrDataMean[:, np.newaxis].T
        
        if (argDebug):
            print('({}): strFullFilename = {}'.format(intFullFilenameIdx, strFullFilename))
            print('        arrDataMin = {}'.format(pp.pformat(arrDataMin)))
            print('        arrDataMax = {}'.format(pp.pformat(arrDataMax)))
            print('        arrDataMean = {}'.format(pp.pformat(arrDataMean)))
            print('        np.argmin(arrData, axis = 1) = {}'.format(np.argmin(arrData, axis = 1)))
            print('        np.argmax(arrData, axis = 1) = {}'.format(np.argmax(arrData, axis = 1)))
            print('        arrData[np.arange(arrData.shape[0]), np.argmin(arrData, axis = 1)] = {}'.format(arrData[np.arange(arrData.shape[0]), np.argmin(arrData, axis = 1)]))
            print('        arrData[np.arange(arrData.shape[0]), np.argmax(arrData, axis = 1)] = {}'.format(arrData[np.arange(arrData.shape[0]), np.argmax(arrData, axis = 1)]))
            print()
            
    if (argDebug):
        print('arrDataSetMin = {}'.format(pp.pformat(arrDataSetMin)))
        print('arrDataSetMax = {}'.format(pp.pformat(arrDataSetMax)))
        print('arrDataSetMean = {}'.format(pp.pformat(arrDataSetMean)))
        print()
        
    print('  arrDataSetMin.shape = {}, arrDataSetMax.shape = {}, arrDataSetMean.shape = {}'.format(
        arrDataSetMin.shape, arrDataSetMax.shape, arrDataSetMean.shape))
        
    return arrDataSetMin, arrDataSetMax, arrDataSetMean


# In[ ]:


# Generate arrDataMin and arrDataMax from arrDataSetMin and arrDataSetMax based
# on the desired scaling mode (argScalingMode). All these arrays have the same
# shape, which is C (channel) x N (EDF file). argScalingMode can be specified to
# scale by:
#
#   (1) Per channel and EDF file      (argScalingMode = 0)
#   (2) Per channel, across EDF files (argScalingMode = 1)
#   (3) Across channels, per EDF file (argScalingMode = 2)
#   (4) Across channels and EDF files (argScalingMode = 3)
#
# Examples shown below:

'''
    argDataSetMin     argDataSetMax
    -1, -4, -6, -3     1, 4, 8, 5
    -2,  0, -1, -4     3, 4, 1, 6
    -3, -2, -8, -5     2, 5, 9, 7
    
      argDataMin       argDataMax
    -1, -4, -6, -3     1, 4, 8, 5
    -2,  0, -1, -4     3, 4, 1, 6     -> per channel, per EDF file
    -3, -2, -8, -5     2, 5, 9, 7
    
    -6, -6, -6, -6     8, 8, 8, 8
    -4, -4, -4, -4     6, 6, 6, 6     -> per channel, across EDF files
    -8, -8, -8, -8     9, 9, 9, 9
    
    -3, -4, -8, -5     3, 5, 9, 7
    -3, -4, -8, -5     3, 5, 9, 7     -> across channels, per file
    -3, -4, -8, -5     3, 5, 9, 7
    
    -8, -8, -8, -8     9, 9, 9, 9
    -8, -8, -8, -8     9, 9, 9, 9     -> across channels and files
    -8, -8, -8, -8     9, 9, 9, 9
    
'''

def fnGenMinMaxArrays(argScalingMode, argDataSetMin, argDataSetMax, argDebug = False):
    intNumChannels, intNumEDFFiles = argDataSetMin.shape
    
    # Per channel, across EDF files
    if (argScalingMode == 1):
        arrDataMin = np.repeat(np.min(argDataSetMin, axis = 1)[:, np.newaxis], intNumEDFFiles, axis = 1)
        arrDataMax = np.repeat(np.max(argDataSetMax, axis = 1)[:, np.newaxis], intNumEDFFiles, axis = 1)
        
    # Across channels, per EDF file
    elif (argScalingMode == 2):
        arrDataMin = np.repeat(np.min(argDataSetMin, axis = 0)[np.newaxis, :], intNumChannels, axis = 0)
        arrDataMax = np.repeat(np.max(argDataSetMax, axis = 0)[np.newaxis, :], intNumChannels, axis = 0)
        
    # Across channels and EDF files
    elif (argScalingMode == 3):
        arrDataMin = np.full((intNumChannels, intNumEDFFiles), np.min(argDataSetMin))
        arrDataMax = np.full((intNumChannels, intNumEDFFiles), np.max(argDataSetMax))
        
    # Per channel and EDF file
    else:
        arrDataMin = argDataSetMin
        arrDataMax = argDataSetMax
    
    print('  arrDataMin.shape = {}, arrDataMax.shape = {}'.format(arrDataMin.shape, arrDataMax.shape))
    
    if (argDebug):
        print('  arrDataMin = {}'.format(pp.pformat(arrDataMin)))
        print('  arrDataMax = {}'.format(pp.pformat(arrDataMax)))
    
    return arrDataMin, arrDataMax


# In[ ]:


import wfdb

# Read the binary annotation file from the CHB-MIT data set and return
# a list of (start time, end time) tuples. The PhysioNet (which hosts
# the CHB-MIT data set) Python library, wfdb, is required

# The expected argFullFilename is the filename of the associated .edf
# file, not the filename of the annotation!

def fnReadCHBMITAnno(argFullFilename, argAnnoSuffix, argDebug = False):
    # Check to see if there is an annotation file. If so, read the
    # file and determine the seizure start and end time(s)
    strFullFilename_Anno = argFullFilename + '.' + argAnnoSuffix
    
    blnAnnoFound = False
    lstStartEndTimePts = []
    
    if (os.path.exists(strFullFilename_Anno)):
        if (argDebug): print('  Annotation file found: {}'.format(strFullFilename_Anno))
        
        # Call the rdann() method to read the annotation file
        objAnno = wfdb.rdann(argFullFilename, argAnnoSuffix)
        
        # Based on the number of pairs of markers returned (in case
        # of multiple seizures), return multiple tuples in a list
        intNumMarkers = objAnno.ann_len
        arrStartEndSymbols = objAnno.symbol
        arrStartEndTimePts = objAnno.sample
        
        # Raise an error if there is an odd number of markers
        if ((intNumMarkers % 2) != 0):
            raise Exception('Annotation markers are not all in pairs!')
        
        if (argDebug):
            print('  intNumMarkers = {}, arrStartEndSymbols = {}, arrStartEndTimePts = {}'.format(intNumMarkers, arrStartEndSymbols, arrStartEndTimePts))
        
        blnAnnoFound = True
        
        # Pair up the seizure markers and start/end times based on
        # whether they are odd or even items in the arrays, and return
        # the pairs as tuples in a list
        lstStartEndSymbols = list(zip(arrStartEndSymbols[0::2], arrStartEndSymbols[1::2]))
        lstStartEndTimePts = list(zip(arrStartEndTimePts[0::2], arrStartEndTimePts[1::2]))
        
        if (argDebug):
            print('  lstStartEndSymbols = {}'.format(lstStartEndSymbols))
            print('  lstStartEndTimePts = {}'.format(lstStartEndTimePts))
        
        if (argDebug): wfdb.plot_wfdb(annotation = objAnno, time_units = 'minutes')
        
    return blnAnnoFound, lstStartEndTimePts


# In[ ]:


# Read the text annotation file from the CHB-MIT data set and return
# a list of (start time, end time) tuples. The text annoation file is
# generated by scrCHB-MITScripts.py

# The expected argFullFilename is the filename of the associated .edf
# file, not the filename of the annotation!

def fnReadCHBMITAnnoTxt(argFullFilename, argAnnoSuffix, argSamplingFreq, argInfo = True, argDebug = False):
    # Check to see if there is an annotation file. If so, read the
    # file and determine the seizure start and end time(s)
    strFullFilename_Anno = argFullFilename + '.' + argAnnoSuffix
    
    blnAnnoFound = False
    lstStartEndTimePts = []
    
    if (os.path.exists(strFullFilename_Anno)):
        if (argInfo): print('  Annotation file found: {}'.format(strFullFilename_Anno))
        blnAnnoFound = True
        
        with open(strFullFilename_Anno) as objAnnoFile:
            objAnnoReader = csv.reader(objAnnoFile, delimiter = ',')
            intRow = 0

            for lstRow in objAnnoReader:  # Read in the row from file
                # Skip the first row of column names
                if (intRow == 0):
                    if (argDebug): print('Skipping the first row. Column names are: [{}]'.format(', '.join(lstRow)))
                else:
                    if (argDebug): print('lstRow = [{}] (len(lstRow) = {}) (type(lstRow) = {})'.format(lstRow, len(lstRow), type(lstRow)))
                    
                    # Skip empty rows, which are lists with len() = 0
                    if (len(lstRow) > 0):
                        fltStartTimeSec    = float(lstRow[0])
                        fltSeizureDuration = float(lstRow[1])
                        strAnnoType        = lstRow[2]
                        
                        intStartTimePt = int(round(fltStartTimeSec * argSamplingFreq))
                        intEndTimePt   = int(round((fltStartTimeSec + fltSeizureDuration) * argSamplingFreq))
                        
                        lstStartEndTimePts.append((intStartTimePt, intEndTimePt))
                        
                intRow += 1
                
    return blnAnnoFound, lstStartEndTimePts


# In[ ]:


# An over-simplified seizure cluster detection method. Currently,
# it only determines whether there are multiple seizures within
# the same segment
def fnClusterDetection(argStartEndTimePts):
    intNumSeizures = len(argStartEndTimePts)
    
    if (intNumSeizures > 1):
        blnSeizureCluster = True
    else:
        blnSeizureCluster = False
        
    return blnSeizureCluster, intNumSeizures


# In[ ]:


# Breaks a CHB-MIT segment (from a single EDF file) into multiple
# segments based on the change in EEG state within the segment
# (segment label and type). Uses the list of seizure start and end
# time points from fnReadCHBMITAnno() as a starting point
def fnBreakCHBMITSegment(argData, argStartEndTimePts, argSamplingFreq, argPreictalDuration = 5, argDebug = False):
    
    lstSegLabels        = []
    lstSegTypes         = []
    lstStartEndTimePts  = []
    lstStartEndTimeSecs = []
    lstSegDurations     = []
    lstNumTimePts       = []
    lstDataSegs         = []
    
    # Perform seizure cluster detection within a single EEG segment
    blnSeizureCluster, intNumSeizures = fnClusterDetection(argStartEndTimePts)
    
    if (intNumSeizures > 0):
        if (blnSeizureCluster):
            print('  ***MULTIPLE SEIZURES DETECTED*** ', end = '')
        else:
            print('  ***SEIZURE DETECTED*** ', end = '')
            
        print('(Number of seizures = {})'.format(intNumSeizures))
        print()
        
    # Get the start and end time point for the entire segment
    intStartTimePt = 0
    intEndTimePt = argData.shape[1]
    print('  intStartTimePt = {}, intEndTimePt = {}'.format(intStartTimePt, intEndTimePt))

    if (argDebug): print('  argStartEndTimePts = {}'.format(argStartEndTimePts))

    lstSegTimePts = []
    intNumSeizures = 0
    
    # If there are seizures within the segment, proceed to break it up
    if (len(argStartEndTimePts) > 0):
        # Loop through each seizure start/end time tuple and flatten
        # all the time points into a single list
        for tupStartEndTimePts in argStartEndTimePts:
            intNumSeizures = intNumSeizures + 1

            lstSegTimePts.extend([*tupStartEndTimePts])
            #lstSegTimePts[-1] = lstSegTimePts[-1] + 1
            
            # If this is not the first seizure in the segment, insert
            # an interictal segment in front of this seizure first
            
            # TODO: Depending on when the next seizure is, we may need
            #       to label this as post-ictal or interictal-cluster
            if (intNumSeizures > 1):
                tupSegState = dctSegStates['interictal']
                lstSegLabels.append(tupSegState[0])
                lstSegTypes.append(tupSegState[1])
                
            # Label this segment as ictal
            
            # TODO: Depending on whether this is a lead seizure of not,
            #       we may need to label this as ictal or ictal-cluster
            tupSegState = dctSegStates['ictal']
            lstSegLabels.append(tupSegState[0])
            lstSegTypes.append(tupSegState[1])

        if (argDebug): print('  lstSegTimePts = {}'.format(lstSegTimePts))
        
        # If the first time point of lstSegTimePts[] is not the same as
        # the first time point of the entire segment, insert the very
        # first time point to the beginning of the list
        if (lstSegTimePts[0] != intStartTimePt):
            lstSegTimePts.insert(0, intStartTimePt)
            
            # Label this newly inserted segment as interictal
            
            # TODO: Depending on when the previous and next seizures are,
            #       we may need to label this as post-ictal or interictal-
            #       cluster
            tupSegState = dctSegStates['interictal']
            lstSegLabels.insert(0, tupSegState[0])
            lstSegTypes.insert(0, tupSegState[1])
            
        # If the last time point of lstSegTimePts[] is not the same as
        # the last time point of the entire segment, append the very
        # last time pount to the end of the list
        if (lstSegTimePts[-1] != intEndTimePt):
            lstSegTimePts.append(intEndTimePt)
            
            # TODO: Depending on when the next seizure this, we may need
            #       to label this as post-ictal or interictal-cluster
            tupSegState = dctSegStates['interictal']
            lstSegLabels.append(tupSegState[0])
            lstSegTypes.append(tupSegState[1])
            
    # Otherwise, simply label the segment as interictal
    else:
        lstSegTimePts.extend([intStartTimePt, intEndTimePt])
        
        tupSegState = dctSegStates['interictal']
        lstSegLabels.append(tupSegState[0])
        lstSegTypes.append(tupSegState[1])
        
    if (argDebug):
        print('  lstSegTimePts = {}'.format(lstSegTimePts))
        print('  lstSegLabels = {}'.format(lstSegLabels))
        print('  lstSegTypes = {}'.format(lstSegTypes))
        print()
    
    # Break and annotate the interictal segment immediately preceding an
    # ictal segment into a preictal segment of length argPreictalDuration
    # (in secs)
    
    # TODO: Go through each ictal segment, and if there is a preceding
    #       interictal segment, label the last portion of that segment
    #       (as specified by argPreictalDuration) as preictal
    
    # TODO: Depending on whether we only want to consider lead seizures,
    #       the logic of this code may need to be modified
    
    # TODO: Maybe we can add an argument to generate an annotation file
    #       for preictal states
    
    lstSegTimePtsPre = []
    lstSegLabelsPre  = []
    lstSegTypesPre   = []
    
    for intSegIdx in range(len(lstSegTimePts) - 1):
        strSegLabel = lstSegLabels[intSegIdx]
        intSegType  = lstSegTypes[intSegIdx]
        
        intStartTimePt = lstSegTimePts[intSegIdx]
        intEndTimePt = lstSegTimePts[intSegIdx + 1]
        
        if (argDebug): print('  intStartTimePt = {}, intEndTimePt = {}'.format(intStartTimePt, intEndTimePt))
        
        if (fnIsSegState('ictal', intSegType) and intSegIdx > 0):
            strSegLabelPrev = lstSegLabels[intSegIdx - 1]
            intSegTypePrev  = lstSegTypes[intSegIdx - 1]
            
            intStartTimePtPrev = lstSegTimePts[intSegIdx - 1]
            intEndTimePtPrev = lstSegTimePts[intSegIdx]
            
            if (argDebug): print('  intStartTimePtPrev = {}, intEndTimePtPrev = {}'.format(intStartTimePtPrev, intEndTimePtPrev))
            
            if (fnIsSegState('interictal', intSegTypePrev)):
                tupPreictal = dctSegStates['preictal']
                
                intNumPreictalTimePts = argPreictalDuration * argSamplingFreq
                intStartTimePtPreictal = intEndTimePtPrev - intNumPreictalTimePts
                intEndTimePtPreictal = intEndTimePtPrev
                
                lstSegTimePtsPre.pop()
                lstSegTimePtsPre.extend([intStartTimePtPreictal, intEndTimePtPreictal])
                lstSegLabelsPre.append(tupPreictal[0])
                lstSegTypesPre.append(tupPreictal[1])
                
        lstSegTimePtsPre.extend([intStartTimePt, intEndTimePt])
        lstSegLabelsPre.append(strSegLabel)
        lstSegTypesPre.append(intSegType)
        
    if (argDebug):
        print('  lstSegTimePtsPre = {}'.format(lstSegTimePtsPre))
        print('  lstSegLabelsPre = {}'.format(lstSegLabelsPre))
        print('  lstSegTypesPre = {}'.format(lstSegTypesPre))        
        
    print()
    
    # Loop through lstSegTimePts[] in tandem with lstSegLabels[] and
    # lstSegTypes[] to generate the start and end time points of each
    # subsegment based on its EEG state, and break the segment data up
    # based on each subsegment's duration
    for intSegIdx in range(len(lstSegTimePts) - 1):
        strSegLabel = lstSegLabels[intSegIdx]
        intSegType  = lstSegTypes[intSegIdx]
        
        intStartTimePt = lstSegTimePts[intSegIdx]
        intEndTimePt = lstSegTimePts[intSegIdx + 1]
        
        fltStartTimeSec = intStartTimePt / argSamplingFreq
        fltEndTimeSec = intEndTimePt / argSamplingFreq

        intNumTimePts = intEndTimePt - intStartTimePt
        fltSegDuration = fltEndTimeSec - fltStartTimeSec
        
        lstStartEndTimePts.append((intStartTimePt, intEndTimePt))
        lstStartEndTimeSecs.append((fltStartTimeSec, fltEndTimeSec))
        lstSegDurations.append(fltSegDuration)
        lstNumTimePts.append(intNumTimePts)
        lstDataSegs.append(argData[:, intStartTimePt:intEndTimePt])
        
        if (fnIsSegState('ictal', intSegType)):
            print('  [***{}***:\t'.format(strSegLabel.upper()), end = '')
        else:
            #print('  [{}]:  '.format(strSegLabel), end = '')
            print('  [\t\t'.format(strSegLabel), end = '')
            
        #print('{:.6f}s ({}) -> {:.6f}s ({}), duration = {:.6f}s ({}), data shape = {}]'.format(fltStartTimeSec, intStartTimePt, fltEndTimeSec, intEndTimePt, fltSegDuration, intNumTimePts, lstDataSegs[-1].shape))
        print('{}s ({}) -> {}s ({}), duration = {}s ({}), data shape = {}]'.format(fltStartTimeSec, intStartTimePt, fltEndTimeSec, intEndTimePt, fltSegDuration, intNumTimePts, lstDataSegs[-1].shape))
        
    print()
    
    return lstSegLabels, lstSegTypes, lstStartEndTimePts, lstStartEndTimeSecs, lstSegDurations, lstNumTimePts, lstDataSegs


# In[ ]:


# Read in one or more EEG segments from the CHB-MIT data set from
# files (.edf) based on which files are specified in argCSVPath

# The units of argResamplingFreq is in Hz and argSubSeqDuration is in
# seconds

# NOTE: Subjects under 3 yrs old, i.e. chb06 (1.5 yrs old) and chb12 (2 yrs old),
# should be excluded, according to Jeffrey, since the base frequency of their EEGs
# have yet reached the adult level of 8.5Hz

def fnReadCHBMITEDFFiles_SlidingWindow(argCSVPath, argTestCSVPath = '', argResamplingFreq = -1, argSubSeqDuration = -1, argScalingParams = (), argScalingInfo = (), argStepSizeTimePts = -1, argStepSizeStates = {}, argSubWindowFraction = -1, argAnnoSuffix = 'seizures', argDebug = False, argTestMode = False):
    lstMatchingFiles = fnReadDataFileListCSV(argCSVPath)
    
    if (argTestCSVPath):
        lstMatchingTestFiles = fnReadDataFileListCSV(argTestCSVPath)
    else:
        lstMatchingTestFiles = []
    
    intNumMatchingFiles = len(lstMatchingFiles)  # Number of matching training files
    intNumMatchingTestFiles = len(lstMatchingTestFiles)  # Number of matching test files
    
    if (argDebug):
        print('intNumMatchingFiles = {}, intNumMatchingTestFiles = {}'.format(intNumMatchingFiles, intNumMatchingTestFiles))
        print()
    
    # Check whether there are mismatched channels in all EDF files
    fnMatchEDFChannels(lstMatchingFiles)
    
    # Read the first .edf file to extract parameters that are global
    # across the data set
    
    # NOTE: For EDF files, it is possible that for each segment, each
    #       channel can have different:
    #         (1) number of time points/segment lengths, and
    #         (2) sampling frequencies
    #
    #       And at a patient level, it is also possible that different
    #       segments have different:
    #         (4) number of channels
    #         (5) number of time points
    #         (6) sampling frequencies
    #
    #       For the CHB-MIT data set, only (5) is true, so we will need
    #       to handle different segment lengths as we read in each EDF
    #       file, and assume that the other parameters are global. This
    #       will simplify the code somewhat without having to handle all
    #       these parameters as variables for each EDF file
    #
    #       ***TODO***:
    #       HOWEVER, at a data set level, the number of channels across
    #       different patients can be different. If we want to train a
    #       generalized model, we will have to find the minimum montage
    #       that is common across a set of patients. We currently do not
    #       have the code to handle this situation
    
    strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts = fnReadEDFUsingPyEDFLib(lstMatchingFiles[0], argDebug = True)
    
    # If argResamplingFreq = -1 (default) or argResamplingFreq > intSamplingFreq,
    # use the *first* sampling frequency (upsampling is not allowed for now)
    if ((argResamplingFreq == -1) or (argResamplingFreq > fltSamplingFreq)):
        argResamplingFreq = fltSamplingFreq        
    print('argResamplingFreq = {}Hz'.format(argResamplingFreq))
    
    # If argSubSeqDuration = -1 (default) or argSubSeqDuration > intSegDuration,
    # default back to the *first* segment duration
    if ((argSubSeqDuration == -1) or (argSubSeqDuration > fltSegDuration)):
        argSubSeqDuration = fltSegDuration  # Do not break segment into subsequences
    print('argSubSeqDuration = {}s'.format(argSubSeqDuration))
    
    # Calculate the number of time points in each segment after splitting up the segment
    # into shorter sequences
    intSubSeqTimePts = int(round(argSubSeqDuration * fltSamplingFreq))
    intResampledSubSeqTimePts = int(round(argSubSeqDuration * argResamplingFreq))
    print('intSubSeqTimePts = {}, intResampledSubSeqTimePts = {}'.format(intSubSeqTimePts, intResampledSubSeqTimePts))
        
    # If argStepSizeTimePts = -1 (default) or argStepSizeTimePts > intSubSeqTimePts,
    # default back to the *first* segment time points
    if ((argStepSizeTimePts == -1) or (argStepSizeTimePts > intSubSeqTimePts)):
        argStepSizeTimePts = intSubSeqTimePts  # No sliding window
    print('argStepSizeTimePts = {}'.format(argStepSizeTimePts))
    
    # If argSubWindowFraction = -1 (default) or argSubWindowFraction <= 0 or > 1,
    # default to the entire window size (intSubSeqTimePts)
    if ((argSubWindowFraction == -1) or (argSubWindowFraction <= 0) or (argSubWindowFraction > 1)):
        argSubWindowFraction = 1.0
    print('argSubWindowFraction = {}'.format(argSubWindowFraction))
        
    print()
    
    # Initialize data structures to store data for the entire data set
    lstAllBaseFilenames = []  # List of segment filenames
    lstAllSegLabels = []      # List of segment labels
    lstAllSegTypes = []       # List of segment types (preictal = 1 or interictal = 2)
    
    # NOTE: arrAllData[] can no longer be initialized prior to reading all the
    #       segments, as each segment are now allowed to have a different length
    #       (intNumTimePts), so each segment may be split into a different number
    #       of subsequences (intNumFullSubSeqs). Therefore, we can no longer
    #       calculate the total number of subsequences (intTotalBatchSize) based
    #       on the number of sequences (intNumMatchingFiles) and the number of
    #       subsequences per file (intNumSubSeqs)
    #
    #       Instead, we will create arrAllData[] with a fixed number of channels
    #       and subsequence time points (both of which are expected to be
    #       consistent across the data set), and a 3rd dimension with zero size
    #       that we will append the subsequences to as we loop through each file
    #
    #       intTotalBatchSize = intNumMatchingFiles * intNumSubSeqs  # Total number of sequences
    
    # ***TEST MODE:
    if (argTestMode):
        intSubSeqTimePts = 8  # Window size
        argSubWindowFraction = 0.5

    arrAllData = np.zeros((intNumChannels, intSubSeqTimePts, 0), dtype = np.float64)
    if (argDebug):
        print('arrAllData.shape = {}'.format(arrAllData.shape))
        print()
    
    lstAllSegDurations  = []  # List of segment durations (in seconds)
    lstAllSamplingFreqs = []  # List of sampling frequencies (in Hz)
    lstAllChannels      = []  # List of lists (of channel names)
    lstAllSequences     = []  # List of sequence indices
    lstAllSubSequences  = []  # List of subsequence indices (if one sequence is broken up into subsequences)
    
    # List of start/end time points and times (in seconds) for each subsequence
    arrAllStartEndTimePts  = np.zeros((0, 2))
    arrAllStartEndTimesSec = np.zeros((0, 2), dtype = np.float64)
    
    lstSeizureDurations = []  # List of seizure durations (in seconds) (not broken into subsequences)
    
    datLoopStart = utils.fnNow()
    print('Loop started on {}'.format(utils.fnGetDatetime(datLoopStart)))
    print()
    
    intNumProcessedFiles = 0
    fltFirstSamplingFreq = fltSamplingFreq  # Remember the sampling frequency of the first file
    lstFirstChannels = lstChannels          # Remember the list of channels of the first file
    
    print('fltFirstSamplingFreq = {}Hz'.format(fltFirstSamplingFreq))
    print('lstFirstChannels = {}'.format(lstFirstChannels))
    print()
    
    # SCALING
    if (argScalingParams):
        intScalingMode  = argScalingParams[0]
        tupScaledMinMax = argScalingParams[1]
        print('Scaling data using: intScalingMode = {}, tupScaledMinMax = {}'.format(intScalingMode, tupScaledMinMax))
        
        if (argScalingInfo):
            print('  Scaling using argScalingInfo({}, {}, {})'.format(len(argScalingInfo[0]), argScalingInfo[1].shape, argScalingInfo[2].shape))
            
            arrDataMin = argScalingInfo[1]
            arrDataMax = argScalingInfo[2]
            
        else:
            # TODO: Currently we are reading in the same data set twice, so this is not speed
            #       efficient. Maybe we should combine these two loops into one    
            # SCALING: Collect the statistics of the specified EDF files
            arrDataSetMin, arrDataSetMax, arrDataSetMean = fnGetCHBMITStats(lstMatchingFiles, lstMatchingTestFiles, argDebug = False)

            # Construct arrDataMin and arrDataMax based on the scaling mode
            arrDataMin, arrDataMax = fnGenMinMaxArrays(intScalingMode, arrDataSetMin, arrDataSetMax, argDebug = False)
            
    else:
        print('Data scaling is not performed')
        
    print()
        
    # ***TEST MODE:
    if (argTestMode):
        lstMatchingFiles = lstMatchingFiles[0:1]
    
    # Loop through each file in the target directory
    for intFullFilenameIdx, strFullFilename in enumerate(lstMatchingFiles):  # SCALING
        # Process only .edf files
        if strFullFilename.endswith('.edf'):
            print('Processing {}...'.format(strFullFilename))
            
            # Extract the filename from the full path
            strPath, strFilename = os.path.split(strFullFilename)
            
            # Read the .edf file
            strSegLabel, arrDataEDF, fltSegDurationEDF, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePtsEDF = fnReadEDFUsingPyEDFLib(strFullFilename, argDebug = False)
            print('  fltSegDurationEDF = {}s, fltSampingFreq = {}Hz'.format(fltSegDurationEDF, fltSamplingFreq))
            print()
            
            if (argScalingParams):
                # SCALING: Construct arrDataMinMax (C x 2) from arrDataMin and arrDataMax
                #          (C x N)
                # SCALING (normalize, filter, smoothing(?), augment (later))
                arrDataMinMax = np.concatenate((arrDataMin[:, intFullFilenameIdx][:, np.newaxis], arrDataMax[:, intFullFilenameIdx][:, np.newaxis]), axis = 1)
                if (argDebug):
                    print('  arrDataMinMax.shape = {}'.format(arrDataMinMax.shape))
                    print()
                
                arrDataEDFScaled = utils.fnMinMaxScaler(arrDataEDF, tupScaledMinMax, arrDataMinMax, argDebug = False)
            else:
                arrDataEDFScaled = arrDataEDF
                
            # Consistency check for sampling rate and channel labels
            if (fltSamplingFreq != fltFirstSamplingFreq):
                raise Exception('Sampling frequency not consistent across the data segments!')
                
            if (lstChannels != lstFirstChannels):
                raise Exception('Channels not consistent across the data segments!')
            
            # Check to see if an annotation file (.seizures) exists. If so, set strSegLabel
            # and determine the seizure start/end times
            
            # TODO: Change argAnnoSuffix to argReadAnnoTxt (boolean) to call fnReadCHBMITAnno()
            #       to read binary 'seizures' files when false, and call fnReadCHBMITAnnoTxt()
            #       to read ASCII 'annotation.txt' files when true
            #blnAnnoFound, lstStartEndTimePts = fnReadCHBMITAnno(strFullFilename, argAnnoSuffix)
            blnAnnoFound, lstStartEndTimePts = fnReadCHBMITAnnoTxt(strFullFilename, argAnnoSuffix, fltSamplingFreq)
            
            # SCALING
            # Break the segment up based on their respective segment types
            lstSegLabels, lstSegTypes, lstStartEndTimePts, lstStartEndTimeSecs, lstSegDurations, lstNumTimePts, lstDataSegs = fnBreakCHBMITSegment(arrDataEDFScaled, lstStartEndTimePts, fltSamplingFreq, argDebug = True)
            
            if (argDebug):
                print('   lstSegLabels = {}'.format(lstSegLabels))
                print('   lstSegTypes = {}'.format(lstSegTypes))
                print('   lstStartEndTimePts = {}'.format(lstStartEndTimePts))
                print('   lstStartEndTimeSecs = {}'.format(lstStartEndTimeSecs))
                print('   lstSegDurations = {}'.format(lstSegDurations))
                print('   lstNumTimePts = {}'.format(lstNumTimePts))
                print('   len(lstDataSegs) = {}'.format(len(lstDataSegs)))  # TODO: No longer used due to sliding window
                print()
            
            # ***TEST MODE:
            if (argTestMode):
                argStepSizeTimePts = 6
                #argStepSizeStates = {}
                argStepSizeStates = {'ictal': 2}
                
                #lstSegLabels = ['interictal', 'ictal', 'interictal']
                #lstStartEndTimePts = [(0, 30), (30, 70), (70, 100)]

                lstSegLabels = ['interictal', 'ictal', 'interictal', 'ictal', 'interictal']
                lstStartEndTimePts = [(0, 30), (30, 50), (50, 60), (60, 80), (80, 100)]
                
                intNumTimePtsEDF = lstStartEndTimePts[-1][1] - lstStartEndTimePts[0][0]
                
                # Create a data array filled with zeros except for the first 2 channels,
                # for debugging purposes
                arrDataEDFScaled = np.zeros((intNumChannels, intNumTimePtsEDF))
                arrDataEDFScaled[0, :] = np.arange(intNumTimePtsEDF)
                arrDataEDFScaled[1, :] = np.arange(intNumTimePtsEDF)
                
                lstSegTypes = []
                for strSegLabel in lstSegLabels:
                    lstSegTypes.append(fnGetSegType(strSegLabel))
                    
                lstStartEndTimeSecs = []
                for tupStartEndTimePts in lstStartEndTimePts:
                    lstStartEndTimeSecs.append(tuple((intTimePt / fltSamplingFreq) for intTimePt in tupStartEndTimePts))
                
                lstSegDurations = []
                for fltStartTimeSec, fltEndTimeSec in lstStartEndTimeSecs:
                    lstSegDurations.append(fltEndTimeSec - fltStartTimeSec)
                
                lstNumTimePts = []
                for intTupIdx, tupStartEndTimePts in enumerate(lstStartEndTimePts):
                    lstNumTimePts.append(lstStartEndTimePts[intTupIdx][1] - lstStartEndTimePts[intTupIdx][0])
                    
                lstDataSegs = []
                for intNumTimePts in lstNumTimePts:
                    lstDataSegs.append(np.zeros((arrDataEDFScaled.shape[0], intNumTimePts)))
                
                print('   *****************')
                print('   *** TEST MODE ***')
                print('   *****************')
                
                if (len(lstSegTypes) != len(lstStartEndTimePts)):
                    print('\n   WARNING: len(lstSegTypes) != len(lstStartEndTimePts)!\n')
                
                print('   intSubSeqTimePts = {}, argStepSizeTimePts = {}, argStepSizeStates = {}, argSubWindowFraction = {}, intNumTimePtsEDF = {}'.format(
                    intSubSeqTimePts, argStepSizeTimePts, argStepSizeStates, argSubWindowFraction, intNumTimePtsEDF))
                print()
                
                print('   lstSegLabels = {}, lstSegTypes = {}'.format(lstSegLabels, lstSegTypes))
                print('   lstStartEndTimePts = {}, lstNumTimePts = {}'.format(lstStartEndTimePts, lstNumTimePts))
                print('   lstStartEndTimeSecs = {}'.format(lstStartEndTimeSecs))
                print('   lstSegDurations = {}'.format(lstSegDurations))
                print('   len(lstDataSegs) = {}'.format(len(lstDataSegs)))  # TODO: No longer used due to sliding window
                print()
                
            # Create an array with same size as the entire segment with each time
            # point filled with intSegType
            arrSegTypeTimePts = np.zeros((intNumTimePtsEDF), dtype = np.int)
            
            for intSegIdx, tupSeg in enumerate(zip(lstSegTypes, lstStartEndTimePts)):
                intSegType, tupStartEndTimePt = [*tupSeg]
                
                intStartTimePt, intEndTimePt = tupStartEndTimePt
                print('   intStartTimePt = {} -> intEndTimePt = {}: intSegType = {}'.format(intStartTimePt, intEndTimePt, intSegType))
                
                arrSegTypeTimePts[intStartTimePt:intEndTimePt] = intSegType
                
            print()
            print('   arrSegTypeTimePts = {} (shape = {}'.format(arrSegTypeTimePts, arrSegTypeTimePts.shape))
            print()
            
            print('  argStepSizeStates = {}'.format(argStepSizeStates))
            
            # Create a list of intStepSizeTimePts that correspond to intSegType of each segment
            lstStepSizeTimePts = []
            for intSegLabel in lstSegLabels:
                if intSegLabel in argStepSizeStates.keys():
                    lstStepSizeTimePts.append(argStepSizeStates[intSegLabel])
                else:
                    lstStepSizeTimePts.append(argStepSizeTimePts)
                    
            print('  lstStepSizeTimePts = {}'.format(lstStepSizeTimePts))
            print()
            
            # Loop through each segment to generate the appropriate subsequences 
            for intSegIdx, tupSeg in enumerate(zip(lstSegLabels, lstSegTypes, lstStartEndTimePts, lstStartEndTimeSecs, lstSegDurations, lstNumTimePts, lstDataSegs)):
                strSegLabel, intSegType, tupStartEndTimePt, tupStartEndTimeSec, fltSegDuration, intNumTimePts, arrDataSeg = [*tupSeg]
                
                # Extract the start and end time points of each segment
                intStartTimePt, intEndTimePt   = tupStartEndTimePt
                fltStartTimeSec, fltEndTimeSec = tupStartEndTimeSec
                
                # Get the intStepSizeTimePts for the current segment
                intStepSizeTimePts = lstStepSizeTimePts[intSegIdx]
                
                # Synchronize the start of the sliding window to the beginning of each segment
                intWindowStartTimePt = intStartTimePt  # Start time point of the first window
                intWindowEndTimePt = intWindowStartTimePt + intSubSeqTimePts  # End time point of the first window
            
                if (fnIsSegState('ictal', intSegType)):
                    lstSeizureDurations.append((strFilename, fltSegDuration))
                    print('  => SEGMENT ({}) = ***{}*** (intStepSizeTimePts = {})'.format(intSegIdx + 1, strSegLabel.upper(), intStepSizeTimePts))
                else:
                    print('  => SEGMENT ({}) = {} (intStepSizeTimePts = {})'.format(intSegIdx + 1, strSegLabel.upper(), intStepSizeTimePts))
                    
                print('    {:.6f}s ({}) -> {:.6f}s ({}), duration = {:.6f}s ({}), arrDataSeg.shape = {}'.format(fltStartTimeSec, intStartTimePt, fltEndTimeSec, intEndTimePt, fltSegDuration, intNumTimePts, arrDataSeg.shape))
                print()
                
                # Analyze the number of subsequences to break down from the main segment, and
                # whether the time points can be equally divided among all the subsequences
                #
                # Calculations specific for sliding window, inspired by:
                #
                #   http://cs231n.github.io/convolutional-networks/
                #
                # on how to calculate padding for CNNs. In this case, we assume zero-padding
                # by setting P = 0:
                #
                #   ((W − F + 2P)/S) + 1
                #
                # where P = padding size, W = input width of layer, F = output width of layer,
                # and S = stride
                #
                #fltNumSubSeqs = ((intNumTimePts - intSubSeqTimePts) / intStepSizeTimePts) + 1
                
                # If we're not in the last segment there is no need to calculate using the
                # sliding window formula specified above since the window can slide past the
                # current segment and into the next time
                if (intSegIdx <  len(lstStartEndTimePts) - 1):
                    intNumSubSeqs = math.ceil(intNumTimePts / intStepSizeTimePts)  # Total number of subsequences required
                    intNumFullSubSeqs = intNumSubSeqs                              # Number of completely filled subsequences
                    intNumOrphanTimePts = 0                                        # Number of orphan time points
                else:
                    intNumSubSeqs = math.ceil((intNumTimePts - intSubSeqTimePts) / intStepSizeTimePts) + 1  # Total number of subsequences required
                    intNumFullSubSeqs = ((intNumTimePts - intSubSeqTimePts) // intStepSizeTimePts) + 1      # Number of completely filled subsequences
                    intNumOrphanTimePts = (intNumTimePts - intSubSeqTimePts) % intStepSizeTimePts           # Number of orphan time points

                    #intPadding = ((intNumSubSeqs - 1) * intStepSizeTimePts) - intNumTimePts + intSubSeqTimePts
                    #if (argDebug): print('intPadding = {}'.format(intPadding))
                    
                # If the number of subsequence time points specified results in the
                # last subsequence not being completely filled up, give a warning message
                if (intNumOrphanTimePts > 0):
                    print('     WARNING: Time points cannot be divided into complete subsequences')

                if (intNumFullSubSeqs > 1):
                    print('     Splitting each segment into {} subsequences based on intSubSeqTimePts = {} and intStepSizeTimePts = {}'.format(intNumFullSubSeqs, intSubSeqTimePts, intStepSizeTimePts))
                    
                strLastSegment = ' (last segment)' if (intSegIdx ==  len(lstStartEndTimePts) - 1) else ''
                print('     [intSegIdx = {}{}: intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts = {} / {} / {} ({:.2f}%)]'.format(intSegIdx, strLastSegment, intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts, intNumOrphanTimePts/intSubSeqTimePts))
                
                print()
                
                arrDataSplit = np.zeros((intNumChannels, intSubSeqTimePts, intNumFullSubSeqs), dtype = np.float64)
                arrStartEndTimePtsSplit = np.zeros((intNumFullSubSeqs, 2))
                
                print('     arrDataSplit.shape = {} ({})'.format(arrDataSplit.shape, type(arrDataSplit[0][0][0])))
                
                if (argTestMode):
                    print('     arrDataEDFScaled[0, intStartTimePt:intEndTimePt] = {}'.format(arrDataEDFScaled[0, intStartTimePt:intEndTimePt]))
                
                lstSegLabels_SlidingWindow = []
                lstSegTypes_SlidingWindow  = []
                
                for intSubSeq in np.arange(intNumFullSubSeqs):
                    # Throw an exception if the start of the sliding window extends into the
                    # next segment, or the end of the window extends beyond the end point of
                    # the entire segment
                    if (intWindowStartTimePt > intEndTimePt):
                        raise Exception('intWindowStartTimePt ({}) > intEndTimePt ({})'.format(intWindowStartTimePt, intEndTimePt))
                    if (intWindowEndTimePt > intNumTimePtsEDF):
                        raise Exception('intWindowEndTimePt ({}) > intNumTimePts ({})'.format(intWindowEndTimePt, intNumTimePtsEDF))
                    
                    # Extract data within the sliding window from arrDataEDFScaled[]
                    arrDataWindow = arrDataEDFScaled[:, intWindowStartTimePt:intWindowEndTimePt]
                    
                    # Fill arrDataSplit[] with data in the sliding window
                    arrDataSplit[:, :, intSubSeq] = arrDataWindow
                    
                    # Record the start and end time points for this subsequence (sliding window)
                    arrStartEndTimePtsSplit[intSubSeq, :] = [intWindowStartTimePt, intWindowEndTimePt]
                    
                    # Create an array with same size as the sliding window with each time
                    # point filled with intSegType
                    arrSegTypeWindow = arrSegTypeTimePts[intWindowStartTimePt:intWindowEndTimePt]
                    
                    # Determine whether the sliding window is within the current segment,
                    # or whether it has extended into the next segment
                    
                    # Sliding window has extended into the next segment
                    if (intWindowEndTimePt > intEndTimePt):
                        strExtendedWindow = '*'
                        
                        # Make sure that the number of time points in the subwindow is at least 1
                        intSubWindowTimePts = math.ceil(argSubWindowFraction * intSubSeqTimePts)
                        
                        # Find the nearest odd number if the number of time points is even (to
                        # ensure that stat.mode() will not raise an exception)
                        intSubWindowOddTimePts = ((intSubWindowTimePts // 2) * 2) + 1
                        
                        # If the nearest odd number is larger than intSubSeqTimePts, trim it down
                        # to intSubSeqTimePts
                        if (intSubWindowOddTimePts > intSubSeqTimePts):
                            intSubWindowOddTimePts = intSubSeqTimePts
                            
                            # If intSubSeqTimePts happens to be an even number, subtract 1 to
                            # make intSubWindowOddTimePts an odd number
                            if (utils.fnIsEven(intSubWindowOddTimePts)):
                                intSubWindowOddTimePts = intSubWindowOddTimePts - 1
                        
                        # Create a subwindow within the sliding window (at the MSB end) with
                        # each time point filled with intSegType
                        arrSegTypeSubWindow = arrSegTypeWindow[-intSubWindowOddTimePts:]
                        
                        # Get the mode (most common) intSegType within the subwindow
                        try:
                            intSegTypeMode = stat.mode(arrSegTypeSubWindow)
                        except:
                            intSegTypeMode = -1
                        
                        # Assign the segment label and type for the sliding window using the
                        # most common label and type in the subwindow
                        lstSegLabels_SlidingWindow.append(fnGetSegLabel(intSegTypeMode))
                        lstSegTypes_SlidingWindow.append(intSegTypeMode)
                        
                    # Sliding window is entirely within the current segment
                    else:
                        strExtendedWindow = ' '
                        
                        # Simply use the current segment label and type for this window
                        lstSegLabels_SlidingWindow.append(strSegLabel)
                        lstSegTypes_SlidingWindow.append(intSegType)
                        
                    # ***TEST MODE:
                    if (argTestMode):
                        print('       {}intSubSeq = {}: sliding window = {} -> {}, intEndTimePt = {}, arrDataWindow = {}'.format(strExtendedWindow, intSubSeq, intWindowStartTimePt, intWindowEndTimePt, intEndTimePt, arrDataWindow[0, :]))
                        print('         arrSegTypeWindow = {}'.format(arrSegTypeWindow))
                        
                        if (intWindowEndTimePt > intEndTimePt):
                            print('         arrSegTypeSubWindow = {} (strSegLabel = {}, intSegType = {})'.format(arrSegTypeSubWindow, lstSegLabels_SlidingWindow[-1], lstSegTypes_SlidingWindow[-1]))
                            
                        print()
                        
                    # Move the sliding window forward one step
                    intWindowStartTimePt = intWindowStartTimePt + intStepSizeTimePts
                    intWindowEndTimePt = intWindowStartTimePt + intSubSeqTimePts
                    
                print('     arrStartEndTimePtsSplit.shape = {} ({})'.format(arrStartEndTimePtsSplit.shape, type(arrStartEndTimePtsSplit[0][0])))
                print('     arrStartEndTimePtsSplit = {} -> {}'.format(arrStartEndTimePtsSplit[0, :], arrStartEndTimePtsSplit[-1:, :]))
                
                arrStartEndTimesSecSplit = arrStartEndTimePtsSplit / fltSamplingFreq
                print('     arrStartEndTimesSecSplit.shape = {} ({})'.format(arrStartEndTimesSecSplit.shape, type(arrStartEndTimesSecSplit[0][0])))
                print('     arrStartEndTimesSecSplit = {} -> {}'.format(arrStartEndTimesSecSplit[0, :], arrStartEndTimesSecSplit[-1:, :]))
                
                print()
                
                # Loop through each subsequence and save the data and metadata into
                # the appropriate data structures

                # TODO: To improve performance, instead of looping through each
                #       subsequence one by one we may be able to concatenate the
                #       whole batch while replicating the other parameters and then
                #       append them to the end of each list (done)

                # This new method takes about 3 mins to execute for chb01
                lstAllBaseFilenames.extend([strFilename] * intNumFullSubSeqs)
                #lstAllSegLabels.extend([strSegLabel] * intNumFullSubSeqs)
                #lstAllSegTypes.extend([intSegType] * intNumFullSubSeqs)
                lstAllSegLabels.extend(lstSegLabels_SlidingWindow)
                lstAllSegTypes.extend(lstSegTypes_SlidingWindow)
                arrAllData = np.concatenate((arrAllData, arrDataSplit), axis = 2)
                lstAllSegDurations.extend([argSubSeqDuration] * intNumFullSubSeqs)   # Record the duration after the split
                lstAllSamplingFreqs.extend([argResamplingFreq] * intNumFullSubSeqs)  # Record the resampled frequency
                lstAllChannels.extend([lstChannels] * intNumFullSubSeqs)
                lstAllSequences.extend([intSequence] * intNumFullSubSeqs)
                lstAllSubSequences.extend(list(np.arange(intNumFullSubSeqs)))
                
                # Concatenate the start/end time points and times (in seconds) for
                # each subsequence to the appropriate output data structures
                arrAllStartEndTimePts = np.concatenate((arrAllStartEndTimePts, arrStartEndTimePtsSplit), axis = 0)
                arrAllStartEndTimesSec = np.concatenate((arrAllStartEndTimesSec, arrStartEndTimesSecSplit), axis = 0)
                print('     arrAllStartEndTimePts.shape = {}, arrAllStartEndTimesSec.shape = {}'.format(arrAllStartEndTimePts.shape, arrAllStartEndTimesSec.shape))
                print()
                
                if (argDebug):
                    print('     arrAllData.shape = {}'.format(arrAllData.shape))
                    print()
                    
            intNumProcessedFiles = intNumProcessedFiles + 1
            
    datLoopEnd = utils.fnNow()
    print('Loop ended on {}'.format(utils.fnGetDatetime(datLoopEnd)))

    datLoopDuration = datLoopEnd - datLoopStart
    print('datLoopDuration = {}'.format(datLoopDuration))
    
    print('intNumProcessedFiles = {}'.format(intNumProcessedFiles))
    
    # Make sure that the number of files that matched the specified
    # extension is the same number of files that we actually processed
    if (intNumMatchingFiles != intNumProcessedFiles):
        print('WARNING: intNumMatchingFiles != intNumProcessedFiles!')
        
    print()
    
    # Resample and round the results to the nearest integer since arrData[channels, time pts]
    # is of type int16
    if (argResamplingFreq != fltFirstSamplingFreq):
        print('Resampling data from {}Hz to {}Hz...'.format(fltFirstSamplingFreq, argResamplingFreq))
        print()
        
        #arrAllDataResampled = signal.resample(arrAllData, intResampledSubSeqTimePts, axis = 1)  # Shape = [channels, subseq time pts, subsequences]
        arrAllDataResampled = np.zeros((arrAllData.shape[0], intResampledSubSeqTimePts, arrAllData.shape[2]), dtype = np.float64)  # Shape = [channels, subseq time pts, subsequences]

        for intSubSeq in np.arange(arrAllData.shape[2]):
            arrAllDataResampled[:, :, intSubSeq] = signal.resample(arrAllData[:, :, intSubSeq], intResampledSubSeqTimePts, axis = 1)
            
    else:
        arrAllDataResampled = arrAllData
        
    if (argDebug): print('arrAllDataResampled.shape = {} ({})'.format(arrAllDataResampled.shape, type(arrAllDataResampled[0][0])))
    
    # Process data structures for scaling across training and test sets
    lstTestBaseFilenames = []
    
    for strTestFullFilename in lstMatchingTestFiles:
        # Extract the filename from the full path
        strPath, strTestFilename = os.path.split(strTestFullFilename)
        
        lstTestBaseFilenames.append(strTestFilename)
        
    if (lstTestBaseFilenames):
        arrTestDataMin = arrDataMin[:, intNumMatchingFiles:]
        arrTestDataMax = arrDataMax[:, intNumMatchingFiles:]
        
        tupScalingInfo = (lstTestBaseFilenames, arrTestDataMin, arrTestDataMax)
    else:
        tupScalingInfo = ()
    
    print('len(lstAllBaseFilenames) = {}'.format(len(lstAllBaseFilenames)))
    print('len(lstAllSegLabels) = {}'.format(len(lstAllSegLabels)))
    print('len(lstAllSegTypes) = {}'.format(len(lstAllSegTypes)))
    print('arrAllData.shape = {} ({}) (features x sequence length x batch size)'.format(arrAllData.shape,  type(arrAllData[0][0][0])))
    print('arrAllDataResampled.shape = {} ({}) (features x sequence length x batch size)'.format(arrAllDataResampled.shape,  type(arrAllDataResampled[0][0][0])))
    print('len(lstAllSegDurations) = {}'.format(len(lstAllSegDurations)))
    print('len(lstAllSamplingFreqs) = {}'.format(len(lstAllSamplingFreqs)))
    print('len(lstAllChannels) = {}'.format(len(lstAllChannels)))
    print('len(lstAllSequences) = {}'.format(len(lstAllSequences)))
    print('len(lstAllSubSequences) = {}'.format(len(lstAllSubSequences)))
    print('len(lstSeizureDurations) = {}'.format(len(lstSeizureDurations)))
    
    # Print the shape of the data structures that store the start/end
    # time points and times (in seconds) for each subsequence
    print('arrAllStartEndTimePts.shape = {} ({})'.format(arrAllStartEndTimePts.shape, type(arrAllStartEndTimePts[0][0])))
    print('arrAllStartEndTimesSec.shape = {} ({})'.format(arrAllStartEndTimesSec.shape, type(arrAllStartEndTimesSec[0][0])))
    
    if (lstTestBaseFilenames):
        # Print data structures for scaling across training and test sets
        print('len(lstTestBaseFilenames) = {}'.format(len(lstTestBaseFilenames)))
        print('arrTestDataMin.shape = {}, arrTestDataMax.shape = {}'.format(arrTestDataMin.shape, arrTestDataMax.shape))
        
    print()

    print('Size of arrAllData          = {:.2f}Gb'.format(arrAllData.nbytes / (1024**3)))
    print('Size of arrAllDataResampled = {:.2f}Gb'.format(arrAllDataResampled.nbytes / (1024**3)))
    
    if (0):
        print('lstAllBaseFilenames = {}'.format(lstAllBaseFilenames))
        print('lstAllSegLabels = {}'.format(lstAllSegLabels))
        print('lstAllSegTypes = {}'.format(lstAllSegTypes))
        print('arrAllData = {}'.format(arrAllData))
        print('arrAllDataResampled = {}'.format(arrAllDataResampled))
        print('lstAllSegDurations = {}'.format(lstAllSegDurations))
        print('lstAllSamplingFreqs = {}'.format(lstAllSamplingFreqs))
        print('lstAllChannels = {}'.format(lstAllChannels))
        print('lstAllSequences = {}'.format(lstAllSequences))
        print('lstAllSubSequences = {}'.format(lstAllSubSequences))
        print('lstSeizureDurations = {}'.format(lstSeizureDurations))
        print('arrAllStartEndTimePts = {}'.format(arrAllStartEndTimePts))
        print('arrAllStartEndTimesSec = {}'.format(arrAllStartEndTimesSec))
        
        print()
        
    return lstAllBaseFilenames, lstAllSegLabels, lstAllSegTypes, arrAllDataResampled, lstAllSegDurations, lstAllSamplingFreqs, lstAllChannels, lstAllSequences, lstAllSubSequences, lstSeizureDurations, arrAllStartEndTimesSec, tupScalingInfo


# In[ ]:


# Read in one or more EEG segments from the CHB-MIT data set from
# files (.edf) based on which files are specified in argCSVPath

# The units of argResamplingFreq is in Hz and argSubSeqDuration is in
# seconds

# NOTE: Subjects under 3 yrs old, i.e. chb06 (1.5 yrs old) and chb12 (2 yrs old),
# should be excluded, according to Jeffrey, since the base frequency of their EEGs
# have yet reached the adult level of 8.5Hz

def fnReadCHBMITEDFFiles(argCSVPath, argResamplingFreq = -1, argSubSeqDuration = -1, argScalingParams = (), argAnnoSuffix = 'seizures', argDebug = False):
    lstMatchingFiles = fnReadDataFileListCSV(argCSVPath)
    
    intNumMatchingFiles = len(lstMatchingFiles)  # Number of matching files
    if (argDebug):
        print('intNumMatchingFiles = {}'.format(intNumMatchingFiles))
        print()
    
    # Check whether there are mismatched channels in all EDF files
    fnMatchEDFChannels(lstMatchingFiles)
    
    # Read the first .edf file to extract parameters that are global
    # across the data set
    
    # NOTE: For EDF files, it is possible that for each segment, each
    #       channel can have different:
    #         (1) number of time points/segment lengths, and
    #         (2) sampling frequencies
    #
    #       And at a patient level, it is also possible that different
    #       segments have different:
    #         (4) number of channels
    #         (5) number of time points
    #         (6) sampling frequencies
    #
    #       For the CHB-MIT data set, only (5) is true, so we will need
    #       to handle different segment lengths as we read in each EDF
    #       file, and assume that the other parameters are global. This
    #       will simplify the code somewhat without having to handle all
    #       these parameters as variables for each EDF file
    #
    #       ***TODO***:
    #       HOWEVER, at a data set level, the number of channels across
    #       different patients can be different. If we want to train a
    #       generalized model, we will have to find the minimum montage
    #       that is common across a set of patients. We currently do not
    #       have the code to handle this situation
    
    strSegLabel, arrData, fltSegDuration, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePts = fnReadEDFUsingPyEDFLib(lstMatchingFiles[0], argDebug = True)
    
    # If argResamplingFreq = -1 (default) or argResamplingFreq > intSamplingFreq,
    # use the *first* sampling frequency (upsampling is not allowed for now)
    if ((argResamplingFreq == -1) or (argResamplingFreq > fltSamplingFreq)):
        argResamplingFreq = fltSamplingFreq        
    print('argResamplingFreq = {}Hz'.format(argResamplingFreq))
    
    # If argSubSeqDuration = -1 (default) or argSubSeqDuration > intSegDuration,
    # default back to the *first* segment duration
    if ((argSubSeqDuration == -1) or (argSubSeqDuration > fltSegDuration)):
        argSubSeqDuration = fltSegDuration  # Do not break segment into subsequences
    print('argSubSeqDuration = {}s'.format(argSubSeqDuration))
    
    print()
    
    # Calculate the number of time points in each segment after the resampling
    intResampledTimePts = int(round((argResamplingFreq / fltSamplingFreq) * intNumTimePts))
    print('intResampledTimePts = {}'.format(intResampledTimePts))

    # Calculate the number of time points in each segment after splitting up the segment
    # into shorter sequences
    intSubSeqTimePts = int(round((argSubSeqDuration / fltSegDuration) * intResampledTimePts))
    print('intSubSeqTimePts = {}'.format(intSubSeqTimePts))
        
    # Initialize data structures to store data for the entire data set
    lstAllBaseFilenames = []  # List of segment filenames
    lstAllSegLabels = []      # List of segment labels
    lstAllSegTypes = []       # List of segment types (preictal = 1 or interictal = 2)
    
    # NOTE: arrAllData[] can no longer be initialized prior to reading all the
    #       segments, as each segment are now allowed to have a different length
    #       (intNumTimePts), so each segment may be split into a different number
    #       of subsequences (intNumFullSubSeqs). Therefore, we can no longer
    #       calculate the total number of subsequences (intTotalBatchSize) based
    #       on the number of sequences (intNumMatchingFiles) and the number of
    #       subsequences per file (intNumSubSeqs)
    #
    #       Instead, we will create arrAllData[] with a fixed number of channels
    #       and subsequence time points (both of which are expected to be
    #       consistent across the data set), and a 3rd dimension with zero size
    #       that we will append the subsequences to as we loop through each file
    #
    #       intTotalBatchSize = intNumMatchingFiles * intNumSubSeqs  # Total number of sequences
    
    arrAllData = np.zeros((intNumChannels, intSubSeqTimePts, 0), dtype = np.float64)
    if (argDebug): print('arrAllData.shape = {}'.format(arrAllData.shape))
    
    print()
    
    lstAllSegDurations = []   # List of segment durations (in seconds)
    lstAllSamplingFreqs = []  # List of sampling frequencies (in Hz)
    lstAllChannels = []       # List of lists (of channel names)
    lstAllSequences = []      # List of sequence indices
    lstAllSubSequences = []   # List of subsequence indices (if one sequence is broken up into subsequences)
    
    # List of start/end time points and times (in seconds) for each subsequence
    arrAllStartEndTimePts  = np.zeros((0, 2))
    arrAllStartEndTimesSec = np.zeros((0, 2), dtype = np.float64)
    
    lstSeizureDurations = []  # List of seizure durations (in seconds) (not broken into subsequences)
    
    datLoopStart = utils.fnNow()
    print('Loop started on {}'.format(utils.fnGetDatetime(datLoopStart)))
    print()
    
    intNumProcessedFiles = 0
    fltFirstSamplingFreq = fltSamplingFreq  # Remember the sampling frequency of the first file
    lstFirstChannels = lstChannels          # Remember the list of channels of the first file
    
    print('fltFirstSamplingFreq = {}Hz'.format(fltFirstSamplingFreq))
    print('lstFirstChannels = {}'.format(lstFirstChannels))
    print()
    
    # SCALING
    if (argScalingParams):
        intScalingMode  = argScalingParams[0]
        tupScaledMinMax = argScalingParams[1]
        print('Scaling data using: intScalingMode = {}, tupScaledMinMax = {}'.format(intScalingMode, tupScaledMinMax))
        print()
        
        # TODO: Currently we are reading in the same data set twice, so this is not speed
        #       efficient. Maybe we should combine these two loops into one    
        # SCALING: Collect the statistics of the specified EDF files
        arrDataSetMin, arrDataSetMax, arrDataSetMean = fnGetCHBMITStats(lstMatchingFiles, argDebug = False)
        
        # Construct arrDataMin and arrDataMax based on the scaling mode
        arrDataMin, arrDataMax = fnGenMinMaxArrays(intScalingMode, arrDataSetMin, arrDataSetMax, argDebug = False)
        
    else:
        print('Data scaling is not performed')
        print()
        
    # Loop through each file in the target directory                             
    for intFullFilenameIdx, strFullFilename in enumerate(lstMatchingFiles):  # SCALING
        # Process only .edf files
        if strFullFilename.endswith('.edf'):
            print('Processing {}...'.format(strFullFilename))
            
            # Extract the filename from the full path
            strPath, strFilename = os.path.split(strFullFilename)
            
            # Read the .edf file
            strSegLabel, arrDataEDF, fltSegDurationEDF, fltSamplingFreq, lstChannels, intSequence, intNumChannels, intNumTimePtsEDF = fnReadEDFUsingPyEDFLib(strFullFilename, argDebug = False)
            print('  fltSegDurationEDF = {}s, fltSampingFreq = {}Hz'.format(fltSegDurationEDF, fltSamplingFreq))
            print()
            
            if (argScalingParams):
                # SCALING: Construct arrDataMinMax (C x 2) from arrDataMin and arrDataMax
                #          (C x N)
                # SCALING (normalize, filter, smoothing(?), augment (later))
                arrDataMinMax = np.concatenate((arrDataMin[:, intFullFilenameIdx][:, np.newaxis], arrDataMax[:, intFullFilenameIdx][:, np.newaxis]), axis = 1)
                arrDataEDFScaled = utils.fnMinMaxScaler(arrDataEDF, tupScaledMinMax, arrDataMinMax, argDebug = False)
            else:
                arrDataEDFScaled = arrDataEDF
                
            # Consistency check for sampling rate and channel labels
            if (fltSamplingFreq != fltFirstSamplingFreq):
                raise Exception('Sampling frequency not consistent across the data segments!')
                
            if (lstChannels != lstFirstChannels):
                raise Exception('Channels not consistent across the data segments!')
            
            # Check to see if an annotation file (.seizures) exists. If so, set strSegLabel
            # and determine the seizure start/end times
            
            # TODO: Change argAnnoSuffix to argReadAnnoTxt (boolean) to call fnReadCHBMITAnno()
            #       to read binary 'seizures' files when false, and call fnReadCHBMITAnnoTxt()
            #       to read ASCII 'annotation.txt' files when true
            #blnAnnoFound, lstStartEndTimePts = fnReadCHBMITAnno(strFullFilename, argAnnoSuffix)
            blnAnnoFound, lstStartEndTimePts = fnReadCHBMITAnnoTxt(strFullFilename, argAnnoSuffix, fltSamplingFreq)
            
            # SCALING
            # Break the segment up based on their respective segment types
            lstSegLabels, lstSegTypes, lstStartEndTimePts, lstStartEndTimeSecs, lstSegDurations, lstNumTimePts, lstDataSegs = fnBreakCHBMITSegment(arrDataEDFScaled, lstStartEndTimePts, fltSamplingFreq, argDebug = True)
                        
            # Loop through each segment to generate the appropriate subsequences 
            for intSegIdx, tupSeg in enumerate(zip(lstSegLabels, lstSegTypes, lstStartEndTimePts, lstStartEndTimeSecs, lstSegDurations, lstNumTimePts, lstDataSegs)):
                strSegLabel, intSegType, tupStartEndTimePt, tupStartEndTimeSec, fltSegDuration, intNumTimePts, arrDataSeg = [*tupSeg]
                
                # Extract the start and end time points of each segment
                intStartTimePt, intEndTimePt   = tupStartEndTimePt
                fltStartTimeSec, fltEndTimeSec = tupStartEndTimeSec
                
                if (fnIsSegState('ictal', intSegType)):
                    lstSeizureDurations.append((strFilename, fltSegDuration))
                    print('  => SEGMENT ({}) = ***{}***'.format(intSegIdx + 1, strSegLabel.upper()))
                else:
                    print('  => SEGMENT ({}) = {}'.format(intSegIdx + 1, strSegLabel.upper()))
                    
                print('    {:.6f}s ({}) -> {:.6f}s ({}), duration = {:.6f}s ({}), arrDataSeg.shape = {}'.format(fltStartTimeSec, intStartTimePt, fltEndTimeSec, intEndTimePt, fltSegDuration, intNumTimePts, arrDataSeg.shape))
                
                # Terminate the function with an error if argSubSeqDuration is longer than
                # fltSegDuration for the current segment
                if (argSubSeqDuration > fltSegDuration):
                    raise Exception('argSubSeqDuration ({}s) > fltSegDuration ({}s)'.format(argSubSeqDuration, fltSegDuration))
                    
                # Calculate the number of time points in each segment after the resampling
                intResampledTimePts = int(round((argResamplingFreq / fltSamplingFreq) * intNumTimePts))
                print('     intResampledTimePts = {}'.format(intResampledTimePts))

                # Calculate the number of time points in each segment after splitting up the segment
                # into shorter sequences
                intSubSeqTimePts = int(round((argSubSeqDuration / fltSegDuration) * intResampledTimePts))
                print('     intSubSeqTimePts = {}'.format(intSubSeqTimePts))
                
                # Analyze the number of subsequences to break down from the main segment, and
                # whether the time points can be equally divided among all the subsequences
                intNumSubSeqs = math.ceil(intResampledTimePts / intSubSeqTimePts)  # Total number of subsequences required
                intNumFullSubSeqs = intResampledTimePts // intSubSeqTimePts        # Number of completely filled subsequences
                intNumOrphanTimePts = intResampledTimePts % intSubSeqTimePts       # Number of orphan time points
                
                if (argDebug):
                    print('     intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts = {} / {} / {} ({:.2f}%)'.format(intNumSubSeqs, intNumFullSubSeqs, intNumOrphanTimePts, intNumOrphanTimePts/intSubSeqTimePts))
                    print()

                # If the number of subsequence time points specified results in the
                # last subsequence not being completely filled up, give a warning message
                if (intNumOrphanTimePts > 0):
                    print('     WARNING: Time points cannot be divided into complete subsequences')

                if (intNumFullSubSeqs > 1):
                    print('     Splitting each segment into {} subsequences based on intSubSeqTimePts = {}'.format(intNumFullSubSeqs, intSubSeqTimePts))
                    
                print()

                # Resample and round the results to the nearest integer since arrData[] is of type int16
                if (argResamplingFreq != fltFirstSamplingFreq):
                    arrDataResampled = signal.resample(arrDataSeg, intResampledTimePts, axis = 1)
                else:
                    arrDataResampled = arrDataSeg
                    
                if (argDebug): print('     arrDataResampled.shape = {} ({})'.format(arrDataResampled.shape, type(arrDataResampled[0][0])))

                # Truncate arrData[channels, time pts] to remove any orphan time points, if any
                intNumUsableTimePts = intSubSeqTimePts * intNumFullSubSeqs
                
                # If interictal segment, trim from the beginning. If ictal, trim from the end.
                # This is because the end of an interictal segment might be preictal if the
                # next segment is ictal, and the beginning of an ictal segment is more crucial
                # for training than the end of such a segment
                if (fnIsSegState('ictal', intSegType)):
                    intTrimStart = 0
                    intTrimEnd   = intNumUsableTimePts
                else:
                    intTrimStart = intResampledTimePts - intNumUsableTimePts
                    intTrimEnd   = intResampledTimePts
                    
                arrDataTruncated = arrDataResampled[:, intTrimStart:intTrimEnd]
                if (argDebug): print('     intNumUsableTimePts = {}, intTrimStart/End = [{}:{}], arrDataTrauncated.shape = {} ({})'.format(intNumUsableTimePts, intTrimStart, intTrimEnd, arrDataTruncated.shape, type(arrDataTruncated [0][0])))

                # Reshape arrDataTruncated[channels, time pts] into arrDataSplit[channels, subseq time pts, subsequences]
                # where the subsequences are split along the 3rd dimension
                arrDataSplit = arrDataTruncated.reshape(intNumChannels, -1, intSubSeqTimePts)
                arrDataSplit = np.swapaxes(arrDataSplit, 1, 2)
                if (argDebug): print('     arrDataSplit.shape = {} ({})'.format(arrDataSplit.shape, type(arrDataSplit[0][0][0])))
                
                # Calculate the actual start/end time points and times (in seconds) for each
                # trimmed segment (interictal or ictal)
                intTrimStartTimePt  = intTrimStart + intStartTimePt
                intTrimEndTimePt    = intTrimEnd + intStartTimePt
                fltTrimStartTimeSec = intTrimStartTimePt / argResamplingFreq
                fltTrimEndTimeSec   = intTrimEndTimePt / argResamplingFreq
                print('     intTrimStartTimePt = {:.6f}s ({}), intTrimEndTImePt = {:.6f}s ({})'.format(fltTrimStartTimeSec, intTrimStartTimePt, fltTrimEndTimeSec, intTrimEndTimePt))
                
                # Calculate the start/end time points and times (in seconds) for each subsequence
                arrStartEndTimePtsSplit = np.concatenate((np.arange(intTrimStartTimePt, intTrimEndTimePt, intSubSeqTimePts)[:, np.newaxis], 
                                                  np.arange(intTrimStartTimePt + intSubSeqTimePts, intTrimEndTimePt + 1, intSubSeqTimePts)[:, np.newaxis]), axis = 1)
                print('     arrStartEndTimePtsSplit.shape = {} ({})'.format(arrStartEndTimePtsSplit.shape, type(arrStartEndTimePtsSplit[0][0])))
                print('     arrStartEndTimePtsSplit = {} -> {}'.format(arrStartEndTimePtsSplit[0, :], arrStartEndTimePtsSplit[-1:, :]))

                arrStartEndTimesSecSplit = arrStartEndTimePtsSplit / argResamplingFreq
                print('     arrStartEndTimesSecSplit.shape = {} ({})'.format(arrStartEndTimesSecSplit.shape, type(arrStartEndTimesSecSplit[0][0])))
                print('     arrStartEndTimesSecSplit = {} -> {}'.format(arrStartEndTimesSecSplit[0, :], arrStartEndTimesSecSplit[-1:, :]))
                
                print()
                
                # Loop through each subsequence and save the data and metadata into
                # the appropriate data structures

                # TODO: To improve performance, instead of looping through each
                #       subsequence one by one we may be able to concatenate the
                #       whole batch while replicating the other parameters and then
                #       append them to the end of each list (done)

                # This new method takes about 3 mins to execute for chb01
                lstAllBaseFilenames.extend([strFilename] * intNumFullSubSeqs)
                lstAllSegLabels.extend([strSegLabel] * intNumFullSubSeqs)
                lstAllSegTypes.extend([intSegType] * intNumFullSubSeqs)
                arrAllData = np.concatenate((arrAllData, arrDataSplit), axis = 2)
                lstAllSegDurations.extend([argSubSeqDuration] * intNumFullSubSeqs)   # Record the duration after the split
                lstAllSamplingFreqs.extend([argResamplingFreq] * intNumFullSubSeqs)  # Record the resampled frequency
                lstAllChannels.extend([lstChannels] * intNumFullSubSeqs)
                lstAllSequences.extend([intSequence] * intNumFullSubSeqs)
                lstAllSubSequences.extend(list(np.arange(intNumFullSubSeqs)))
                
                # Concatenate the start/end time points and times (in seconds) for
                # each subsequence to the appropriate output data structures
                arrAllStartEndTimePts = np.concatenate((arrAllStartEndTimePts, arrStartEndTimePtsSplit), axis = 0)
                arrAllStartEndTimesSec = np.concatenate((arrAllStartEndTimesSec, arrStartEndTimesSecSplit), axis = 0)
                print('     arrAllStartEndTimePts.shape = {}, arrAllStartEndTimesSec.shape = {}'.format(arrAllStartEndTimePts.shape, arrAllStartEndTimesSec.shape))
                print()
                
                if (argDebug):
                    print('     arrAllData.shape = {}'.format(arrAllData.shape))
                    print()
                    
            intNumProcessedFiles = intNumProcessedFiles + 1
            
    datLoopEnd = utils.fnNow()
    print('Loop ended on {}'.format(utils.fnGetDatetime(datLoopEnd)))

    datLoopDuration = datLoopEnd - datLoopStart
    print('datLoopDuration = {}'.format(datLoopDuration))
    
    print('intNumProcessedFiles = {}'.format(intNumProcessedFiles))
    
    # Make sure that the number of files that matched the specified
    # extension is the same number of files that we actually processed
    if (intNumMatchingFiles != intNumProcessedFiles):
        print('WARNING: intNumMatchingFiles != intNumProcessedFiles!')
        
    print()
    
    print('len(lstAllBaseFilenames) = {}'.format(len(lstAllBaseFilenames)))
    print('len(lstAllSegLabels) = {}'.format(len(lstAllSegLabels)))
    print('len(lstAllSegTypes) = {}'.format(len(lstAllSegTypes)))
    print('arrAllData.shape = {} ({}) (features x sequence length x batch size)'.format(arrAllData.shape,  type(arrAllData[0][0][0])))
    print('len(lstAllSegDurations) = {}'.format(len(lstAllSegDurations)))
    print('len(lstAllSamplingFreqs) = {}'.format(len(lstAllSamplingFreqs)))
    print('len(lstAllChannels) = {}'.format(len(lstAllChannels)))
    print('len(lstAllSequences) = {}'.format(len(lstAllSequences)))
    print('len(lstAllSubSequences) = {}'.format(len(lstAllSubSequences)))
    print('len(lstSeizureDurations) = {}'.format(len(lstSeizureDurations)))
    
    # Print the shape of the data structures that store the start/end
    # time points and times (in seconds) for each subsequence
    print('arrAllStartEndTimePts.shape = {} ({})'.format(arrAllStartEndTimePts.shape, type(arrAllStartEndTimePts[0][0])))
    print('arrAllStartEndTimesSec.shape = {} ({})'.format(arrAllStartEndTimesSec.shape, type(arrAllStartEndTimesSec[0][0])))
    
    print()

    if (0):
        print('lstAllBaseFilenames = {}'.format(lstAllBaseFilenames))
        print('lstAllSegLabels = {}'.format(lstAllSegLabels))
        print('lstAllSegTypes = {}'.format(lstAllSegTypes))
        print('arrAllData = {}'.format(arrAllData))
        print('lstAllSegDurations = {}'.format(lstAllSegDurations))
        print('lstAllSamplingFreqs = {}'.format(lstAllSamplingFreqs))
        print('lstAllChannels = {}'.format(lstAllChannels))
        print('lstAllSequences = {}'.format(lstAllSequences))
        print('lstSeizureDurations = {}'.format(lstSeizureDurations))
        
        print()
    
    return lstAllBaseFilenames, lstAllSegLabels, lstAllSegTypes, arrAllData, lstAllSegDurations, lstAllSamplingFreqs, lstAllChannels, lstAllSequences, lstAllSubSequences, lstSeizureDurations, arrAllStartEndTimesSec


# In[ ]:


def fnWriteTestAnnoFiles(argEDFFilenames, argTestResults, argStartEndTimesSec, argFalsePositivesMask, argFalseNegativeMask, argParentPath, argAnnoSuffix = 'annotation.txt', argDebug = False):
    arrEDFFilenames_Incorrect = np.unique(
        argEDFFilenames[argTestResults[:, 1] != argTestResults[:, 2]], return_index = False, return_inverse = False, return_counts = False, axis = 0)
    
    utils.fnOSMakeDir(argParentPath)
    
    for strEDFFilename in arrEDFFilenames_Incorrect:
        if (argDebug): print('strEDFFilename = {}:'.format(strEDFFilename))
        
        arrEDFFilenameMask = argEDFFilenames == strEDFFilename
        
        strAnnoFullFilename = argParentPath + strEDFFilename + '.' + argAnnoSuffix
        print('strAnnoFullFilename = {}'.format(strAnnoFullFilename))
        
        # Open the annotation file for writing
        with open(strAnnoFullFilename, 'w') as fhAnnoFile:
            fhAnnoFile.write('Onset,Duration,Annotation\n')
            #print('Onset,Duration,Annotation')
            
            strAnnotation = 'Ii->Ic'
            for fltStartTime, fltEndTime in argStartEndTimesSec[np.logical_and(arrEDFFilenameMask, argFalsePositivesMask), :]:
                fhAnnoFile.write('+{:.7f},{:.6f},{}\n'.format(fltStartTime, fltEndTime - fltStartTime, strAnnotation))
                #print('+{:.7f},{:.6f},{}'.format(fltStartTime, fltEndTime - fltStartTime, strAnnotation))
                
            strAnnotation = 'Ic->Ii'
            for fltStartTime, fltEndTime in argStartEndTimesSec[np.logical_and(arrEDFFilenameMask, argFalseNegativeMask), :]:
                fhAnnoFile.write('+{:.7f},{:.6f},{}\n'.format(fltStartTime, fltEndTime - fltStartTime, strAnnotation))
                #print('+{:.7f},{:.6f},{}'.format(fltStartTime, fltEndTime - fltStartTime, strAnnotation))


# In[ ]:


# Function to export Kaggle prediction EEG data to a CSV file
#
# (1) Generate first column of timestamps based on sampling frequency
#     Use lstAllSegDurations[0] to get the final segment duration, and
#     lstAllSamplingFreqs[0] for the final sampling frequency
#
# (2) Convert raw data from row- to column-based

# NOTE: Assumes argData is a single, multi-channel segment with dimensions m x n,
#       where m is the number of channels, and n is the number of time points

def fnExportEEG2CSV(argFilename, argSegLabel, argData, argSegDuration, argSamplingFreq, argChannels, argSequence, argSubSequence, argCSVPath = './', argDebug = False):
    fltResamplingFreq = argSamplingFreq
    fltSubSeqDuration = argSegDuration
    
    # Extract the filename and file extension
    strBasename, strFileExt = os.path.splitext(argFilename)

    strCSVFilename = '{}-{}_{:.2f}Hz_{}s.csv'.format(strBasename, argSubSequence, argSamplingFreq, argSegDuration)
    strCSVFullFilename = os.path.join(argCSVPath, strCSVFilename)

    print('Saving CSV file to {}...'.format(strCSVFullFilename))
    
    if (argDebug):
        print('fltResamplingFreq = {}, fltSubSeqDuration = {}'.format(fltResamplingFreq, fltSubSeqDuration))
        print('argData.shape = {}'.format(argData.shape))

    arrSegmentData = argData
    if (argDebug):
        print('arrSegmentData.shape = {}'.format(arrSegmentData.shape))
        print()
        
    fltTimeDelta = 1/fltResamplingFreq * 1000  # Unit = ms
    intNumChannels, intNumTimePts = arrSegmentData.shape

    # Open the CSV file for writing
    with open(strCSVFullFilename, 'w') as fhCSVFile:

        if (argDebug):
            print('% OpenBCI Raw EEG Data Format (adapted by Caleb Chan)')
            print('%')
            print('% Segment label = {}'.format(argSegLabel))
            print('% Channels = {}'.format(argChannels))
            print('% Sequence = {}'.format(argSequence))
            print('% SubSequence = {}'.format(argSubSequence))
            print('%')
            print('% Sample Rate = {:.2f} Hz'.format(fltResamplingFreq))
            print('% SubSequence Duration = {}s'.format(fltSubSeqDuration))
            print('%')
            print('% First Column = Time (ms)')
            print('% Other Columns = EEG data in microvolts (uV), with one channel per column')

        fhCSVFile.write('% OpenBCI Raw EEG Data Format (adapted by Caleb Chan)\n')
        fhCSVFile.write('%\n')
        fhCSVFile.write('% Segment label = {}\n'.format(argSegLabel))
        fhCSVFile.write('% Channels = {}\n'.format(argChannels))
        fhCSVFile.write('% Sequence = {}\n'.format(argSequence))
        fhCSVFile.write('% SubSequence = {}\n'.format(argSubSequence))
        fhCSVFile.write('%\n')
        fhCSVFile.write('% Sample Rate = {:.2f} Hz\n'.format(fltResamplingFreq))
        fhCSVFile.write('% SubSequence Duration = {}s\n'.format(fltSubSeqDuration))
        fhCSVFile.write('%\n')
        fhCSVFile.write('% First Column = Time (ms)\n')
        fhCSVFile.write('% Other Columns = EEG data in microvolts (uV) with one channel per column\n')

        for intTimeStep in np.arange(intNumTimePts):
            fltTimeInMS = intTimeStep * fltTimeDelta

            lstDataRow = list(arrSegmentData[:, intTimeStep])
            strDataRowFormat = len(lstDataRow) * '{:.2f}, '
            strDataRow = strDataRowFormat.format(*lstDataRow).rstrip(', ')

            if (argDebug):
                print('{}, {:.1f}, {}'.format(intTimeStep, fltTimeInMS, strDataRow))

            fhCSVFile.write('{:.1f}, {}\n'.format(fltTimeInMS, strDataRow))

            if (argDebug):
                if (intTimeStep == 19): break

    fhCSVFile.close()  # Close the CSV file (optional)

