#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt
import pprint as pp


# In[ ]:


# Detect whether the script is running in Jupyter or in
# batch mode
def fnIsBatchMode(argDebug = False):
    try:
        get_ipython
        
    except:
        blnBatchMode = True
        if (argDebug): print('Batch mode detected')
        
    else:
        blnBatchMode = False
        print('Interactive mode detected')
        
    return blnBatchMode


# In[ ]:


def fnByte2GB(argBytes):
    return argBytes / (1024**3)


# In[ ]:


import psutil

def fnShowMemUsage():
    objVM = psutil.virtual_memory()
    print('Memory usage: Total = {:.2f}GB, Avail = {:.2f}GB, Used = {:.2f}GB ({}%), Free = {:.2f}GB'.format(
        fnByte2GB(objVM.total), fnByte2GB(objVM.available), fnByte2GB(objVM.used), objVM.percent, fnByte2GB(objVM.free)))


# In[ ]:


# Checks to see if a specified directory exists. If not, create a new one
def fnOSMakeDir(argPath):
    if (not os.path.exists(argPath)):
        print('The specified directory does not exist. Creating a new one: {}'.format(argPath))
        os.mkdir(argPath)
        
# Returns a list of full filenames recursively from a parent directory,
# sorted and filtered based on argFileExt
def fnGetFullFilenames(argParentPath, argFileExt, argDebug = False):
    lstFullFilenames = []
    
    # Loop recursively using argParentPath as the starting point
    for strRoot, lstDirs, lstFilenames in os.walk(argParentPath):
        if (argDebug):
            print('strRoot = {}'.format(strRoot))
            print('lstDirs = {}'.format(lstDirs))
            print('lstFilenames = {}'.format(lstFilenames))
            print()
            
        if (len(lstFilenames) > 0):
            lstFilenames.sort()
            
            # Loop through the list of filenames and reconstruct
            # with their full paths
            for strFilename in lstFilenames:
                # Process only files with extension argFileExt
                if strFilename.endswith(argFileExt):
                    # Append full path to each filename
                    strFullFilename = os.path.join(strRoot, strFilename)
                    lstFullFilenames.append(strFullFilename)
                    if (argDebug): print('  {}'.format(strFullFilename))
                    
    lstFullFilenames.sort()
    
    return lstFullFilenames


# In[ ]:


import datetime
from pytz import timezone

strTimeZone = 'America/Los_Angeles'

# Returns the current datetime
def fnNow():
    return datetime.datetime.now(timezone(strTimeZone))

# Returns a timestamp (for use in filenames) in the format:
# YYYY/MM/DD-HH:MM:SS
#def fnGenTimestamp(argDatetime = fnNow()):  # TODO: For some reason this doesn't work
def fnGenTimestamp(argDatetime = None):
    if (argDatetime == None):
        argDatetime = fnNow()
    return argDatetime.strftime('%Y%m%d-%H%M%S')

# Returns a date and time in human-readable format
#def fnGetDatetime(argDatetime = fnNow()):  # TODO: For some reason this doesn't work
def fnGetDatetime(argDatetime = None):
    if (argDatetime == None):
        argDatetime = fnNow()
    return argDatetime.strftime('%m/%d/%Y %H:%M:%S')


# In[ ]:


# Returns the appropriate boolean data type based on the
# value of the input string
def fnBool(argStr):
    if (argStr == 'True'):
         return True
    elif (argStr == 'False'):
         return False
    else:
         print('ERROR: fnBool() - argument must have value \'True\' or \'False\'')

# Returns true if the input argument is odd
def fnIsEven(argValue):
    if (argValue % 2 == 0):
        return True
    else:
        return False

# Returns true if all elements within a 1D numpy array are
# identical
def fnAllIdentical1D(argArray):
    return np.all(argArray == argArray[0], axis = 0)

# Perform minmax scaling on a C (channels) x T (time pts) data array to range
# specified by argScaledMinMax (tuple = (min, max)) based on the max and min
# values of the original data, as specified by argDataMinMax (a C x 2 array
# where C = channel, col[0] = min, and col[1] = max)
def fnMinMaxScaler(argData, argScaledMinMax, argDataMinMax, argDebug = False):
    fltScaledMin = argScaledMinMax[0]  # Get the min value after scaling
    fltScaledMax = argScaledMinMax[1]  # Get the max value after scaling
    
    # Extract the min and max columns from argDataMinMax
    arrDataMin = argDataMinMax[:, 0][:, np.newaxis]
    arrDataMax = argDataMinMax[:, 1][:, np.newaxis]
    if (argDebug):
        print('arrDataMin = {}'.format(pp.pformat(arrDataMin)))
        print('arrDataMax = {}'.format(pp.pformat(arrDataMax)))
    
    # Raise exceptions if the min/max of argData is outside of the range as
    # specified by argDataMinMax
    arrMinCheck = arrDataMin > argData
    arrMaxCheck = arrDataMax < argData
    
    if (arrMinCheck.any()):
        arrRows, arrCols = np.where(arrMinCheck)
        arrMinCheckRC = np.concatenate((arrRows[:, np.newaxis], arrCols[:, np.newaxis]), axis = 1)
        print('Min. check errors:\n{}'.format(np.concatenate((arrMinCheckRC, arrDataMin[arrMinCheckRC[:, 0]], argData[arrMinCheck][:, np.newaxis]), axis = 1)))
        raise Exception('arrDataMin > argData')
    if (arrMaxCheck.any()):
        arrRows, arrCols = np.where(arrMaxCheck)
        arrMaxCheckRC = np.concatenate((arrRows[:, np.newaxis], arrCols[:, np.newaxis]), axis = 1)
        print('Max. check errors:\n{}'.format(np.concatenate((arrMaxCheckRC, arrDataMax[arrMaxCheckRC[:, 0]], argData[arrMaxCheck][:, np.newaxis]), axis = 1)))
        raise Exception('arrDataMax < argData')
    
    # Perform the scaling operation and return a scaled C x T array
    arrScale = (fltScaledMax - fltScaledMin) / (arrDataMax - arrDataMin)
    arrDataScaled = (arrScale * argData) + fltScaledMin - (arrDataMin * arrScale)
    if (argDebug):
        print('arrScale = {}'.format(pp.pformat(arrScale)))
        print('arrDataScaled = {}'.format(pp.pformat(arrDataScaled)))
    
    return arrDataScaled


# In[ ]:


# Convert a tensor back to an np.array
def fnTensor2Array(argTensor, argTrainOnGPU):
    if (argTrainOnGPU):
        return argTensor.cpu().numpy()
    else:
        return argTensor.numpy()


# In[ ]:


# Find a key-value pair in a dictionary. If not found, return a default value
def fnFindInDct(argDictionary, argKey, argDefault):
    if (argKey in argDictionary):
        return argDictionary[argKey]
    else:
        return argDefault


# In[ ]:


import smtplib

# Send mail from Python script via SMTP

# NOTE: Need to login to Gmail account and turn on access
#       for less secure apps: Account -> Security ->
#       Less secure app access

def fnSendMail(argSMTPServer, argSMTPPort, argLogin, argPasswd, argFromEmail, argToEmails, argSubject, argBody):
    # Create an SMTP session
    objSMTP = smtplib.SMTP(argSMTPServer, argSMTPPort)

    # Use TLS for security protocol
    objSMTP.starttls()

    # Authentication using login and password in plain text
    objSMTP.login(argLogin, argPasswd)

    # Compose message to be sent
    strMesg = 'Subject: {}\n\n{}'.format(argSubject, argBody)

    # sending the mail
    objSMTP.sendmail(argFromEmail, argToEmails, strMesg)

    # terminating the session
    objSMTP.quit()


# In[ ]:


# Calculate performance metrics
def fnCalcPerfMetrics(argNumFalsePositives, argNumFalseNegatives, argNumTruePositives, argNumTrueNegatives):
    fltTruePositiveRate = argNumTruePositives / (argNumTruePositives + argNumFalseNegatives)
    fltTrueNegativeRate = argNumTrueNegatives / (argNumTrueNegatives + argNumFalsePositives)
    
    return fltTruePositiveRate, fltTrueNegativeRate


# In[ ]:


# NOTE: Assumes argData is a single, multi-channel segment with dimensions m x n,
#       where m is the number of channels, and n is the number of time points

def fnPlotEEG(argFilename, argSegLabel, argData, argSegDuration, argSamplingFreq, argChannels, argSequence, argSubSequence, argYMax = 0, argYMin = 0, argPlotFig = True, argSaveFig = False, argFigPath = './', argDebug = False):
    # NOTE: Not entire clear on why we need this, but without this
    #       Patient_1_preictal_segment_0001 to 0006 will fail to plot
    #       with the error:
    #
    #       OverflowError: In draw_path: Exceeded cell block limit
    
    #plt.rcParams['agg.path.chunksize'] = 10000
    
    if (argDebug):
        print('argFilename = {}'.format(argFilename))
        print('argSegLabel = {}'.format(argSegLabel))
        print('argData.shape = {}'.format(argData.shape))
        print('argSamplingFreq = {}'.format(argSamplingFreq))
        print('argChannel = {}'.format(argChannels))
        print('argSequence = {}'.format(argSequence))
        print('argSubSequence = {}'.format(argSubSequence))
        print()
        
    # If argData[] is 1D (single channel) add the channel dimension
    # back since the code assumes the size of the first dimension is
    # the number of channels
    if (len(argData.shape) == 1):
        argData = argData[np.newaxis, :]
        
    intNumChannels = argData.shape[0]
    arrTimeIntervals = np.arange(argData.shape[1])
    
    # These are the max, min, and mean values per channel
    arrDataMax  = np.max(argData, axis = 1)
    arrDataMin  = np.min(argData, axis = 1)
    arrDataMean = np.mean(argData, axis = 1)
    
    # These are the max, min, and mean values across all channels
    fltSegMax   = np.max(arrDataMax)
    fltSegMin   = np.min(arrDataMin)
    fltSegMean  = np.mean(arrDataMean)
    
    strTitle = '{} (subSeq {})'.format(argSegLabel, argSubSequence)
    print('{}:'.format(strTitle))
    
    arrXTicks = np.arange(arrTimeIntervals[0], arrTimeIntervals[-1], step = argSamplingFreq * 60)  # One min intervals
    arrXTicks = np.append(arrXTicks, arrTimeIntervals[-1] + 1)  # Append the very last interval
    if (argDebug): print('arrXTicks = {}'.format(arrXTicks))
    
    arrXLabels = np.arange(arrXTicks.shape[0]) * 60
    if (argDebug): print('arrXLabels = {}'.format(arrXLabels))
    
    for tupZip in zip(argChannels, arrDataMax, arrDataMin, arrDataMean):
        print('    {}\t{:.1f}\t{:.1f}\t{:.4f}'.format(*tupZip))
        
    print('\n    fltSegMax = {:.1f}, fltSegMin = {:.1f}, fltSegMean = {:.4f}\n'.format(fltSegMax, fltSegMin, fltSegMean))
    
    if (argPlotFig):
        figPlot, arrSubplot = plt.subplots(intNumChannels, 1, facecolor = 'w')  # Set background color to white

        if (argDebug): print('type(arrSubplot) = {}'.format(type(arrSubplot)))

        # If there is only one channel, arrSubplot is returned as a single
        # AxesSubplot object instead of an np.array, and the following code
        # will break. In this case, force create an np.array with a single
        # AxesSubplot object
        if (argData.shape[0] == 1):
            arrSubplot = np.array([arrSubplot])

        if (argDebug):
            print('type(arrSubplot) = {}'.format(type(arrSubplot)))
            print()

        arrSubplot[0].set_title(strTitle, fontsize = 16)
        
        # Loop through each channel and plot the data into each subplot
        for intChannel in range(intNumChannels):
            if (argYMax == 0):
                fltYMax = arrDataMax[intChannel]  # Default: use the max value in that channel for YMax
            else:
                fltYMax = argYMax  # Otherwise, use the value set in argYMax

            if (argYMin == 0):
                fltYMin = arrDataMin[intChannel]  # Default: use the max value in that channel for YMin
            else:
                fltYMin = argYMin  # Otherwise, use the value set in argYMin

            if (argDebug): print('    fltYMax = {:.1f},\tfltYMin = {:.1f}'.format(fltYMax, fltYMin))

            arrSubplot[intChannel].plot(arrTimeIntervals, argData[intChannel, :], label = argChannels[intChannel] + ' (***NOT*** Centered)')

            arrSubplot[intChannel].set_xlabel('time (s)')
            arrSubplot[intChannel].set_xlim(left = arrTimeIntervals[0], right = arrTimeIntervals[-1])
            arrSubplot[intChannel].set_xticks(arrXTicks)
            arrSubplot[intChannel].set_xticklabels(arrXLabels)
            for intXTick in arrXTicks:
                arrSubplot[intChannel].axvline(intXTick, color = (0.75, 0.75, 0.75), linestyle = ':')

            arrSubplot[intChannel].set_ylabel('uV (?)')
            arrSubplot[intChannel].set_ylim(top = fltYMax, bottom = fltYMin)

            arrSubplot[intChannel].legend(loc = 'lower left')

        # Bring subplots closer to each other
        figPlot.subplots_adjust(hspace = 0.1)

        # Hide x labels and tick labels for all but bottom plot.
        for ax in arrSubplot:
            ax.label_outer()

        # Set figure size to specific dimensions
        figPlot.set_size_inches(20, 50)
        figPlot.tight_layout(pad = 0)
        
        # NOTE: According to the following post online, when using the Jupyter
        #       magic command '%matplotlib inline', the figure must be saved using
        #       savefig() first before being displayed using show(). Doing this in
        #       the reverse order will cause the saved figure to be blank. Refer
        #       to the following post for more details:
        #
        #         https://problemsolvingwithpython.com/06-Plotting-with-Matplotlib/06.04-Saving-Plots/
        #
        
        if (argSaveFig):
            # Extract the filename and file extension
            strBasename, strFileExt = os.path.splitext(argFilename)

            strFigFilename = '{}-{}_{:.2f}Hz_{}s.png'.format(strBasename, argSubSequence, argSamplingFreq, argSegDuration)
            strFigFullFilename = os.path.join(argFigPath, strFigFilename)

            print('Saving figure to {}...'.format(strFigFullFilename))
            plt.savefig(strFigFullFilename)

        plt.show()

    return arrDataMax, arrDataMin, arrDataMean


# In[ ]:


# Plot the training versus validation loss after a model is trained
def fnPlotTrainValLosses(argTrainingLosses, argValidationLosses, argValPerEpoch = -1, argFigSize = (16, 8), argXLim = (), argYLim = (), argargDebug = False):
    plt.figure(figsize = argFigSize)
    
    if (argXLim):
        plt.xlim(argXLim)
        
    if (argYLim):
        plt.ylim(argYLim)
    
    plt.plot(argTrainingLosses, linestyle='--', marker='o', color='b', label = 'Training loss')
    plt.plot(argValidationLosses, linestyle='-', marker='x', color='r', label = 'Validation loss')
    
    if (argValPerEpoch > 0):
        for intEpoch in np.arange(argValPerEpoch, len(argTrainingLosses), argValPerEpoch):
            plt.axvline(intEpoch, label = 'Epoch {}'.format(int(intEpoch / argValPerEpoch)), linestyle = ':')
    
    plt.legend(frameon = False)

