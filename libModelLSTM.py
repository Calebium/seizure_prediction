#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn


# In[ ]:


'''
Create an LSTM model that will be used to analyze multichannel EEG signal
'''

class clsLSTM(nn.Module):
    # Initialize the model by setting up the layers
    def __init__(self, argFeaturesDim, argHiddenDim, argNumLayers, argOutputSize, argDropProb = 0.5, argDebug = False):
        super(clsLSTM, self).__init__()
        
        # Make these parameters accessible in the other methods
        self.intFeaturesDim = argFeaturesDim
        self.intHiddenDim   = argHiddenDim
        self.intNumLayers   = argNumLayers
        self.intOutputSize  = argOutputSize
        self.fltDropProb    = argDropProb
        
        # Define the structure of each layer in the model
        self.LSTMLayer = nn.LSTM(argFeaturesDim, argHiddenDim, argNumLayers,  # LSTM layer
                                 dropout = argDropProb, batch_first = True)
        
        self.DropoutLayer = nn.Dropout(p = argDropProb)                       # Dropout layer
        
        self.FCLayer = nn.Linear(argHiddenDim, argOutputSize)                 # Fully-connected layer
        
        
    # Display all the named parameters and their shapes in the model
    def showParams(self):
        intParamIdx = 0
        
        for tupParam in self.named_parameters():
            print('{}: {} -> {}'.format(intParamIdx, tupParam[0], tupParam[1].data.shape))
            intParamIdx = intParamIdx + 1            
            
            
    # Perform a forward pass on the model provided with input data and a previous
    # hidden state
    def forward(self, argDataIn, argHiddenIn, argDebug = False):        
        # NOTE: intBatchSize should be based on the first dimension of the input
        #       data, not by the intBatchSize defined when formatting data (this
        #       way we can feed in data with any batch size for testing). Otherwise,
        #       things may work during training, but will break during testing,
        #       when we feed in test data with different batch sizes (which is
        #       perfectly legal)
        intBatchSize = argDataIn.shape[0]
        
        # arrDataIn (batch size x sequence length, features dim) -> input data
        # arrHiddenIn/Out (num layers, batch size, hidden dim) -> hidden state
        # arrLSTMOut (batch size, sequence length, hidden dim) -> LSTM output
        # arrFCOut (batch size * sequence length, output size) -> FC layer output
        # arrOutput (batch_size, output_size) -> final output
        
        # Feed input data and the previous hidden state through the LSTM
        # (just like passing input date through a CNN)
        #arrLSTMOut, arrHiddenOut = self.LSTMLayer(argDataIn, argHiddenIn)
        arrLSTMOut, arrHiddenOut = self.LSTMLayer(argDataIn.float(), argHiddenIn)
        if (argDebug): print('arrLSTMOut.shape = {}'.format(arrLSTMOut.shape))
        
        # Pass through a dropout layer
        arrDropoutOut = self.DropoutLayer(arrLSTMOut)
        if (argDebug): print('arrDropoutOut.shape = {}'.format(arrDropoutOut.shape))
        
        # Reshape output to be [batch size * sequence length, hidden dim]
        #arrDropoutFlat = arrDropoutOut.view(-1, self.intHiddenDim)
        arrDropoutFlat = arrDropoutOut.contiguous().view(-1, self.intHiddenDim)
        if (argDebug): print('arrDropoutFlat.shape = {}'.format(arrDropoutFlat.shape))
        
        # Put the flattened LSTM output through a fully-connected layer to get final
        # output so arrFCOut[] will be [batch size * sequence length, output size]
        arrFCOut = self.FCLayer(arrDropoutFlat)
        if (argDebug): print('arrFCOut.shape = {}'.format(arrFCOut.shape))
        
        # Reshape the FC layer output to be [batch size, sequence length, output size]
        arrFCOutReshaped = arrFCOut.view(intBatchSize, -1, self.intOutputSize)
        if (argDebug): print('arrFCOutReshaped.shape = {}'.format(arrFCOutReshaped.shape))
                
        # Get the last output for each sequence, with shape becoming [batch size, output size]
        arrOutput = arrFCOutReshaped[:, -1, :]
        if (argDebug): print('arrOutput.shape = {}'.format(arrOutput.shape))
                
        # Return the last output of each sequence and a hidden state
        return arrOutput, arrHiddenOut
    
    
    # Initialize the hidden and cell states with zeros
    def initHidden(self, argBatchSize, argTrainOnGPU = False, argDebug = False):
        # Create two new tensors with sizes num layers x batch size x hidden dim,
        # initialized to zero, for both hidden state and cell state of the LSTM
        
        # Hidden state carries the short-term memory and cell state carries the
        # long-term memory:
        #   https://colah.github.io/posts/2015-08-Understanding-LSTMs/
        arrWeight = next(self.parameters()).data  # NOTE: Apparently self.parameters() is an iterator. But why
                                                  #       only get the next item? This is just so that we can
                                                  #       use new() to create the hidden & cell states with the
                                                  #       same dtype (but different shape)
        if (argDebug):
            print('arrWeight.shape = {} ({})'.format(arrWeight.shape, arrWeight.type()))
            
        # Hidden is a tuple of two tensors (hidden state, cell state) of the same shape (n_layers x
        # batch_size x hidden_dim) create using new() with the same dtype as weight, initialized to
        # zero_() in-place. The tuple requirement is documented in the PyTorch LSTM documentation
        # as (h_0, c_0)
        #   - https://pytorch.org/docs/stable/nn.html#lstm
        #   - https://pytorch.org/docs/0.3.1/tensors.html?highlight=new#torch.Tensor.new
        if (argTrainOnGPU):
            arrHiddenState = (arrWeight.new(self.intNumLayers, argBatchSize, self.intHiddenDim).zero_().cuda(),
                              arrWeight.new(self.intNumLayers, argBatchSize, self.intHiddenDim).zero_().cuda())
        else:
            arrHiddenState = (arrWeight.new(self.intNumLayers, argBatchSize, self.intHiddenDim).zero_(),
                              arrWeight.new(self.intNumLayers, argBatchSize, self.intHiddenDim).zero_())
        
        if (argDebug):
            print('intNumLayers = {}, argBatchSize = {}, intHiddenDim = {}'.format(self.intNumLayers, argBatchSize, self.intHiddenDim))
            print('arrHiddenState.shape = ({}, {})'.format(arrHiddenState[0].shape, arrHiddenState[1].shape))
            
        return arrHiddenState


# In[ ]:


def fnSaveLSTMModel(argModelDir, argModelName, argNumChannels, argSeqLen, argNumSegments, argModel, 
                    argNumEpochs, argBatchSize, argShuffleIndices, argShuffleData, argLearningRate, argPrintEvery, argGradClip, 
                    argTrainingStepLosses, argValidationStepLosses, argDebug = False, **argModelProperties):
    print('Saving model: argModelName = {}'.format(argModelName))
        
    dctModelCheckPt = {
        'intNumChannels':          argNumChannels,
        'intSeqLen':               argSeqLen,
        'intNumSegments':          argNumSegments,

        'intFeaturesDim':          argModel.intFeaturesDim,
        'intHiddenDim':            argModel.intHiddenDim,
        'intNumLayers':            argModel.intNumLayers,
        'intOutputSize':           argModel.intOutputSize,
        'fltDropProb':             argModel.fltDropProb,
        'dctStateDict':            argModel.state_dict(),

        'intNumEpochs':            argNumEpochs,        # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'intBatchSize':            argBatchSize,        # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'blnShuffleIndices':       argShuffleIndices,   # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'blnShuffleData':          argShuffleData,      # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'fltLearningRate':         argLearningRate,     # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'intPrintEvery':           argPrintEvery,       # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)
        'fltGradClip':             argGradClip,         # TODO: Replicated in dctModelProperties (not removed for backwards-compatibility)

        'lstTrainingStepLosses':   argTrainingStepLosses,
        'lstValidationStepLosses': argValidationStepLosses,
        
        'dctModelProperties':      argModelProperties   # Important training and model-specific parameters        
    }
    
    if (argDebug): print(dctModelCheckPt)
    
    # Save model to file system with 'write' and 'binary' options
    with open(argModelDir + argModelName, 'wb') as objModelFile:
        torch.save(dctModelCheckPt, objModelFile)


# In[ ]:


def fnLoadLSTMModel(argModelDir, argModelName, argDebug = False):
    print('Loading model: argModelName = {}'.format(argModelName))
    
    with open(argModelDir + argModelName, 'rb') as objModelFile:
        dctModelCheckPt = torch.load(objModelFile)

    # Extract saved parameters from the model file
    intNumChannels          = dctModelCheckPt['intNumChannels']
    intSeqLen               = dctModelCheckPt['intSeqLen']
    intNumSegments          = dctModelCheckPt['intNumSegments']

    intFeaturesDim          = dctModelCheckPt['intFeaturesDim']
    intHiddenDim            = dctModelCheckPt['intHiddenDim']
    intNumLayers            = dctModelCheckPt['intNumLayers']
    intOutputSize           = dctModelCheckPt['intOutputSize']
    fltDropProb             = dctModelCheckPt['fltDropProb']
    dctStateDict            = dctModelCheckPt['dctStateDict']

    intNumEpochs            = dctModelCheckPt['intNumEpochs']
    intBatchSize            = dctModelCheckPt['intBatchSize']
    blnShuffleIndices       = dctModelCheckPt['blnShuffleIndices']
    blnShuffleData          = dctModelCheckPt['blnShuffleData']
    fltLearningRate         = dctModelCheckPt['fltLearningRate']
    intPrintEvery           = dctModelCheckPt['intPrintEvery']
    fltGradClip             = dctModelCheckPt['fltGradClip']

    lstTrainingStepLosses   = dctModelCheckPt['lstTrainingStepLosses']
    lstValidationStepLosses = dctModelCheckPt['lstValidationStepLosses']
    
    # Return dctModelProperties{} if it exists. Otherwise, return an empty dictionary
    if ('dctModelProperties' in dctModelCheckPt.keys()):
        dctModelProperties  = dctModelCheckPt['dctModelProperties']
        blnHasModelProperties = True
    else:
        dctModelProperties  = {}
        blnHasModelProperties = False
    
    # Reconstruct the LSTM model with the saved parameters
    objModelLSTM = clsLSTM(intFeaturesDim, intHiddenDim, intNumLayers, intOutputSize, argDropProb = fltDropProb)
    objModelLSTM.load_state_dict(dctStateDict)
    
    return (intNumChannels, intSeqLen, intNumSegments, objModelLSTM,
            intNumEpochs, intBatchSize, blnShuffleIndices, blnShuffleData, fltLearningRate, intPrintEvery, fltGradClip,
            lstTrainingStepLosses, lstValidationStepLosses, dctModelProperties)

