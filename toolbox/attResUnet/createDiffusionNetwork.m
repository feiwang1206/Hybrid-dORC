function net = createDiffusionNetwork(numImageChannels)
% CREATEDIFFUSIONNETWORK Create a network to predict the noise added to an
% image.

%   Copyright 2023 The MathWorks, Inc.

inputSize = [32 32 numImageChannels];
initialNumChannels = 64;
filterSize = [3 3];
numGroups = 32;
numHeads = 1;

% Backbone
layers = [
    % Image input
    imageInputLayer(inputSize,'Normalization',"none")
    convolution2dLayer(filterSize,initialNumChannels,'Padding',"same",'Name',"conv_in")

    % Encoder
    residualBlock(initialNumChannels,filterSize,numGroups,"1")
    residualBlock(initialNumChannels,filterSize,numGroups,"2")

    convolution2dLayer(filterSize,2*initialNumChannels,'Padding',"same",'Stride',2,'Name',"downsample_2")

    residualBlock(2*initialNumChannels,filterSize,numGroups,"3")
    attentionBlock(numHeads,2*initialNumChannels,numGroups,"3")
    residualBlock(2*initialNumChannels,filterSize,numGroups,"4")
    attentionBlock(numHeads,2*initialNumChannels,numGroups,"4")

    convolution2dLayer(filterSize,4*initialNumChannels,'Padding',"same",'Stride',2,'Name',"downsample_4")

    residualBlock(4*initialNumChannels,filterSize,numGroups,"5")
    residualBlock(4*initialNumChannels,filterSize,numGroups,"6")

    % Bridge
    residualBlock(4*initialNumChannels,filterSize,numGroups,"7")
    attentionBlock(numHeads,4*initialNumChannels,numGroups,"7")
    residualBlock(4*initialNumChannels,filterSize,numGroups,"8")
    attentionBlock(numHeads,4*initialNumChannels,numGroups,"8")

    % Decoder
    depthConcatenationLayer(2,'Name',"cat_9")
    residualBlock(4*initialNumChannels,filterSize,numGroups,"9")
    depthConcatenationLayer(2,'Name',"cat_10")
    residualBlock(4*initialNumChannels,filterSize,numGroups,"10")

    transposedConv2dLayer(filterSize,2*initialNumChannels,'Cropping',"same",'Stride',2,'Name',"upsample_10")

    depthConcatenationLayer(2,'Name',"cat_11")
    residualBlock(2*initialNumChannels,filterSize,numGroups,"11")
    attentionBlock(numHeads,2*initialNumChannels,numGroups,"11")
    depthConcatenationLayer(2,'Name',"cat_12")
    residualBlock(2*initialNumChannels,filterSize,numGroups,"12")
    attentionBlock(numHeads,2*initialNumChannels,numGroups,"12")

    transposedConv2dLayer(filterSize,initialNumChannels,'Cropping',"same",'Stride',2,'Name',"upsample_12")

    depthConcatenationLayer(2,'Name',"cat_13")
    residualBlock(initialNumChannels,filterSize,numGroups,"13")
    depthConcatenationLayer(2,'Name',"cat_14")
    residualBlock(initialNumChannels,filterSize,numGroups,"14")
    depthConcatenationLayer(2,'Name',"cat_end")

    % Output
    groupNormalizationLayer(numGroups)
    swishLayer
    convolution2dLayer(filterSize,numImageChannels,'Padding',"same");
    ];
net = dlnetwork(layers, 'Initialize',false);

% Add the noise step embedding input
numNoiseChannels = 1;
noiseStepEmbeddingLayers = [
    featureInputLayer(numNoiseChannels)
    sinusoidalPositionEncodingLayer(initialNumChannels)
    fullyConnectedLayer(4*initialNumChannels)
    swishLayer
    fullyConnectedLayer(4*initialNumChannels,'Name',"noiseEmbed")
    ];
net = addLayers(net, noiseStepEmbeddingLayers);

% Connect the noise step embedding to each residual block
numResidualBlocks = 14;
channelMultipliers = [1 1 2 2 4 4 4 4 4 4 2 2 1 1];
attentionBlockIndices = [3 4 7 8 11 12];

for ii = 1:numResidualBlocks
    numChannels = channelMultipliers(ii)*initialNumChannels;
    noiseStepConnectorLayers = [
        groupNormalizationLayer(numGroups,'Name',"normEmbed_"+ii)
        fullyConnectedLayer(numChannels,'Name',"fcEmbed_"+ii)
        ];
    net = addLayers(net,noiseStepConnectorLayers);
    net = connectLayers(net,"noiseEmbed", "normEmbed_"+ii);
    net = connectLayers(net,"fcEmbed_"+ii,"addEmbedRes_"+ii+"/in2");
end

% Add missing skip connections in each residual and attention block
for ii = 1:numResidualBlocks
    skipConnectionSource = "norm1Res_" + ii;
    numChannels = channelMultipliers(ii)*initialNumChannels;
    % Add 1x1 convolution to ensure the correct number of channels
    net = addLayers(net, convolution2dLayer([1,1], numChannels, 'Name',"skipConvRes_"+ii));
    net = connectLayers(net,skipConnectionSource,"skipConvRes_"+ii);
    net = connectLayers(net,"skipConvRes_"+ii,"addRes_"+ii+"/in2");
    if ismember(ii,attentionBlockIndices)
        skipConnectionSource = "normAttn_"+ii;
        net = connectLayers(net,skipConnectionSource,"addAttn_"+ii+"/in2");
    end
end

% Add missing skip connections between encoder and decoder
numEncoderResidualBlocks = 6;
for ii = 1:numEncoderResidualBlocks
    correspondingDecoderBlockIdx = numResidualBlocks - ii + 1;
    net = connectLayers(net,"addRes_"+ii, "cat_"+correspondingDecoderBlockIdx+"/in2");
end
net = connectLayers(net,"conv_in","cat_end/in2");
% Initialize the network
net = initialize(net);
end

% Helper functions
% Residual block
function layers = residualBlock(numChannels,filterSize,numGroups,name)
layers = [
    groupNormalizationLayer(numGroups,'Name',"norm1Res_"+name)
    swishLayer()
    convolution2dLayer(filterSize,numChannels,'Padding',"same")
    functionLayer(@(x,y) x + y,'Formattable',true,'Name',"addEmbedRes_"+name)
    groupNormalizationLayer(numGroups)
    swishLayer()
    convolution2dLayer(filterSize,numChannels,'Padding',"same")
    additionLayer(2,'Name',"addRes_"+name)
    ];
end
% Attention block
function layers = attentionBlock(numHeads,numKeyChannels,numGroups,name)
layers = [
    groupNormalizationLayer(numGroups,'Name',"normAttn_"+name)
    SpatialFlattenLayer()
    selfAttentionLayer(numHeads,numKeyChannels)
    SpatialUnflattenLayer()
    additionLayer(2,'Name',"addAttn_"+name)
    ];
end
