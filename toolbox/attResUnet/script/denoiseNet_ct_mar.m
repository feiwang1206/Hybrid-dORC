addpath(genpath('/home/ubuntu/Documents/MATLAB/metal_artifact_simulation-master'));

% diffpath='/home/ubuntu/Desktop/share/diffusion/';
% N=64;
% addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
% doTraining = false;
% 
% dataFolder = diffpath;
% volReader = @(x) matRead(x);
% volLoc = dataFolder;
% imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
% imgSize = [N N];
% audsImds = augmentedImageDatastore(imgSize,imds);

doTraining = false;
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder,'IncludeSubfolders',true);
imgSize = [32 32];
audsImds = augmentedImageDatastore(imgSize,imds);

numInputChannels = 1;

numNoiseSteps = 500;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
%%
img = read(audsImds);
img = img{floor(rand*100)+1,1};
img = img{:};
% img = rescale(img,-1,1);

img_input=double(img+1)/5;
metal=zeros(size(img));
metal(25:25,15:15)=max(col(img))*10;
img_metal=img_input+metal;
config = set_config_for_artifact_simulation2(20/N,N,N);
proj = fanbeam(img_metal,...  
                  config.SOD,...
                  'FanSensorGeometry','arc',...
                  'FanSensorSpacing', config.angle_size, ...
                  'FanRotationIncrement',360/config.angle_num);
y=exp(-proj);
scale=1e7;
noisy_y = scale*imnoise(y/scale,'poisson');
proj = -log(noisy_y); 
sim = ifanbeam(proj,...
               config.SOD,...
               'FanSensorGeometry','arc',...
               'FanSensorSpacing',config.angle_size,...
               'OutputSize',config.output_size,... 
               'FanRotationIncrement',360/config.angle_num,...
               'Filter', config.filter_name,...
               'FrequencyScaling', config.freqscale);

% img_input=imresize(double(img+1)/2,dim);
% metal_input=imresize(metal,dim);
% img_input(img_input<0)=0;
% metal_input(metal_input<0)=0;
% metal_input(metal_input>0)=1;
% phantom = create_phantom(dim(1), dim(1), floor(dim(1)/2)-2, config.mu_water);
% config.correction_coeff = water_correction(phantom, config);
% [sim,proj_clean,proj,sim_clean] = metal_artifact_simulation2(img_input, metal_input, config);
%%

Img=img;

tiledlayout("flow");
nexttile
imshow(img,[])
title("t = 0");
for i = 1:5
    nexttile
    noise = randn(size(img),'like',img);
    noiseStepsToApply = numNoiseSteps/5 * i;
    noisyImg = applyNoiseToImage(img,noise,noiseStepsToApply,varianceSchedule);

    % Extract the data from the dlarray.
    noisyImg = extractdata(noisyImg);
    imshow(noisyImg,[])
    title("t = " + string(noiseStepsToApply));
end
%%
% deepNetworkDesigner(net)

miniBatchSize = 128;
numEpochs = 50;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

mbq = minibatchqueue(audsImds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatch, ...
    'MiniBatchFormat',"SSCB", ...
    'PartialMiniBatch',"discard");

averageGrad = [];
averageSqGrad = [];

numObservationsTrain = numel(imds.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

if doTraining
    monitor = trainingProgressMonitor(...
        'Metrics',"Loss", ...
        'Info',["Epoch","Iteration"], ...
        'XLabel',"Iteration");
end

if doTraining
    %%
    net = createDiffusionNetwork(numInputChannels);
    
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            img = next(mbq);

            % Generate random noise.
            targetNoise = randn(size(img),'Like',img);

            % Generate a random noise step.
            noiseStep = dlarray(randi(numNoiseSteps,[1 miniBatchSize],'Like',img),"CB");

            % Apply noise to the image.
            noisyImage = applyNoiseToImage(img,targetNoise,noiseStep,varianceSchedule);

            % Compute loss.
            [loss,gradients] = dlfeval(@modelLoss,net,noisyImage,noiseStep,targetNoise);

            % Update model.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, ...
                learnRate,gradientDecayFactor,squaredGradientDecayFactor);

            % Record metrics.
            recordMetrics(monitor,iteration,'Loss',loss);
            updateInfo(monitor,'Epoch',epoch,'Iteration',iteration);
            monitor.Progress = 100 * iteration/numIterations;
        end

        % Generate and display a batch of generated images.
        numImages = 16;
        displayFrequency = numNoiseSteps + 1;
        generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);
    end
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/denoiseNet2d_50.mat',"net");
else
    % If doTraining is false, download and extract the pretrained network from the MathWorks website.
    pretrainedNetZipFile = matlab.internal.examples.downloadSupportFile('nnet','data/TrainedDiffusionNetwork.zip');
    unzip(pretrainedNetZipFile);
    load("DiffusionNetworkTrained/DiffusionNetworkTrained.mat");
end
%%
% numImages = 1;
% displayFrequency = 10;
% figure
% tic;generatedImages = generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages_mar = generateAndDisplayImages_mar(net,proj,metal,config,varianceSchedule,imgSize,numImages,numInputChannels);toc
%%
function recon = generateAndDisplayImages_mar(net,ref,metal,config,varianceSchedule,imageSize,numImages,numChannels)
% Generate random noise.
images = gpuArray(randn([imageSize numChannels numImages]));

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

% Reverse the diffusion process.
numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    if noiseStep ~= 1
        z = randn([imageSize,numChannels,numImages]);
    else
        z = zeros([imageSize,numChannels,numImages]);
    end

    % Predict the noise using the network.
    predictedNoise = predict(net,real(images),noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));

    images=images - predNoise;
    images_input=abs(gather((images+1)/5));
    proj = fanbeam(images_input,...  
                      config.SOD,...
                      'FanSensorGeometry','arc',...
                      'FanSensorSpacing', config.angle_size, ...
                      'FanRotationIncrement',360/config.angle_num);
    proj_metal = fanbeam(metal,...  
                      config.SOD,...
                      'FanSensorGeometry','arc',...
                      'FanSensorSpacing', config.angle_size, ...
                      'FanRotationIncrement',360/config.angle_num);
    proj_metal(proj_metal<0)=0;
    proj_metal(proj_metal>0)=1;
    images_p=ref.*(1-proj_metal)+proj.*proj_metal;
    recon = ifanbeam(images_p,...
                   config.SOD,...
                   'FanSensorGeometry','arc',...
                   'FanSensorSpacing',config.angle_size,...
                   'OutputSize',config.output_size,... 
                   'FanRotationIncrement',360/config.angle_num,...
                   'Filter', config.filter_name,...
                   'FrequencyScaling', config.freqscale);
    images=(recon)*5-1;
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;
    if mod(noiseStep,10)==0
        imagesc(recon);colorbar;colormap gray;
        title(num2str(noiseStep));
        drawnow
    end
end
end

function noisyImg = applyNoiseToImage(img,noiseToApply,noiseStep,varianceSchedule)
alphaBar = cumprod(1 - varianceSchedule);
alphaBarT = dlarray(alphaBar(noiseStep),"CBSS");

noisyImg = sqrt(alphaBarT).*img + sqrt(1 - alphaBarT).*noiseToApply;
end
function [loss, gradients] = modelLoss(net,X,Y,T)
% Forward data through the network.
noisePrediction = forward(net,X,Y);

% Compute mean squared error loss between predicted noise and target.
loss = mse(noisePrediction,T);

gradients = dlgradient(loss,net.Learnables);
end
function X = preprocessMiniBatch(data)
% Concatenate mini-batch.
X = cat(4,data{:});

% Rescale the images so that the pixel values are in the range [-1 1].
X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end


