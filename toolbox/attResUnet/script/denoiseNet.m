addpath(genpath('/home/ubuntu/Documents/MATLAB/metal_artifact_simulation-master'));

% unzip("DigitsData.zip");
doTraining = true;
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder,'IncludeSubfolders',true);
imgSize = [32 32];
audsImds = augmentedImageDatastore(imgSize,imds);
    numInputChannels = 1;

% numNoiseSteps = 50;
% betaMin = 1e-4;
% betaMax = 0.2;
% varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
numNoiseSteps = 500;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
%%
img = read(audsImds);
img = img{1,1};
img = img{:};
img = rescale(img,-1,1);

dim=[32 32]*1;
mask=zeros(size(img));
% mask(:,1:2:end)=1;
idx=randperm(1024,256);
for i=1:length(idx)
    mask(floor((idx(i)-1)/32)+1,mod(idx(i)-1,32)+1)=1;
end
ref=fft(fft(img,[],1),[],2).*mask;
img_input=double(img+1);
metal=zeros(size(img));
metal(25:25,15:15)=max(col(img))*10;
img_metal=img_input+metal;
% metal_p=radon(metal,1:180);
% mask_mar=ones(size(metal_p));
% mask_mar(metal_p>0)=0;
% img_metal=img+metal;
% proj=exp(-radon(img_metal,1:180)/10);
% proj=imnoise(proj/1e12,'poisson');
% proj=-log(proj)*10;
% proj(proj<=0)=0;
config = set_config_for_artifact_simulation2(20/dim(1),dim(1),32);
proj = fanbeam(img_metal,...  
                  config.SOD,...
                  'FanSensorGeometry','arc',...
                  'FanSensorSpacing', config.angle_size, ...
                  'FanRotationIncrement',360/config.angle_num);
y=1e7*exp(-proj);
noisy_y = power(10, config.noise_scale)*imnoise(y/power(10, config.noise_scale),'poisson');
proj = -log(noisy_y/1e7); 
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
%%
if doTraining
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
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages = generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
noisyImg = applyNoiseToImage(Img,noise,100,varianceSchedule);
noisyImg = extractdata(noisyImg);
figure
tic;generatedImages2 = generateAndDisplayImages2(net,gpuArray(noisyImg),100,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages3 = generateAndDisplayImages3(net,ref,mask,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages_mar = generateAndDisplayImages_mar(net,proj,metal,config,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
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
function images = generateAndDisplayImages(net,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
% Generate random noise.
images = randn([imageSize numChannels numImages]);

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
    predictedNoise = predict(net,images,noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));

    images = 1/sqrtOneMinusBeta*(images - predNoise);

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imshow(images(:,:,:,ii),[])
        end
        drawnow
    end
    images = images + addedNoise;
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(images(:,:,:,ii),[])
end
end

function images = generateAndDisplayImages2(net,images,numNoiseSteps,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
% Generate random noise.
% images = gpuArray(randn([imageSize numChannels numImages]));

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

% Reverse the diffusion process.
% numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    if noiseStep ~= 1
        z = randn([imageSize,numChannels,numImages]);
    else
        z = zeros([imageSize,numChannels,numImages]);
    end

    % Predict the noise using the network.
    predictedNoise = predict(net,images,noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));

    images = 1/sqrtOneMinusBeta*(images - predNoise);

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imshow((images(:,:,:,ii)),[])
        end
        drawnow
    end
    images = images + addedNoise;
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow((images(:,:,:,ii)),[])
end
end

function images = generateAndDisplayImages3(net,ref,mask,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
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
        z_r = randn([imageSize,numChannels,numImages]);
        z_i = randn([imageSize,numChannels,numImages]);
    else
        z_r = zeros([imageSize,numChannels,numImages]);
        z_i = zeros([imageSize,numChannels,numImages]);
    end

    % Predict the noise using the network.
    predictedNoise_r = predict(net,real(images),noiseStep);
    predictedNoise_i = predict(net,imag(images),noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise_r = sqrt(posteriorVariance(noiseStep))*z_r;
    addedNoise_i = sqrt(posteriorVariance(noiseStep))*z_i;
    predNoise_r = varianceSchedule(noiseStep)*predictedNoise_r/sqrt(1 - alphaBar(noiseStep));
    predNoise_i = varianceSchedule(noiseStep)*predictedNoise_i/sqrt(1 - alphaBar(noiseStep));
    predNoise=predNoise_r+1i*predNoise_i;

    images=images - predNoise;
    imagesf=fft(fft(images,[],1),[],2);
    imagesf=ref.*mask+imagesf.*(1-mask);
    images=ifft(ifft(imagesf,[],1),[],2);
    images = 1/sqrtOneMinusBeta*(images) + addedNoise_r+1i*addedNoise_i;

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imshow(real(images(:,:,:,ii)),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(real(images(:,:,:,ii)),[])
end
end


function images = generateAndDisplayImages_mar(net,ref,metal,config,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
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
    images_input=gather((images+1)+metal);
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
    images = ifanbeam(images_p,...
                   config.SOD,...
                   'FanSensorGeometry','arc',...
                   'FanSensorSpacing',config.angle_size,...
                   'OutputSize',config.output_size,... 
                   'FanRotationIncrement',360/config.angle_num,...
                   'Filter', config.filter_name,...
                   'FrequencyScaling', config.freqscale);
    images=images-1-metal;
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imshow(real(images(:,:,:,ii)),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(real(images(:,:,:,ii)),[])
end
end
