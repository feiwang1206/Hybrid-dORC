% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));

dataFolder = "/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image_abs_diff/labelsTra";
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [64 64];
audsImds = augmentedImageDatastore(imgSize,imds);

numNoiseSteps = 500;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);

img = read(audsImds);
img = img{1,1};
img = img{:};
img = rescale(img,-1,1);

mask=zeros(size(img));
% mask(:,1:2:end)=1;
idx=randperm(prod(imgSize),prod(imgSize)/4);
for i=1:length(idx)
    mask(floor((idx(i)-1)/imgSize(1))+1,mod(idx(i)-1,imgSize(1))+1)=1;
end
ref=fft(fft(img,[],1),[],2).*mask;
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

numInputChannels = 1;
net = createDiffusionNetwork(numInputChannels);

% deepNetworkDesigner(net)
%%
miniBatchSize = 32;
numEpochs = 50;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

doTraining = true;
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
        % numImages = 16;
        % displayFrequency = numNoiseSteps + 1;
        % generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);
    end
end
save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/net2d.mat',"net");
%%
% numImages = 1;
% displayFrequency = 10;
% figure
% tic;generatedImages = generateAndDisplayImages2(net,gpuArray(noisyImg),varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
pname_s=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13/'];
pname_s1=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image/'];
param=dc(pname_s,pname_s1,1,1);
figure
tic;generatedImages2 = generateAndDisplayImages2(net,param,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
% numImages = 1;
% displayFrequency = 10;
% figure
% tic;generatedImages = generateAndDisplayImages3(net,gpuArray(noisyImg),gpuArray(ref),gpuArray(mask),varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
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
% X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
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

    images = 1/sqrtOneMinusBeta*(images - predNoise) + addedNoise;

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
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(images(:,:,:,ii),[])
end
end
function images = generateAndDisplayImages2(net,param,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

% Reverse the diffusion process.
numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    for slice=1:param.dim(3)
        % Generate random noise.
        images = gpuArray(randn([imageSize numChannels numImages])+1i*randn([imageSize numChannels numImages]));
    
        % Predict the noise using the network.
        predictedNoise_r = predict(net,real(images),noiseStep);
        predictedNoise_i = predict(net,imag(images),noiseStep);
    
        sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
        predNoise_r = varianceSchedule(noiseStep)*predictedNoise_r/sqrt(1 - alphaBar(noiseStep));
        predNoise_i = varianceSchedule(noiseStep)*predictedNoise_i/sqrt(1 - alphaBar(noiseStep));
        predNoise=predNoise_r+1i*predNoise_i;
    
        images = images - predNoise;
        images3d(:,:,slice)=images;
    end
    images3d = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',1,'verbose_flag', 0,'tol',1e-5,'z0',images3d);
    for slice=1:param.dim(3)
        if noiseStep ~= 1
            z_r = randn([imageSize,numChannels,numImages]);
            z_i = randn([imageSize,numChannels,numImages]);
        else
            z_r = zeros([imageSize,numChannels,numImages]);
            z_i = zeros([imageSize,numChannels,numImages]);
        end
        addedNoise_r = sqrt(posteriorVariance(noiseStep))*z_r;
        addedNoise_i = sqrt(posteriorVariance(noiseStep))*z_i;
        images3d(:,:,slice) = 1/sqrtOneMinusBeta*(images3d(:,:,slice)) + addedNoise_r+1i*addedNoise_i;
    end
    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imagesc(array2mosaic(abs(images3d)));
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imagesc(array2mosaic(abs(images3d)));
end
end

function images = generateAndDisplayImages3(net,images,ref,mask,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
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
            imshow(abs(images(:,:,:,ii)),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(images(:,:,:,ii)),[])
end
end
