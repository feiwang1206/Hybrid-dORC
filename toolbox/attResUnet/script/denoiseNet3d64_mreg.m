clear
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = true;

scale=1;
dataFolder = ['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista14_image_field/diffTra'];
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [64 64 48]/scale;
audsImds = augmentedImageDatastore(imgSize,imds);
numInputChannels = 2;
numOutputChannels=2;

numNoiseSteps = 20;
betaMin = 1e-4;
betaMax = 0.2;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);

img = read(audsImds);
img = img{1,1};
img = img{:};
% img = rescale(img,-1,1);
Img=img;

tiledlayout("flow");
nexttile
imshow(abs(complex(img(:,:,imgSize(3)/2,1),img(:,:,imgSize(3)/2))),[])
title("t = 0");
for i = 1:5
    nexttile
    noise = randn(size(img),'like',img);
    noiseStepsToApply = numNoiseSteps/5 * i;
    noisyImg = applyNoiseToImage(img,noise,noiseStepsToApply,varianceSchedule);

    % Extract the data from the dlarray.
    noisyImg = extractdata(noisyImg);
    imshow(abs(complex(noisyImg(:,:,imgSize(3)/2,1),noisyImg(:,:,imgSize(3)/2))),[])
    title("t = " + string(noiseStepsToApply));
end

pname_s=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista14/'];
pname_s1=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista14_image_field/'];
param=dc(pname_s,pname_s1,1,1);
%%
miniBatchSize = 2;
numEpochs = 200;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

mbq = minibatchqueue(audsImds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatch, ...
    'MiniBatchFormat',"SSSCB", ...
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

netname='/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/denoiseNet3d_mreg.mat';
if doTraining
    if exist(netname,'file')
        load(netname,"net");
    else
        net = createDiffusionNetwork3d_att1(imgSize,numInputChannels,numOutputChannels);
    end
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
        % printf(['loss:' num2str(loss)])
        % numImages = 1;
        % displayFrequency = 10;
        % pname_s=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13/'];
        % pname_s1=['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image/'];
        % param=dc(pname_s,pname_s1,1,scale);
        % generateAndDisplayImages2(net,param,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);
        % %%
        generateAndDisplayImages2(net,param,varianceSchedule,imgSize,numInputChannels);
    end
    save(netname,"net");
else
    load(netname,"net");
end
%%
% numImages = 1;
% displayFrequency = 10;
% figure
% tic;generatedImages = generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
figure
tic;generatedImages2 = generateAndDisplayImages2(net,param,varianceSchedule,imgSize,numInputChannels);toc
%%
function noisyImg = applyNoiseToImage(img,noiseToApply,noiseStep,varianceSchedule)
alphaBar = cumprod(1 - varianceSchedule);
alphaBarT = dlarray(alphaBar(noiseStep),"CBSSS");

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
X = cat(5,data{:});

% Rescale the images so that the pixel values are in the range [-1 1].
% X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end
%%
function images = generateAndDisplayImages(net,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
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
    predictedNoise = predict(net,images,noiseStep);
    % predictedNoise = predict(net,images);

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
            imshow(abs(complex(images(:,:,imageSize(3)/2,1,ii),images(:,:,imageSize(3)/2,2,ii))),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(complex(images(:,:,imageSize(3)/2,1,ii),images(:,:,imageSize(3)/2,2,ii))),[])
end
end

%%
function images = generateAndDisplayImages2(net,param,varianceSchedule,imageSize,numChannels)
% Generate random noise.
images(:,:,:,1) = gpuArray(randn([imageSize 1 1]));
images(:,:,:,2) = gpuArray(randn([imageSize 1 1]));

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

% Reverse the diffusion process.
numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    if noiseStep ~= 1
        z = randn([imageSize,numChannels,1]);
    else
        z = zeros([imageSize,numChannels,1]);
    end

    % Predict the noise using the network.
    predictedNoise = predict(net,images,noiseStep);
    % predictedNoise = predict(net,images);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));

    tmp = images - predNoise;
    images0=complex(tmp(:,:,:,1),tmp(:,:,:,2));

    % images0 = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',1,'verbose_flag', 0,'tol',1e-5,'z0',images0);
    images0 = regularizedReconstruction(param.Fg,param.rawdata,param.P{:},'maxit',1,'verbose_flag', 0,'tol',1e-5,'z0',images0);
    % images0 = single(gather(ReconWavFISTA_mreg(param.init, param.Fg, double(0.05*max(abs(col(param.init)))),...
    %     param.W, double(param.alpha), double(images0), 1, true)));
    images(:,:,:,1)=real(images0);
    images(:,:,:,2)=imag(images0);
    
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;

    % loss=l2norm(gather(param.ref-complex(images(:,:,:,1),images(:,:,:,2))));
    % printf([num2str(noiseStep) '-loss:' num2str(loss)])
    % imagesc(abs(complex(images(:,:,imageSize(3)/2,1),images(:,:,imageSize(3)/2,2))));colormap gray;title("t = "+ noiseStep)

    loss=l2norm(gather(param.ref-images0));
    printf([num2str(noiseStep) '-loss:' num2str(loss)])
    imagesc(abs(images0(:,:,imageSize(3)/2,1)));colormap gray;title("t = "+ noiseStep)

    drawnow
end

% Display final images.
% tLay = tiledlayout("flow");
% title(tLay,['t = 0, loss:' num2str(loss)])
% for ii = 1:numImages
%     nexttile
%     imshow(abs(complex(images(:,:,imageSize(3)/2,1,ii),images(:,:,imageSize(3)/2,2,ii))),[]),
% end
end
%%
function images = generateAndDisplayImages3(net,Img,images,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
% Generate random noise.
% images = gpuArray(randn([imageSize numChannels numImages]));

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
    % predictedNoise = predict(net,images);

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
            imshow(abs(complex(images(:,:,imageSize(3)/2,1,ii),images(:,:,imageSize(3)/2,2,ii))),[])
        end
        drawnow
    end
end

% Display final images.
loss=mse(col(Img),col(images));
printf(['loss:' num2str(loss)])
tLay = tiledlayout("flow");
title(tLay,['t = 0, loss:' num2str(loss)])
for ii = 1:numImages
    nexttile
    imshow(abs(complex(images(:,:,imageSize(3)/2,1,ii),images(:,:,imageSize(3)/2,2,ii))),[]),
end
end
