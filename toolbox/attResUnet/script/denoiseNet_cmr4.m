% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = true;

dataFolder = "/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2_zf/diff";
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [512 512]/8;
audsImds = augmentedImageDatastore(imgSize,imds);

% dataFolder = "/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2_zf/labelsTra";
% volReader = @(x) matRead(x);
% volLoc = dataFolder;
% imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
% imgSize = [512 512]/8;
% audsImds_label = augmentedImageDatastore(imgSize,imds);

numInputChannels = 2;

% numNoiseSteps = 50;
% betaMin = 1e-4;
% betaMax = 0.2;
% varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
numNoiseSteps = 1;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
%%
img = read(audsImds);
img = img{2,1};
img = img{:};
% img = img(:,:,1)+1i*img(:,:,2);
% img = rescale(img,-1,1);
mask=zeros(imgSize);
% mask(:,1:2:end)=1;
% idx=randperm(imgSize(1),floor(imgSize(1)/4));
idx=mod(floor(randn(floor(imgSize(1)/4),1)*imgSize(1)/4+imgSize(1)/2),imgSize(1))+1;
for i=1:length(idx)
    mask(idx,:)=1;
end
mask(imgSize(1)/2+1,:)=1;
% idx=randperm(prod(imgSize),floor(prod(imgSize)/2));
% for i=1:length(idx)
%     mask(floor((idx(i)-1)/imgSize(1))+1,mod(idx(i)-1,imgSize(2))+1)=1;
% end
Img=img;
% %%
% tiledlayout("flow");
% nexttile
% imshow(abs(complex(img(:,:,1),img(:,:,2))),[])
% title("t = 0");
% for i = 1:5
%     nexttile
%     noise = randn(size(img),'like',img);
%     noiseStepsToApply = 1/5 * i;
%     noisyImg = applyNoiseToImage(img,noise,noiseStepsToApply/numNoiseSteps);
% 
%     % Extract the data from the dlarray.
%     % noisyImg = extractdata(noisyImg);
%     imshow(abs(complex(noisyImg(:,:,1),noisyImg(:,:,2))),[])
%     title("t = " + string(noiseStepsToApply));
% end
%%
% deepNetworkDesigner(net)

miniBatchSize = 32;
numEpochs = 200;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

mbq = minibatchqueue(audsImds, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatch, ...
    'MiniBatchFormat',"SSCB", ...
    'PartialMiniBatch',"discard");
% mbq_label = minibatchqueue(audsImds_label, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'MiniBatchFcn',@preprocessMiniBatch, ...
%     'MiniBatchFormat',"SSCB", ...
%     'PartialMiniBatch',"discard");

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
    % net = createDiffusionNetwork(numInputChannels);
     load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/deartefactNet2d_cmr_iter2_1.mat',"net");
   
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            img = next(mbq);
            % targetNoise = next(mbq_input);

            % Generate a random noise step.
            noiseStep = dlarray(randi(numNoiseSteps,[1 miniBatchSize],'Like',img),"CB");

            % Apply noise to the image.
            [~,Img_degrade] = applyNoiseToImage(img,(noiseStep)/numNoiseSteps);
            [~,Img_degrade2] = applyNoiseToImage(img,(noiseStep-1)/numNoiseSteps);
            % Compute loss.
            [loss,gradients] = dlfeval(@modelLoss,net,Img_degrade,(noiseStep),Img_degrade2-Img_degrade);

            % Update model.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, ...
                learnRate,gradientDecayFactor,squaredGradientDecayFactor);

            % Record metrics.
            recordMetrics(monitor,iteration,'Loss',loss);
            updateInfo(monitor,'Epoch',epoch,'Iteration',iteration);
            monitor.Progress = 100 * iteration/numIterations;
        end

        % Generate and display a batch of generated images.
        printf(['loss:' num2str(loss)])
        generateAndDisplayImages(net,Img,numNoiseSteps);

    end
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/deartefactNet2d_cmr_iter2_1.mat',"net");
else
    load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/deartefactNet2d_cmr_iter2_1.mat',"net");
end
%%
figure
tic;generatedImages = generateAndDisplayImages(net,Img,numNoiseSteps);toc
%%
function [Img_clean,Img_degrade] = applyNoiseToImage(img,noiseStep)
    img = extractdata(img);
    Img_clean = dlarray(img(:,:,3:4,:),'SSCB');
    Img_input = dlarray(img(:,:,1:2,:),'SSCB');
    Img_degrade = (1-noiseStep).*Img_clean + noiseStep.*Img_input;
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
% X = rescale(X,-1,1);
end
%%
function Img_predict = generateAndDisplayImages(net,Img,numsteps)
% Generate random noise.
Img_input=Img(:,:,1:2);
Img_clean=Img(:,:,3:4);
Img_degrade=Img_input;

for noiseStep = (numsteps):-1:1
    Img_predict = predict(net,Img_degrade,noiseStep)+Img_degrade;
    Img_degrade = Img_predict;
% loss=l2norm(Img_clean-Img_predict);
% printf(['loss:' num2str(loss)])

    tLay = tiledlayout("flow");
    title(tLay,"t = "+ noiseStep)
    for ii = 1:1
        nexttile
        imshow(abs(complex(Img_degrade(:,:,1,ii),Img_degrade(:,:,2,ii))),[])
    end
    drawnow
end

% Display final images.
loss=l2norm(Img_clean-Img_predict);
printf(['loss:' num2str(loss)])

tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(abs(complex(Img_degrade(:,:,1,ii),Img_degrade(:,:,2,ii))),[])
end
end

function images = generateAndDisplayImages2(net,images,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
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
            imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
end
end

function images = generateAndDisplayImages3(net,Img,mask,varianceSchedule,imageSize,numImages,numChannels,displayFrequency)
% Generate random noise.
images = gpuArray(randn([imageSize numChannels numImages]));

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

ref=fftshift(fft(fft(complex(Img(:,:,1),Img(:,:,2)),[],1),[],2));
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

    images=images - predNoise;
    imagesf=fftshift(fft(fft(complex(images(:,:,1),images(:,:,2)),[],1),[],2));
    imagesf=ref.*mask+imagesf.*(1-mask);
    tmp=ifft(ifft(fftshift(imagesf),[],1),[],2);
    images(:,:,1)=real(tmp);
    images(:,:,2)=imag(tmp);
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ noiseStep)
        for ii = 1:numImages
            nexttile
            imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
end
end
