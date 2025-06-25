% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = false;

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
numNoiseSteps = 10;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
%%
img = read(audsImds);
img = img{1,1};
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
netname='/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/deartefactNet2d_cmr_iter_10.mat';
if doTraining
    if exist(netname,'file')
        load(netname,"net");
    else
        net = createDiffusionNetwork(numInputChannels);
    end
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
            [Img_clean,Img_degrade] = applyNoiseToImage(img,(noiseStep)/numNoiseSteps);

            % Compute loss.
            [loss,gradients] = dlfeval(@modelLoss,net,Img_degrade,(noiseStep),Img_clean);

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
    save(netname,"net");
else
    load(netname,"net");
end
%%
figure
tic;generatedImages = generateAndDisplayImages(net,Img,numNoiseSteps);toc
%%
figure
image=load('/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2_zf/inputsVal/P001_1_10.mat');
imagef=fftshift(fft(fft(complex(image.image(:,:,1),image.image(:,:,2)),[],1),[],2));
imagef=imagef((512/2-32):(512/2+31),(256/2-32):(256/2+31),:);
image=ifft(ifft(ifftshift(imagef),[],1),[],2)/32;
image_input(:,:,1)=real(image);
image_input(:,:,2)=imag(image);
image=load('/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2_zf/labelsVal/P001_1_10.mat');
imagef=fftshift(fft(fft(complex(image.image(:,:,1),image.image(:,:,2)),[],1),[],2));
imagef=imagef((512/2-32):(512/2+31),(256/2-32):(256/2+31),:);
image=ifft(ifft(ifftshift(imagef),[],1),[],2)/32;
image_label(:,:,1)=real(image);
image_label(:,:,2)=imag(image);
mask=load('/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2_zf/maskVal/P001_1_10.mat');
mask=mask.image((512/2-32):(512/2+31),(256/2-32):(256/2+31),1);
tic;generatedImages2 = generateAndDisplayImages2(net,image_input,image_label,mask,numNoiseSteps);toc
 %%
function [Img_clean,Img_degrade] = applyNoiseToImage(img,noiseStep)
    img = extractdata(img);
    Img_clean = dlarray(img(:,:,3:4,:),'SSCB');
    Img_input = dlarray(img(:,:,1:2,:),'SSCB');
    Img_degrade = (1-noiseStep).*Img_clean + noiseStep.*Img_input;
    % noise = dlarray(randn(size(Img_input)),'SSCB')*0.01;
    % Img_degrade = Img_degrade+sqrt(noiseStep).*noise;
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
% noise = randn(size(Img_input))*0.01;
% Img_degrade=Img_degrade+sqrt(1).*noise;

for noiseStep = (numsteps):-1:1
    Img_predict = predict(net,Img_degrade,noiseStep);
    if noiseStep>1
        % Img_degrade = (1-(noiseStep-1)/numsteps).*Img_predict + (noiseStep-1)/numsteps.*Img_degrade;
        % Img_degrade = 1/(noiseStep).*Img_predict + (1-1/(noiseStep)).*Img_degrade;
        Img_degrade = 1/numsteps.*(Img_predict-Img_input) + Img_degrade;
        % noise = randn(size(Img_input))*0.01;
        % Img_degrade=Img_degrade+sqrt(noiseStep/numsteps).*noise;
    end
    loss(numsteps-noiseStep+1)=l2norm(Img_clean-Img_predict);
    % printf(['loss:' num2str(loss)])

    tLay = tiledlayout("flow");
    title(tLay,"t = "+ noiseStep)
    for ii = 1:1
        nexttile
        % imshow(abs(complex(Img_degrade(:,:,1,ii),Img_degrade(:,:,2,ii))),[])
        imagesc(abs(complex(Img_predict(:,:,1,ii),Img_predict(:,:,2,ii))))
    end
    drawnow
end

% Display final images.
% loss=l2norm(Img_clean-Img_predict);
printf(['loss:' num2str(loss)])
loss=l2norm(Img_clean-Img_degrade);
printf(['loss:' num2str(loss)])

tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(abs(complex(Img_degrade(:,:,1,ii),Img_degrade(:,:,2,ii))),[])
end
end

%%
function Img_predict = generateAndDisplayImages2(net,Img_input,Img_clean,mask,numsteps)
Img_degrade=Img_input;
Img_cleanf=fftshift(fft2(complex(Img_clean(:,:,1),Img_clean(:,:,2)))).*mask;
for noiseStep = (numsteps):-1:1
    Img_predict = predict(net,Img_degrade,noiseStep);
    % Img_predictf=fftshift(fft2(complex(Img_predict(:,:,1),Img_predict(:,:,2)))).*(1-mask)+Img_cleanf;
    % Img_predict1=ifft2(ifftshift(Img_predictf));
    % Img_predict(:,:,1)=real(Img_predict1);
    % Img_predict(:,:,2)=imag(Img_predict1);
    if noiseStep>1
        % Img_degrade = (1-(noiseStep-1)/numsteps).*Img_predict + (noiseStep-1)/numsteps.*Img_degrade;
        % Img_degrade = 1/(noiseStep).*Img_predict + (1-1/(noiseStep)).*Img_degrade;
        Img_degrade = 1/numsteps.*(Img_predict-Img_input) + Img_degrade;
        % noise = randn(size(Img_input))*0.01;
        % Img_degrade=Img_degrade+sqrt(noiseStep/numsteps).*noise;
    end
    loss(numsteps-noiseStep+1)=l2norm(Img_clean-Img_predict);
    % printf(['loss:' num2str(loss)])

    tLay = tiledlayout("flow");
    title(tLay,"t = "+ noiseStep)
    for ii = 1:1
        nexttile
        % imshow(abs(complex(Img_degrade(:,:,1,ii),Img_degrade(:,:,2,ii))),[])
        imagesc(abs(complex(Img_predict(:,:,1,ii),Img_predict(:,:,2,ii))))
    end
    drawnow
end

% Display final images.
% loss=l2norm(Img_clean-Img_predict);
printf(['loss:' num2str(loss)])
loss=l2norm(Img_clean-Img_degrade);
printf(['loss:' num2str(loss)])

end

