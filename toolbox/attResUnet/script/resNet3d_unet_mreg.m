% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));



scale=2;
dataFolder = ['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image_abs_diff3d' num2str(scale) '/labelsTra'];
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [64 64 48]/scale;
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 3;

img = read(audsImds);
img = img{1,1};
img = img{:};
img = rescale(img,-1,1);
Img = img;
%%
tiledlayout("flow");
nexttile
imshow(img(:,:,imgSize(3)/2),[])
title("t = 0");
for i = 1:numDownSampSteps
    nexttile
    DownSampStepsToApply = 2^(i);
    DownSampImg = applyDownSampToImage(img,DownSampStepsToApply);

    % Extract the data from the dlarray.
    imshow(DownSampImg(:,:,imgSize(3)/2),[])
    title("t = " + string(DownSampStepsToApply));
end
%%
numInputChannels = 1;
net = unet3dLayers_recon_dlnetwork(imgSize, numInputChannels,'EncoderDepth',3);
% net = createDiffusionNetwork3d(imgSize,numInputChannels);
% deepNetworkDesigner(net)
%%
miniBatchSize = 5;
numEpochs = 50;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

doTraining = true;
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

if doTraining
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            img = next(mbq);

            % Generate a random noise step.
            DownSampFactor = dlarray(randi(numDownSampSteps,[1 miniBatchSize],'Like',img),"CB");

            % Apply noise to the image.
            DownSampImage = applyDownSampToImage_dl(img,2.^DownSampFactor);

            % Compute loss.
            [loss,gradients] = dlfeval(@modelLoss,net,DownSampImage,DownSampFactor,img);

            % Update model.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, ...
                learnRate,gradientDecayFactor,squaredGradientDecayFactor);

            % Record metrics.
            recordMetrics(monitor,iteration,'Loss',loss);
            updateInfo(monitor,'Epoch',epoch,'Iteration',iteration);
            monitor.Progress = 100 * iteration/numIterations;
        end

        % Generate and display a batch of generated images.
        numImages = 1;
        displayFrequency = numDownSampSteps + 1;
        generateAndDisplayImages(net,numDownSampSteps,imgSize,numImages,numInputChannels,displayFrequency);
    end
end
save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/resnet3d_unet.mat',"net");
%%
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages = generateAndDisplayImages(net,numDownSampSteps,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
pname_s='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13/';
pname_s1='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image/';
param=dc(pname_s,pname_s1,1,scale);
figure
tic;generatedImages2 = generateAndDisplayImages2(net,param,numDownSampSteps,imgSize,numImages,displayFrequency);toc
%%
function DownSampImg = applyDownSampToImage(img,DownSampFactor)
    img=gather(img);
    DownSampImg=zeros(size(img));
    for i=1:size(img,4)
        for j=1:size(img,5)
            tmp=img(:,:,:,i,j);
            tmp=imresize3D(tmp,size(tmp)/DownSampFactor(j));
            for i1=1:DownSampFactor(j)
                for i2=1:DownSampFactor(j)
                    for i3=1:1:DownSampFactor(j)
                        DownSampImg(i1:DownSampFactor(j):end,i2:DownSampFactor(j):end,i3:DownSampFactor(j):end,i,j) = tmp;
                    end
                end
            end
        end
    end
end
%%
function DownSampImg = applyDownSampToImage_dl(img,DownSampFactor)
    img=gather(extractdata(img));
    DownSampFactor=gather(extractdata(DownSampFactor));
    DownSampImg=zeros(size(img));
    for i=1:size(img,4)
        for j=1:size(img,5)
            tmp=img(:,:,:,i,j);
            tmp=imresize3D(tmp,size(tmp)/DownSampFactor(j));
            for i1=1:DownSampFactor(j)
                for i2=1:DownSampFactor(j)
                    for i3=1:1:DownSampFactor(j)
                        DownSampImg(i1:DownSampFactor(j):end,i2:DownSampFactor(j):end,i3:DownSampFactor(j):end,i,j) = tmp;
                    end
                end
            end
        end
    end
    DownSampImg=dlarray(DownSampImg,'SSSCB');
end
%%
function [loss, gradients] = modelLoss(net,X,Y,T)
% Forward data through the network.
noisePrediction = forward(net,X);

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
function images = generateAndDisplayImages(net,numDownSampSteps,imageSize,numImages,numChannels,displayFrequency)
% Generate random noise.
images = gpuArray(imresize3D(randn([4 4 3]),imageSize,'nearest'));

% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampStep-1);
        % images = images-applyDownSampToImage(images0,DownSampStep)+applyDownSampToImage(images0,DownSampStep-1);
    else
        images=images0;
    end
    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:numImages
            nexttile
            imshow(images(:,:,imageSize(3)/2,ii),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(images(:,:,imageSize(3)/2,ii),[])
end
end

%%
function images = generateAndDisplayImages2(net,param,numDownSampSteps,imageSize,numImages,displayFrequency)
% Generate random noise.
DownSampImg = randn([4 4 3])+1i*randn([4 4 3]);
images=zeros(imageSize);
DownSampFactor=2^numDownSampSteps;
for i1=1:DownSampFactor
    for i2=1:DownSampFactor
        for i3=1:1:DownSampFactor
            images(i1:DownSampFactor:end,i2:DownSampFactor:end,i3:DownSampFactor:end) = DownSampImg;
        end
    end
end

for DownSampStep = numDownSampSteps:-1:1
    images_r = predict(net,real(images));
    images_i = predict(net,imag(images));
    images0=images_r+1i*images_i;
    images0 = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',1,'verbose_flag', 0,'tol',1e-5,'z0',images0);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampStep-1);
        % images = images-applyDownSampToImage(images0,DownSampStep)+applyDownSampToImage(images0,DownSampStep-1);
    else
        images=gather(images0);
    end

    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:numImages
            nexttile
            imshow(abs(images(:,:,imageSize(3)/2,ii)),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(images(:,:,imageSize(3)/2,ii)),[])
end
end
