% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));

dataFolder = "/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image_abs_diff/labelsTra";
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [64 64];
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 5;

img = read(audsImds);
img = img{1,1};
img = img{:};
img = rescale(img,-1,1);
%%
mask=zeros(imgSize);
% mask(:,1:2:end)=1;
idx=randperm(prod(imgSize),prod(imgSize)/2);
for i=1:length(idx)
    mask(floor((idx(i)-1)/imgSize(1))+1,mod(idx(i)-1,imgSize(2))+1)=1;
end
ref=fft(fft(img,[],1),[],2).*mask;
Img=img;
%%
tiledlayout("flow");
nexttile
imshow(img,[])
title("t = 0");
for i = 1:numDownSampSteps
    nexttile
    DownSampStepsToApply = 2^(i);
    DownSampImg = applyDownSampToImage(img,DownSampStepsToApply);

    % Extract the data from the dlarray.
    imshow(DownSampImg,[])
    title("t = " + string(DownSampStepsToApply));
end
%%
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
        generateAndDisplayImages(net,numDownSampSteps,imgSize);
    end
end
save(['/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/resNet2d_mreg' ...
    '.mat'],"net");
%%
numImages = 1;
figure
tic;generatedImages = generateAndDisplayImages(net,numDownSampSteps,imgSize);toc
%%
pname_s='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13/';
pname_s1='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image/';
param=dc(pname_s,pname_s1,1,1);
figure
tic;generatedImages2 = generateAndDisplayImages2(net,param,5,[64 64],5);toc
%%
numImages = 1;
displayFrequency = 1;
figure
tic;generatedImages3 = generateAndDisplayImages3(net,ref,mask,numDownSampSteps,imgSize,numImages,displayFrequency);toc
%%
function DownSampImg = applyDownSampToImage(img,DownSampFactor)
    img=gather(img);
    DownSampImg=zeros(size(img));
    for i=1:size(img,3)
        for j=1:size(img,4)
            tmp=img(:,:,i,j);
            DownSampImg(:,:,i,j) = imresize(imresize(tmp,size(tmp)/DownSampFactor),size(tmp),"nearest");
        end
    end
end
%%
function DownSampImg = applyDownSampToImage_dl(img,DownSampFactor)
    img=gather(extractdata(img));
    DownSampFactor=gather(extractdata(DownSampFactor));
    for i=1:size(img,3)
        for j=1:size(img,4)
            tmp=img(:,:,i,j);
            DownSampImg(:,:,i,j) = imresize(imresize(tmp,size(tmp)/DownSampFactor(j)),size(tmp),"nearest");
        end
    end
    DownSampImg=dlarray(DownSampImg,'SSCB');
end
%%
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
%%
function images = generateAndDisplayImages(net,numDownSampSteps,imageSize)
% Generate random noise.
images = gpuArray(imresize(randn([2 2]),imageSize,'nearest'));

% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep);
    DownSampFactor=2^(DownSampStep-1);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampFactor);
        % images = images-applyDownSampToImage(images0,DownSampFactor*2)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end
    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:1
            nexttile
            imshow(images(:,:,:,ii),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(images(:,:,:,ii),[])
end
end
%%
function images3d = generateAndDisplayImages2(net,param,numDownSampSteps,imageSize,maxit)
for slice=1:param.dim(3)
    images3d(:,:,slice) = gpuArray(imresize(randn([2 2])+1i*randn([2 2]),imageSize,'nearest'));
end
% Reverse the diffusion process.
for DownSampStep = numDownSampSteps:-1:1
    for slice=1:param.dim(3)
        % Generate random noise.
    
        % Predict the noise using the network.
        images0_r = predict(net,real(images3d(:,:,slice)),DownSampStep);
        images0_i = predict(net,imag(images3d(:,:,slice)),DownSampStep);
        images3d0(:,:,slice)=images0_r+1i*images0_i;
    end
    images3d0 = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',maxit,'verbose_flag', 0,'tol',1e-5,'z0',images3d0);
    DownSampFactor=2^(DownSampStep-1);
    for slice=1:param.dim(3)
        if DownSampStep~=1
            images3d_r = applyDownSampToImage(real(images3d0(:,:,slice)),DownSampFactor);
            images3d_i = applyDownSampToImage(imag(images3d0(:,:,slice)),DownSampFactor);
            images3d(:,:,slice)=images3d_r+1i*images3d_i;
            % images = images-applyDownSampToImage(images0,DownSampStep)+applyDownSampToImage(images0,DownSampStep-1);
        else
            images3d(:,:,slice)=images3d0(:,:,slice);
        end

    end

    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:1
            nexttile
            imagesc(array2mosaic(abs(images3d)));
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imagesc(array2mosaic(abs(images3d)));
end
end
%%
function images = generateAndDisplayImages3(net,ref,mask,numDownSampSteps,imageSize,numImages,displayFrequency)
% Generate random noise.
images = gpuArray(imresize(randn([2 2])+1i*randn([2 2]),imageSize,'nearest'));

% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0_r = predict(net,real(images),DownSampStep);
    images0_i = predict(net,imag(images),DownSampStep);
    images0=images0_r+1i*images0_i;

    imagesf=fft(fft(images0,[],1),[],2);
    imagesf=ref.*mask+imagesf.*(1-mask);
    images0=ifft(ifft(imagesf,[],1),[],2);

    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampStep-1);
        % images = images-applyDownSampToImage(images0,DownSampStep)+applyDownSampToImage(images0,DownSampStep-1);
    else
        images=images0;
    end

    % Display intermediate images.
    % if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:numImages
            nexttile
            imshow(abs(images(:,:,:,ii)),[])
        end
        drawnow
    % end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(abs(images(:,:,:,ii)),[])
end
end

