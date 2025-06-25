% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = false;

dataFolder = "/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/DigitsData";
imds = imageDatastore(dataFolder,'IncludeSubfolders',true);
imgSize = [32 32];
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 4;
numInputChannels = 1;

img = read(audsImds);
img = img{2,1};
img = img{:};
% img = rescale(img,-1,1);
img = rescale(img,-1,1,'InputMin',0,'InputMax',255);
%%
mask=zeros(imgSize);
% mask(:,1:2:end)=1;
idx=randperm(prod(imgSize),prod(imgSize)/4);
for i=1:length(idx)
    mask(floor((idx(i)-1)/imgSize(1))+1,mod(idx(i)-1,imgSize(2))+1)=1;
end
ref=fft(fft(img,[],1),[],2);
Img=img;
%%
figure,
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
    
    % deepNetworkDesigner(net)
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
        generateAndDisplayImages(net,Img,numDownSampSteps,imgSize);
        printf(['loss:' num2str(loss)])
    end
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/resNet2d_mnist.mat',"net");
else
    load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/resNet2d_mnist.mat',"net");
end
%%
figure
tic;generatedImages = generateAndDisplayImages(net,numDownSampSteps,imgSize);toc
%%
figure
tic;generatedImages2 = generateAndDisplayImages2(net,Img,numDownSampSteps);toc
%%
numImages = 1;
displayFrequency = 1;
figure
tic;generatedImages3 = generateAndDisplayImages3(net,Img,mask,numDownSampSteps,imgSize,numImages);toc
%%
function DownSampImg = applyDownSampToImage(img,DownSampFactor)
    img=gather(img);
    DownSampImg=zeros(size(img));
    for i=1:size(img,3)
        for j=1:size(img,4)
            tmp=img(:,:,i,j);
            tmp2=imresize(tmp,size(tmp)/DownSampFactor);
            DownSampImg(:,:,i,j) = imresize(tmp2,size(tmp),"nearest");
            % tmp2=zeros(size(img)/DownSampFactor);
            % for i1=1:size(img,1)/DownSampFactor
            %     for i2=1:(size(img,2)/DownSampFactor)
            %         tmp3=tmp((i1-1)*DownSampFactor+(1:DownSampFactor),(i2-1)*DownSampFactor+(1:DownSampFactor));
            %         minx=min(col(tmp3));
            %         maxx=max(col(tmp3));
            %         if abs(maxx)>=abs(minx)
            %             tmp2(i1,i2,i,j)=maxx;
            %         else
            %             tmp2(i1,i2,i,j)=minx;
            %         end
            %     end
            % end
            % for i1=1:DownSampFactor
            %     for i2=1:DownSampFactor
            %         DownSampImg(i1:DownSampFactor:end,i2:DownSampFactor:end,i,j)=tmp2;
            %     end
            % end
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
X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end
%%
function images = generateAndDisplayImages(net,numDownSampSteps,imageSize)
% Generate random noise.
images = gpuArray(imresize(randn([2 2]),imageSize,'nearest'));
% images = applyDownSampToImage(img,2^numDownSampSteps);
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
function images = generateAndDisplayImages2(net,Img,numDownSampSteps)
% Generate random noise.
images = applyDownSampToImage(Img,2^(numDownSampSteps-0));

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
function images = generateAndDisplayImages3(net,Img,mask,numDownSampSteps,imageSize,numImages)
% Generate random noise.
% images_r = imresize(randn([2 2]),imageSize,'nearest')*0.1-0.8;
% images_i = imresize(randn([2 2]),imageSize,'nearest')*0.1-0.8;
% images = images_r+1i*images_i;

tmp = applyDownSampToImage(Img,2^numDownSampSteps);
images = tmp+1i*tmp;
ref=fft(fft(Img,[],1),[],2);
% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0_r = predict(net,real(images),DownSampStep);
    images0_i = predict(net,imag(images),DownSampStep);
    images0=images0_r+1i*images0_i;

    imagesf=fft(fft(images0,[],1),[],2);
    imagesf=ref.*mask+imagesf.*(1-mask);
    images0=ifft(ifft(imagesf,[],1),[],2);

    DownSampFactor=2^(DownSampStep-1);
    % if DownSampStep~=1
        images = applyDownSampToImage((images0),DownSampFactor);
        % images_r = applyDownSampToImage(real(images0),DownSampFactor);
        % images_i = applyDownSampToImage(imag(images0),DownSampFactor);
        % images = images_r+1i*images_i;
        % images = images-applyDownSampToImage(images0,DownSampFactor*2)+applyDownSampToImage(images0,DownSampFactor);
    % else
    %     images=images0;
    % end

    % Display intermediate images.
    % if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:numImages
            nexttile
            imshow(real(images(:,:,:,ii)),[])
        end
        drawnow
    % end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    imshow(real(images(:,:,:,ii)),[])
end
end

