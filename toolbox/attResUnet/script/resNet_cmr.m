% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = false;
numInputChannels = 2;

dataFolder = "/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2/labelsTra";
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [512 512]/8;
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 5;

%%
img = read(audsImds);
img = img{1,1};
img = img{:};
% img = rescale(img,-1,1);
mask=zeros(imgSize);
% mask(:,1:2:end)=1;
% idx=randperm(imgSize(1),floor(imgSize(1)/4));
idx=mod(floor(randn(floor(imgSize(1)/4),1)*imgSize(1)/4+imgSize(1)/2),imgSize(1))+1;
for i=1:length(idx)
    mask(idx,:)=1;
end
mask(imgSize(1)/2+1,:)=1;
Img=img;
% %%
figure,
tiledlayout("flow");
nexttile
imshow(abs(complex(img(:,:,1),img(:,:,2))),[])
title("t = 0");
for i = 1:numDownSampSteps
    nexttile
    DownSampStepsToApply = 2^(i);
    DownSampImg = applyDownSampToImage(img,DownSampStepsToApply);

    % Extract the data from the dlarray.
    imshow(abs(complex(DownSampImg(:,:,1),DownSampImg(:,:,2))),[])
    title("t = " + string(DownSampStepsToApply));
end

% deepNetworkDesigner(net)
%%
miniBatchSize = 32;
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
    % net = createDiffusionNetwork(numInputChannels);
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
            [loss,gradients] = dlfeval(@modelLoss,net,DownSampImage,DownSampFactor,img-DownSampImage);

            % Update model.
            [net,averageGrad,averageSqGrad] = adamupdate(net,gradients,averageGrad,averageSqGrad,iteration, ...
                learnRate,gradientDecayFactor,squaredGradientDecayFactor);

            % Record metrics.
            recordMetrics(monitor,iteration,'Loss',loss);
            updateInfo(monitor,'Epoch',epoch,'Iteration',iteration);
            monitor.Progress = 100 * iteration/numIterations;
        end
        printf(['loss:' num2str(loss)])

        % Generate and display a batch of generated images.
        generateAndDisplayImages(net,Img,numDownSampSteps);
    end
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/resNet2d_cmr.mat',"net");
else
    load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/resNet2d_cmr.mat',"net");
end
%%
figure
tic;generatedImages = generateAndDisplayImages(net,Img,numDownSampSteps);toc
%%
numImages = 1;
displayFrequency = 1;
figure
tic;generatedImages3 = generateAndDisplayImages3(net,Img,mask,numDownSampSteps,imgSize,numImages,displayFrequency);toc
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
function images = generateAndDisplayImages(net,Img,numDownSampSteps)
% Generate random noise.
% images = gpuArray(imresize(randn([2 2]),size(Img),'nearest'));
images = applyDownSampToImage(Img,2^numDownSampSteps);
% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep)+images;
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
            imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
end
end
%%
function images = generateAndDisplayImages3(net,Img,mask,numDownSampSteps,imageSize,numImages,displayFrequency)
% Generate random noise.
% images = gpuArray(imresize(randn([2 2])+1i*randn([2 2]),imageSize,'nearest'));
images = applyDownSampToImage(Img,2^numDownSampSteps);
ref=fftshift(fft(fft(complex(Img(:,:,1),Img(:,:,2)),[],1),[],2));
% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep)+images;

    images0f=fftshift(fft(fft(complex(images0(:,:,1),images0(:,:,2)),[],1),[],2));
    images0f=ref.*mask+images0f.*(1-mask);
    tmp=ifft(ifft(fftshift(images0f),[],1),[],2);
    images0(:,:,1)=real(tmp);
    images0(:,:,2)=imag(tmp);

    DownSampFactor=2^(DownSampStep-1);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampFactor);
        % images = images-applyDownSampToImage(images0,DownSampFactor*2)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end

    % Display intermediate images.
    % if mod(noiseStep,displayFrequency) == 0
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:1
            nexttile
            imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
        end
        drawnow
    % end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(abs(complex(images(:,:,1,ii),images(:,:,2,ii))),[])
end
end

