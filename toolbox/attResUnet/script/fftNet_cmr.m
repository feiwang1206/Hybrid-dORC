% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = false;

dataFolder = "/media/ubuntu/data/SingleCoil/Cine/TrainingSet/AccFactor04/2/labelsTra";
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [512 512]/8;
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 32;
numInputChannels = 2;

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
ref=fft(fft(img,[],1),[],2).*mask;
Img=img;
% %%
figure,
tiledlayout("flow");
nexttile
imshow(abs(complex(img(:,:,1),img(:,:,2))),[])
title("t = 0");
for i = 4:4:numDownSampSteps
    nexttile
    DownSampStepsToApply = i;
    DownSampImg = applyDownSampToImage(img,DownSampStepsToApply);

    % Extract the data from the dlarray.
    imshow(abs(complex(DownSampImg(:,:,1),DownSampImg(:,:,2))),[])
    title("t = " + string(DownSampStepsToApply));
end
%%
miniBatchSize = 32;
numEpochs = 200;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;
% gradientDecayFactor = 0.99;
% squaredGradientDecayFactor = 0.9999;

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
            DownSampImage = applyDownSampToImage_dl(img,DownSampFactor);

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
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/fftNet2d_cmr.mat',"net");
else
    load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/fftNet2d_cmr.mat',"net");
end
%%
figure
tic;generatedImages = generateAndDisplayImages(net,Img,numDownSampSteps);toc
%%
numImages = 1;
displayFrequency = 1;
figure
tic;generatedImages3 = generateAndDisplayImages3(net,Img,mask,numDownSampSteps);toc
%%
function DownSampImg = applyDownSampToImage(img,DownSampFactor)
    ss=size(img);
    DownSampImg=zeros(ss);
    if length(ss)==2
        ss(3:4)=1;
    end
    if length(ss)==3
        ss(4)=1;
    end
    for i=1:1
        for j=1:ss(4)
            % imgf=fftshift(fft(fft(img(:,:,i,j),[],1),[],2));
            imgf=fftshift(fft(fft(complex(img(:,:,1,j),img(:,:,2,j)),[],1),[],2));
            mask=zeros(ss(1:2));
            mask((DownSampFactor(j)):(end-DownSampFactor(j)+1),(DownSampFactor(j)):(end-DownSampFactor(j)+1))=1;
            imgf=imgf.*mask;
            img2=ifft(ifft(fftshift(imgf),[],1),[],2);
            DownSampImg(:,:,1,j)=real(img2);
            DownSampImg(:,:,2,j)=imag(img2);
        end
    end
end
%%
function DownSampImg = applyDownSampToImage_dl(img,DownSampFactor)
    img=gather(extractdata(img));
    DownSampFactor=gather(extractdata(DownSampFactor));
    ss=size(img);
    DownSampImg=zeros(ss);
    for i=1:1
        for j=1:ss(4)
            % imgf=fftshift(fft(fft(img(:,:,i,j),[],1),[],2));
            imgf=fftshift(fft(fft(complex(img(:,:,1,j),img(:,:,2,j)),[],1),[],2));
            mask=zeros(ss(1:2));
            mask((DownSampFactor(j)):(end-DownSampFactor(j)+1),(DownSampFactor(j)):(end-DownSampFactor(j)+1))=1;
            imgf=imgf.*mask;
            img2=ifft(ifft(fftshift(imgf),[],1),[],2);
            DownSampImg(:,:,1,j)=real(img2);
            DownSampImg(:,:,2,j)=imag(img2);
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
% X = rescale(X,-1,1);
end
%%
function images = generateAndDisplayImages(net,Img,numDownSampSteps)
% Generate random noise.
% images = gpuArray(imresize(randn([2 2]),size(Img),'nearest'));
images = applyDownSampToImage(Img,numDownSampSteps);
% Reverse the diffusion process.
imagesf = fftshift(fft(fft(complex(images(:,:,1),images(:,:,2)),[],1),[],2));
% mask=zeros(size(imagesf));
% mask(abs(imagesf)>1e-5)=1;
for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep)+images;
    % images0f = fftshift(fft(fft(complex(images0(:,:,1),images0(:,:,2)),[],1),[],2));
    % tmp = ifft(ifft(fftshift(imagesf.*mask+images0f.*(1-mask)),[],1),[],2);
    % images0(:,:,1)=real(tmp);
    % images0(:,:,2)=imag(tmp);

    DownSampFactor=(DownSampStep-1);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampFactor);
        % images = images-applyDownSampToImage(images0,DownSampFactor+1)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end
    imagesf = fftshift(fft(fft(complex(images(:,:,1),images(:,:,2)),[],1),[],2));
    mask=zeros(size(imagesf));
    mask(abs(imagesf)>1e-5)=1;
    % Display intermediate images.
    tLay = tiledlayout("flow");
    title(tLay,"t = "+ DownSampStep)
    nexttile
    imshow(abs(complex(images(:,:,1),images(:,:,2))),[]);colorbar
    drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
nexttile
imshow(abs(complex(images(:,:,1),images(:,:,2))),[]);colorbar
end
%%
function images = generateAndDisplayImages3(net,Img,mask0,numDownSampSteps)
% Generate random noise.
% images = gpuArray(imresize(randn([2 2])+1i*randn([2 2]),imageSize,'nearest'));
images = applyDownSampToImage(Img,numDownSampSteps);
% imagesf = fftshift(fft(fft(complex(images(:,:,1),images(:,:,2)),[],1),[],2));
% mask=zeros(size(imagesf));
% mask(abs(imagesf)>1e-5)=1;

ref=fftshift(fft(fft(complex(Img(:,:,1),Img(:,:,2)),[],1),[],2));
% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep)+images;
    images0f = fftshift(fft(fft(complex(images0(:,:,1),images0(:,:,2)),[],1),[],2));
    % tmp = ifft(ifft(fftshift(imagesf.*mask+images0f.*(1-mask)),[],1),[],2);
    % images0(:,:,1)=real(tmp);
    % images0(:,:,2)=imag(tmp);

    imagesf=ref.*mask0+images0f.*(1-mask0);
    tmp=ifft(ifft(fftshift(imagesf),[],1),[],2);
    images0(:,:,1)=real(tmp);
    images0(:,:,2)=imag(tmp);

    DownSampFactor=(DownSampStep-1);
    if DownSampStep~=1
        % images = applyDownSampToImage(images0,DownSampFactor);
        images = images-applyDownSampToImage(images0,DownSampFactor+1)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end
    % imagesf = fftshift(fft(fft(complex(images(:,:,1),images(:,:,2)),[],1),[],2));
    % mask=zeros(size(imagesf));
    % mask(abs(imagesf)>1e-5)=1;

    % Display intermediate images.
    % if mod(noiseStep,displayFrequency) == 0
    tLay = tiledlayout("flow");
    title(tLay,"t = "+ DownSampStep)
    nexttile
    imshow(abs(complex(images(:,:,1),images(:,:,2))),[]);colorbar
        drawnow
    % end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
nexttile
imshow(abs(complex(images(:,:,1),images(:,:,2))),[]);colorbar
end

