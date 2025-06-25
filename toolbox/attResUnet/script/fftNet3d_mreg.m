% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath /home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox
setup


doTraining = true;

scale=2;
dataFolder = ['/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image_abs_diff3d' num2str(scale) '/labelsTra'];
volReader = @(x) matRead(x);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [64 64 48]/scale;
audsImds = augmentedImageDatastore(imgSize,imds);

numDownSampSteps = 16;

img = read(audsImds);
img = img{1,1};
img = img{:};
% img = rescale(img,-1,1);
Img = img;
%%
tiledlayout("flow");
nexttile
imshow(img(:,:,imgSize(3)/2),[])
title("t = 0");
for i = 2:2:numDownSampSteps
    nexttile
    DownSampStepsToApply = (i);
    DownSampImg = applyDownSampToImage(img,DownSampStepsToApply);

    % Extract the data from the dlarray.
    imshow(DownSampImg(:,:,imgSize(3)/2),[])
    title("t = " + string(DownSampStepsToApply));
end
%%
numInputChannels = 1;
%%
miniBatchSize = 16;
numEpochs = 50;

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

if doTraining
    % net = unet3dLayers_recon_dlnetwork(imgSize, numInputChannels,'EncoderDepth',3);
    % net = createDiffusionNetwork3d(imgSize,numInputChannels);
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

        printf(['loss:' num2str(loss)])
        % Generate and display a batch of generated images.
        generateAndDisplayImages(net,Img,numDownSampSteps,imgSize);
    end
    save('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/fftnet3d32_mreg.mat',"net");
else
    load('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/fftnet3d32_mreg.mat',"net");
end
%%
% numDownSampSteps=3;
imgSize=[32 32 24];
figure
tic;generatedImages = generateAndDisplayImages(net,Img,numDownSampSteps,imgSize);toc
%%
numImages = 1;
scale=2;
displayFrequency = 10;
% numDownSampSteps=3;
imgSize=[32 32 24];
pname_s='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13/';
pname_s1='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista13_image/';
param=dc(pname_s,pname_s1,1,scale);
figure
tic;generatedImages2 = generateAndDisplayImages2(net,Img,param,numDownSampSteps,imgSize,5);toc
%%
function DownSampImg = applyDownSampToImage(img,DownSampFactor)
    ss=size(img);
    DownSampImg=zeros(ss);
    if length(ss)==3
        ss(4:5)=1;
    end
    if length(ss)==4
        ss(5)=1;
    end
    for i=1:ss(4)
        for j=1:ss(5)
            imgf=fftshift(fft(fft(fft(img(:,:,:,i,j),[],1),[],2),[],3));
            mask=zeros(ss(1:3));
            range{1}=(DownSampFactor(j)):(ss(1)-DownSampFactor(j)+1);
            range{2}=(DownSampFactor(j)):(ss(2)-DownSampFactor(j)+1);
            range{3}=min(DownSampFactor(j),ss(3)/2):max(ss(3)-DownSampFactor(j)+1,ss(3)/2);
            mask(range{1},range{2},range{3})=1;
            imgf=imgf.*mask;
            DownSampImg(:,:,:,i,j)=real(ifft(ifft(ifft(fftshift(imgf),[],1),[],2),[],3));
        end
    end
end
%%
function DownSampImg = applyDownSampToImage_dl(img,DownSampFactor)
    img=gather(extractdata(img));
    DownSampFactor=gather(extractdata(DownSampFactor));
    ss=size(img);
    DownSampImg=zeros(ss);
    if length(ss)==3
        ss(4:5)=1;
    end
    if length(ss)==4
        ss(5)=1;
    end
    for i=1:ss(4)
        for j=1:ss(5)
            imgf=fftshift(fft(fft(fft(img(:,:,:,i,j),[],1),[],2),[],3));
            mask=zeros(ss(1:3));
            range{1}=(DownSampFactor(j)):(ss(1)-DownSampFactor(j)+1);
            range{2}=(DownSampFactor(j)):(ss(2)-DownSampFactor(j)+1);
            range{3}=min(DownSampFactor(j),ss(3)/2):max(ss(3)-DownSampFactor(j)+1,ss(3)/2);
            mask(range{1},range{2},range{3})=1;
            imgf=imgf.*mask;
            DownSampImg(:,:,:,i,j)=real(ifft(ifft(ifft(fftshift(imgf),[],1),[],2),[],3));
        end
    end
    DownSampImg=dlarray(DownSampImg,'SSSCB');
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
X = cat(5,data{:});

% Rescale the images so that the pixel values are in the range [-1 1].
% X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end
%%
function images = generateAndDisplayImages(net,Img,numDownSampSteps,imageSize)
% Generate random noise.
% DownSampImg = randn([4 4 3]);
% images=zeros(imageSize);
% DownSampFactor=2^numDownSampSteps;
% for i1=1:DownSampFactor
%     for i2=1:DownSampFactor
%         for i3=1:1:DownSampFactor
%             images(i1:DownSampFactor:end,i2:DownSampFactor:end,i3:DownSampFactor:end) = DownSampImg;
%         end
%     end
% end
images=applyDownSampToImage(Img,numDownSampSteps);
% Reverse the diffusion process.

for DownSampStep = numDownSampSteps:-1:1
    % Predict the noise using the network.
    images0 = predict(net,images,DownSampStep);
    DownSampFactor=(DownSampStep-1);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampFactor);
        % images = images-applyDownSampToImage(images0,DownSampFactor+1)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end
    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:1
            nexttile
            imshow(images(:,:,imageSize(3)/2,ii),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(images(:,:,imageSize(3)/2,ii),[])
end
end

%%
function images = generateAndDisplayImages2(net,Img,param,numDownSampSteps,imageSize,maxit)
% Generate random noise.
% DownSampImg = randn([4 4 3])+1i*randn([4 4 3]);
% images=zeros(imageSize);
% DownSampFactor=2^numDownSampSteps;
% for i1=1:DownSampFactor
%     for i2=1:DownSampFactor
%         for i3=1:1:DownSampFactor
%             images(i1:DownSampFactor:end,i2:DownSampFactor:end,i3:DownSampFactor:end) = DownSampImg;
%         end
%     end
% end
images=applyDownSampToImage(Img,numDownSampSteps)+1i*applyDownSampToImage(Img,numDownSampSteps);

for DownSampStep = numDownSampSteps:-1:1
    images0_r = predict(net,real(images),DownSampStep);
    images0_i = predict(net,imag(images),DownSampStep);
    images0=images0_r+1i*images0_i;
    images0 = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',maxit,'verbose_flag', 0,'tol',1e-5,'z0',images0);
    DownSampFactor=(DownSampStep-1);
    if DownSampStep~=1
        images = applyDownSampToImage(images0,DownSampFactor);
        % images = images-applyDownSampToImage(images0,DownSampFactor+1)+applyDownSampToImage(images0,DownSampFactor);
    else
        images=images0;
    end

    % Display intermediate images.
        tLay = tiledlayout("flow");
        title(tLay,"t = "+ DownSampStep)
        for ii = 1:1
            nexttile
            imshow(abs(images(:,:,imageSize(3)/2,ii)),[])
        end
        drawnow
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:1
    nexttile
    imshow(abs(images(:,:,imageSize(3)/2,ii)),[])
end
end
