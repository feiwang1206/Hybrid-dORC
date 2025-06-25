addpath(genpath('/home/ubuntu/Documents/MATLAB/metal_artifact_simulation-master'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath /home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox
setup
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles2'));
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/MRIsimu&recon-keep'));


%%
pathname='/home/ubuntu/Desktop/share/brain/';
dirs=dir(pathname);
dirs=dirs(3:end);
diffpath='/home/ubuntu/Desktop/share/diffusion/';
mkdir(diffpath)
diffpath_net='/home/ubuntu/Desktop/share/diffusion_net/';
mkdir(diffpath_net)
N=64*1;
%%
% nn=1;
% for ii=1:length(dirs)
%     data_ct=niftiread([dirs(ii).folder '/' dirs(ii).name '/ct']);
%     data_mr=niftiread([dirs(ii).folder '/' dirs(ii).name '/mr']);
%     data_ct=data_ct+1000;
%     for jj=1:size(data_mr,3)
%         tmp_ct=mean(mean(data_ct,1),2);
%         tmp_mr=mean(mean(data_mr,1),2);
%         if tmp_ct(jj)>0.5*max(tmp_ct) && tmp_mr(jj)>0.5*max(tmp_mr)
%             image=zeros(N,N);
%             tmp=imresize(data_ct(:,:,jj),[N N]);
%             image(:,:,1)=tmp/max(col(tmp));
%             tmp=imresize(data_mr(:,:,jj),[N N]);
%             image(:,:,2)=tmp/max(col(tmp));
%             filename=[diffpath num2str(nn)];
%             save(filename,'image');
%             nn=nn+1;
%         end
%     end
% end
%%
% unzip("DigitsData.zip");
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = true;

dataFolder = diffpath;
volReader = @(x) matRead_rescale(x,-1,1);
volLoc = dataFolder;
imds = imageDatastore(volLoc, 'FileExtensions','.mat','ReadFcn',volReader);
imgSize = [N N];
audsImds = augmentedImageDatastore(imgSize,imds);
numInputChannels = 2;

numNoiseSteps = 500;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = gpuArray(linspace(betaMin,betaMax,numNoiseSteps));

%%
miniBatchSize = 2;
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
    % net = createDiffusionNetwork2(imgSize,numInputChannels);
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        shuffle(mbq);

        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;

            img = gpuArray(next(mbq));

            % Generate random noise.
            targetNoise = gpuArray(randn(size(img),'Like',img));

            % Generate a random noise step.
            noiseStep = dlarray(gpuArray(randi(numNoiseSteps,[1 miniBatchSize],'Like',img)),"CB");

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
        printf(['loss:' num2str(loss)])
        numImages = 1;
        displayFrequency = numNoiseSteps + 1;
        generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);
        save([diffpath_net 'net.mat'],"net");
    end
else
    load([diffpath_net 'net.mat'],"net");
end
%% ct
img = read(audsImds);
img = img{floor(rand*100)+1,1};
img = img{:};
img=rescale(img,-1,1);
img_ct=double(img(:,:,1))/10;
metal=zeros(size(img_ct));
metal((floor(N/2)-3):(floor(N/2)+3),(floor(N/2)-1):(floor(N/2)+1))=1;
metal((floor(N/4)-1):(floor(N/4)+1),(floor(N/2)-3):(floor(N/2)+3))=1;
img_ct_metal=img_ct+metal*2;
config = set_config_for_artifact_simulation2(20/N,N,N*4);
proj_metal = fanbeam(metal,...  
                  config.SOD,...
                  'FanSensorGeometry','arc',...
                  'FanSensorSpacing', config.angle_size, ...
                  'FanRotationIncrement',360/config.angle_num);
proj_metal(proj_metal<0)=0;
proj_metal(proj_metal>0)=1;
proj = fanbeam(img_ct_metal,...  
                  config.SOD,...
                  'FanSensorGeometry','arc',...
                  'FanSensorSpacing', config.angle_size, ...
                  'FanRotationIncrement',360/config.angle_num);
y=exp(-proj);
scale=1e5;
noisy_y = scale*imnoise(y/scale,'poisson');
proj = -log(noisy_y); 
sim = ifanbeam(proj,...
               config.SOD,...
               'FanSensorGeometry','arc',...
               'FanSensorSpacing',config.angle_size,...
               'OutputSize',config.output_size,... 
               'FanRotationIncrement',360/config.angle_num,...
               'Filter', config.filter_name,...
               'FrequencyScaling', config.freqscale);
figure(10),subplot(1,2,1),imagesc(real(proj));colorbar;colormap gray;axis equal;title(num2str(0));
subplot(1,2,2),imagesc(real(sim),[min(col(img_ct)) max(col(img_ct))]);colorbar;colormap gray;axis equal;title(num2str(0));

param.config=config;
param.ref_ct=proj.*(1-proj_metal);
param.ref_ct(isnan(param.ref_ct))=0;
param.proj_metal=proj_metal;
%% mr
metal3=zeros(N,N,N);
metal3(:,:,floor(N/2))=metal;
fov=[200 200 200];
reso=fov/N;
px=repmat((-N/2:(N/2-1))',[1 N N])*reso(1);
py=repmat(-N/2:(N/2-1),[N 1 N])*reso(2);
pz=repmat(permute((-N/2:(N/2-1))',[3 2 1]),[N N 1])*reso(3);
pp=pz.^2./(px.^2+py.^2+pz.^2);
kai=1.8*10^-3;
B_temp=fftshift((1/3-pp).*fftshift(fftn(kai*metal3)));
B_temp(isnan(B_temp))=0;
w=3*2*pi*42.58*10^6*ifftn(B_temp);
slice=N/2;
w2d=w(:,:,slice);
% %%
dt=5e-6;
traj=cell(N/1,1);
scale_field=4;
nn=1;
for ii=1:1:N
    traj{nn}(:,1)=(-N/2:N/2-1)/N*2*pi; 
    traj{nn}(:,2)=(ii-N/2-1)/N*2*pi;
    traj2{nn}(:,1)=(-N/2:N/2-1)/N*2*pi/scale_field; 
    traj2{nn}(:,2)=(ii-N/2-1)/N*2*pi/scale_field;
    T{nn}=dt*(1:N);
    nn=nn+1;
end
img_mr=double(img(:,:,2)).*(1-metal);
seg=3;
F=orc_segm_nuFTOperator_multi_sub(traj2,[N N]*scale_field,ones(N*scale_field,N*scale_field),imresize(w2d,[N N]*scale_field),dt,seg,T);
rawdata=F*imresize(img_mr,[N N]*scale_field)/scale_field;
F=orc_segm_nuFTOperator_multi_sub(traj,[N N],ones(N,N),w2d,dt,seg,T);
recon=imresize(gather(regularizedReconstruction(F,rawdata,'maxit',50,'verbose_flag', 1,'tol',1e-5,'z0',[])),size(img_mr));
figure(11),subplot(1,2,1),imagesc(real(recon));colorbar;colormap gray;axis equal;title(num2str(0));
subplot(1,2,2),imagesc(real(img_mr));colorbar;colormap gray;axis equal;title(num2str(0));
drawnow

param.F=F;
param.rawdata=rawdata;
param.img=img;
param.metal=metal;
param.mask=0*w2d;
param.mask(abs(w2d)>0.01*max(col(abs(w2d))))=1;
param.recon_mr=recon;

%%
numImages = 1;
displayFrequency = 10;
figure
tic;generatedImages = generateAndDisplayImages(net,varianceSchedule,imgSize,numImages,numInputChannels,displayFrequency);toc
%%
numImages = 1;
displayFrequency = 10;
figure(12)
tic;generatedImages2 = generateAndDisplayImages2(net,param,varianceSchedule,imgSize,displayFrequency);toc


%%
function images = generateAndDisplayImages2(net,param,varianceSchedule,imageSize,displayFrequency)
numChannels=2;
% Generate random noise.
images = (randn([imageSize numChannels]));

% Compute variance schedule parameters.
alphaBar = cumprod(1 - varianceSchedule);
alphaBarPrev = [1 alphaBar(1:end-1)];
posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);

% Reverse the diffusion process.
numNoiseSteps = length(varianceSchedule);

for noiseStep = numNoiseSteps:-1:1
    if noiseStep ~= 1
        z = randn([imageSize,numChannels]);
    else
        z = zeros([imageSize,numChannels]);
    end

    % Predict the noise using the network.
    predictedNoise = predict(net,images,noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));

    images = images - predNoise;
    % images_ct=abs(gather((images(:,:,1))))/10;
    % proj = fanbeam(images_ct,...  
    %                   param.config.SOD,...
    %                   'FanSensorGeometry','arc',...
    %                   'FanSensorSpacing', param.config.angle_size, ...
    %                   'FanRotationIncrement',360/param.config.angle_num);
    % proj1=param.ref_ct+proj.*param.proj_metal;
    % recon_ct = ifanbeam(proj1,...
    %                param.config.SOD,...
    %                'FanSensorGeometry','arc',...
    %                'FanSensorSpacing',param.config.angle_size,...
    %                'OutputSize',param.config.output_size,... 
    %                'FanRotationIncrement',360/param.config.angle_num,...
    %                'Filter', param.config.filter_name,...
    %                'FrequencyScaling', param.config.freqscale);
    % images(:,:,1)=real(recon_ct)*10;
    % 
    % images_mr=images(:,:,2);
    % recon_mr=gather(regularizedReconstruction(param.F,param.rawdata,'maxit',5,'verbose_flag', 0,...
    %     'tol',1e-5,'z0',images_mr));
    % 
    % images(:,:,2)=real(recon_mr);
    % recons(:,:,1)=real(recon_ct)*10;
    % recons(:,:,2)=real(recon_mr);
    
    images(:,:,2)=real(param.recon_mr).*(1-param.mask)+images(:,:,2).*param.mask;
    images(:,:,1)=param.img(:,:,1);
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;

    % Display intermediate images.
    if mod(noiseStep,displayFrequency) == 0
        subplot(1,2,1)
        imagesc(array2mosaic(images));colorbar;colormap gray;axis equal;
        title(num2str(noiseStep));
        subplot(1,2,2)
        imagesc(array2mosaic(param.img));colorbar;colormap gray;axis equal;
        title(num2str(noiseStep));
        drawnow
    end
end
subplot(1,2,1)
imagesc(array2mosaic(images));colorbar;colormap gray;axis equal;
title(num2str(noiseStep));
subplot(1,2,2)
imagesc(array2mosaic(param.img));colorbar;colormap gray;axis equal;
title(num2str(noiseStep));
end
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
            subplot(1,2,1),imshow(images(:,:,1),[]);
            subplot(1,2,2),imshow(images(:,:,2),[]);
        end
        drawnow
    end
end

% Display final images.
tLay = tiledlayout("flow");
title(tLay,"t = 0")
for ii = 1:numImages
    nexttile
    subplot(1,2,1),imshow(images(:,:,1),[]);
    subplot(1,2,2),imshow(images(:,:,2),[]);
end
end