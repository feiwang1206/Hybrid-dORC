addpath(genpath('/home/ubuntu/Documents/MATLAB/metal_artifact_simulation-master'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath /home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox
setup


%%
diffpath='/home/ubuntu/Desktop/share/diffusion/';
mkdir(diffpath)
N=32;
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));

doTraining = false;
dataFolder = "DigitsData";
imds = imageDatastore(dataFolder,'IncludeSubfolders',true);
imgSize = [32 32];
audsImds = augmentedImageDatastore(imgSize,imds);

numInputChannels = 1;

numNoiseSteps = 500;
betaMin = 1e-4;
betaMax = 0.02;
varianceSchedule = linspace(betaMin,betaMax,numNoiseSteps);
%%
img = read(audsImds);
img = img{floor(rand*100)+1,1};
img = img{:};
img = rescale(img,-1,1);
%%
metal3=zeros(N,N,N);
metal3((N/2-2):(N/2+2),(N/2-2):(N/2+2),(N/2-1):(N/2+1))=1;
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
slice=N/2+2;
metal=metal3(:,:,slice);
w2d=w(:,:,slice)*1;
% %%
dt=5e-6;
traj=cell(N/1,1);
traj2=cell(N/1,1);
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
img_mr=double(img+1);%.*(1-metal);
seg=3;
F=orc_segm_nuFTOperator_multi_sub(traj2,[N N]*scale_field,ones(N*scale_field,N*scale_field),imresize(w2d,[N N]*scale_field),dt,seg,T);
rawdata=F*imresize(img_mr,[N N]*scale_field)/scale_field;
F=orc_segm_nuFTOperator_multi_sub(traj,[N N],ones(N,N),w2d,dt,seg,T);
param.F=F;
param.rawdata=rawdata;
param.img=img_mr;

% param.W = DWT(3*[1,1],[N N]*scale_field, true, 'haar');
% param.alpha = gather(PowerIteration_mreg(param.F, ones(N*scale_field,N*scale_field)));
% param.W = DWT(3*[1,1],[N N], true, 'haar');
% param.alpha = gather(PowerIteration_mreg(param.F, ones(N,N)));
% param.init=gather(param.F'*(double(param.rawdata)));
% recon= gather(ReconWavFISTA_mreg(param.init, param.F, 0.05*max(col(abs(param.init))), param.W, param.alpha, param.init, 50, true));
recon=imresize(gather(regularizedReconstruction(F,rawdata,'maxit',50,'verbose_flag', 0,'tol',1e-5,'z0',[])),size(img));
figure(11),subplot(1,2,1),imagesc(real(recon));colorbar;colormap gray;axis equal;title(num2str(0));
subplot(1,2,2),imagesc(real(img));colorbar;colormap gray;axis equal;title(num2str(0));
drawnow

%%
% % If doTraining is false, download and extract the pretrained network from the MathWorks website.
pretrainedNetZipFile = matlab.internal.examples.downloadSupportFile('nnet','data/TrainedDiffusionNetwork.zip');
unzip(pretrainedNetZipFile);
load("DiffusionNetworkTrained/DiffusionNetworkTrained.mat");

% load([diffpath 'net.mat'],"net");

%%
figure(12)
tic;generatedImages_mar = generateAndDisplayImages_mar_mnist(net,param,varianceSchedule(1:10:end),imgSize);toc
%%


function recon = generateAndDisplayImages_mar_mnist(net,param,varianceSchedule,imageSize)
numChannels=1;
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
        z = randn([imageSize,numChannels,1]);
    else
        z = zeros([imageSize,numChannels,1]);
    end

    % Predict the noise using the network.
    predictedNoise = predict(net,(images),noiseStep);

    sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
    addedNoise = sqrt(posteriorVariance(noiseStep))*z;
    predNoise = varianceSchedule(noiseStep)*predictedNoise/sqrt(1 - alphaBar(noiseStep));
    
    images=images - predNoise;
    recon=imresize(gather(regularizedReconstruction(param.F,param.rawdata,'maxit',5,'verbose_flag', 0,'tol',1e-5,...
        'z0',imresize(images+1,size(param.F')))),size(images));
    % recon= gather(ReconWavFISTA_mreg(param.init, param.F, 0.05*max(col(abs(param.init))), param.W, param.alpha,...
    %     double(images+1), 5, true));
    images=real(recon)-1;
    % recon=images;
    images = 1/sqrtOneMinusBeta*(images) + addedNoise;
    if mod(noiseStep,10)==0
        subplot(1,2,1)
        imagesc(real(recon));colorbar;colormap gray;axis equal;
        title(num2str(noiseStep));
        subplot(1,2,2)
        imagesc(real(param.img));colorbar;colormap gray;axis equal;
        title(num2str(noiseStep));
        
        drawnow
    end
end
subplot(1,2,1)
imagesc(real(recon));colorbar;colormap gray;axis equal;
title(num2str(noiseStep));
subplot(1,2,2)
imagesc(real(param.img));colorbar;colormap gray;axis equal;
title(num2str(noiseStep));

drawnow
end


% function recon = generateAndDisplayImages_mar(net,rawdata,F,varianceSchedule,imageSize,numChannels)
% % Generate random noise.
% images = gpuArray(randn([imageSize numChannels]));
% 
% % Compute variance schedule parameters.
% alphaBar = cumprod(1 - varianceSchedule);
% alphaBarPrev = [1 alphaBar(1:end-1)];
% posteriorVariance = varianceSchedule.*(1 - alphaBarPrev)./(1 - alphaBar);
% 
% % Reverse the diffusion process.
% numNoiseSteps = length(varianceSchedule);
% 
% for noiseStep = numNoiseSteps:-1:1
%     if noiseStep ~= 1
%         z_r = randn([imageSize,numChannels,1]);
%         z_i = randn([imageSize,numChannels,1]);
%     else
%         z_r = zeros([imageSize,numChannels,1]);
%         z_i = zeros([imageSize,numChannels,1]);
%     end
% 
%     % Predict the noise using the network.
%     predictedNoise_r = predict(net,real(images),noiseStep);
%     predictedNoise_i = predict(net,imag(images),noiseStep);
% 
%     sqrtOneMinusBeta = sqrt(1 - varianceSchedule(noiseStep));
%     addedNoise_r = sqrt(posteriorVariance(noiseStep))*z_r;
%     addedNoise_i = sqrt(posteriorVariance(noiseStep))*z_i;
%     addedNoise=addedNoise_r+1i*addedNoise_i;
%     predNoise_r = varianceSchedule(noiseStep)*predictedNoise_r/sqrt(1 - alphaBar(noiseStep));
%     predNoise_i = varianceSchedule(noiseStep)*predictedNoise_i/sqrt(1 - alphaBar(noiseStep));
%     predNoise=predNoise_r+1i*predNoise_i;
% 
%     images=images - predNoise;
%     recon=imresize(gather(regularizedReconstruction(F,rawdata,'maxit',5,'verbose_flag', 0,'tol',1e-5,'z0',imresize(images,size(F')))),size(images));
%     images=recon;
%     images = 1/sqrtOneMinusBeta*(images) + addedNoise;
%     % if mod(noiseStep,10)==0
%         subplot(1,2,1)
%         imagesc(real(recon));colorbar;colormap gray;axis equal;
%         title(num2str(noiseStep));
%         subplot(1,2,2)
%         imagesc(real(images));colorbar;colormap gray;axis equal;
%         title(num2str(noiseStep));
% 
%         drawnow
%     % end
% end
% end