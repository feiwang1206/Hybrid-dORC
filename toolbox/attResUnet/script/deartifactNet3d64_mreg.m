clear
addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
doTraining = true;

scale=1;
imgSize = [64 64 48]/scale;

volReader = @(x) matRead(x);
dataFolder = '/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista17_image4_field/inputsTra';
imds_input = imageDatastore(dataFolder, 'FileExtensions','.mat','ReadFcn',volReader);
volReader = @(x) matRead3(x);
dataFolder = '/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista17_image4_field/labelsTra';
imds_label = imageDatastore(dataFolder, 'FileExtensions','.mat','ReadFcn',volReader);

% audsImds = augmentedImageDatastore(imgSize,imds_input,imds_label);

numInputChannels = 3;
numOutputChannels = 2;

numNoiseSteps = 10;

pname_s='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista17/';
pname_s1='/home/ubuntu/Documents/work/train/dataset/simu_fbu_modl_fista17_image4_field/';
param=dc(pname_s,pname_s1,1,1);

%%
miniBatchSize = 2;
numEpochs = 100;

learnRate = 0.0005;
gradientDecayFactor = 0.9;
squaredGradientDecayFactor = 0.9999;

mbq_input = minibatchqueue(imds_input, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatch, ...
    'MiniBatchFormat',"SSSCB", ...
    'PartialMiniBatch',"discard");
mbq_label = minibatchqueue(imds_label, ...
    'MiniBatchSize',miniBatchSize, ...
    'MiniBatchFcn',@preprocessMiniBatch, ...
    'MiniBatchFormat',"SSSCB", ...
    'PartialMiniBatch',"discard");

averageGrad = [];
averageSqGrad = [];

numObservationsTrain = numel(imds_input.Files);
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

if doTraining
    monitor = trainingProgressMonitor(...
        'Metrics',"Loss", ...
        'Info',["Epoch","Iteration"], ...
        'XLabel',"Iteration");
end
% %%
netname='/home/ubuntu/Documents/work/train/diffisionnet/nnet/GenerateImagesUsingDiffusionExample/net/deartefactNet3d_mreg.mat';
if doTraining
    if exist(netname,'file')
        load(netname,"net");
    else
        net = createDiffusionNetwork3d_att1(imgSize,numInputChannels,numOutputChannels);
    end
    iteration = 0;
    epoch = 0;

    while epoch < numEpochs && ~monitor.Stop
        epoch = epoch + 1;
        % shuffle(mbq);
        reset(mbq_input);
        reset(mbq_label);

        while hasdata(mbq_input) && ~monitor.Stop
            iteration = iteration + 1;
            

            img_input = next(mbq_input);
            img_label = next(mbq_label);
            % Generate random noise.
            targetNoise = randn(size(img_input),'Like',img_input);

            % Generate a random noise step.
            noiseStep = dlarray(randi(numNoiseSteps,[1 miniBatchSize],'Like',img_input),"CB");

            % Apply noise to the image.
            [Img_clean,Img_degrade] = applyNoiseToImage_ds(img_input,img_label,(noiseStep)/numNoiseSteps);

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
        generateAndDisplayImages2(net,param,numNoiseSteps);
    end
    save(netname,"net");
else
    load(netname,"net");
end
%%
% figure
tic;generatedImages2 = generateAndDisplayImages2(net,param,numNoiseSteps);toc

%%
function Img_degrade = generateAndDisplayImages2(net,param,numsteps)
Img_input = param.image0;
Img_degrade = param.image0;
images0=complex(Img_input(:,:,:,1),Img_input(:,:,:,2));
images01=images0;
loss=l2norm(col(param.ref/param.norm)-col(images0));
printf(['loss:' num2str(loss)])
for noiseStep = numsteps:-1:1
    % Predict the noise using the network.
    Img_predict = predict(net,Img_degrade,noiseStep);
    images0=complex(Img_predict(:,:,:,1),Img_predict(:,:,:,2));

    % res=param.Fg'*(param.rawdata-param.Fg*gather(double(images0*param.norm)));
    % res2=param.Fg'*(param.Fg*gather(res));
    % alpha=sum(col(conj(res) .* res))/sum(col(conj(res2) .* res2));
    % images0=images0+(1-0.0)*alpha*res/param.norm;
    % images0 = regularizedReconstruction_dc(param.Fg,param.rawdata,0.0,'maxit',1,'verbose_flag', 0,'tol',1e-5,'z0',gather(images0*param.norm))/param.norm;
    % images0 = single(gather(ReconWavFISTA_mreg(param.init, param.Fg, ...
    %     double(0.05*max(abs(col(param.init)))),...
    %     param.W, double(param.alpha), double(images01*param.norm), noiseStep, true)))/param.norm;
    lambda=0.1;
    images0 = single(gather(ReconWavFISTA_mreg_dc(param.init*(1-lambda), param.Fg, ...
        double(0.05*max(abs(col(param.init)))),...
        param.W, double(param.alpha), double(images0*param.norm), 1, true,lambda)))/param.norm;
    Img_predict(:,:,:,1)=real(images0);
    Img_predict(:,:,:,2)=imag(images0);

    % Img_degrade(:,:,:,1:2) = 1/numsteps.*(Img_predict-Img_input(:,:,:,1:2)) + Img_degrade(:,:,:,1:2);
    Img_degrade(:,:,:,1:2) = 1/noiseStep.*Img_predict + (1-1/noiseStep)*Img_degrade(:,:,:,1:2);

    % Display intermediate images.
    % imagesc(array2mosaic(gather(abs(complex(Img_predict(:,:,:,1),Img_predict(:,:,:,2)))))); colorbar;colormap gray
    % drawnow
    loss=l2norm(col(param.ref/param.norm)-col(images0));
    printf(['loss:' num2str(loss)])
end
end
%%
function Img_predict = generateAndDisplayImages(net,Img,numsteps)
% Generate random noise.
Img_input=Img(:,:,:,1:2,:);
% Img_clean=Img(:,:,:,3:4);
% Img_degrade=Img_input;
Img_degrade = Img(:,:,:,1:4,:);
Img_clean = Img(:,:,:,5:6,:);
% noise = randn(size(Img_input))*0.01;
% Img_degrade=Img_degrade+sqrt(1).*noise;

for noiseStep = (numsteps):-1:1
    Img_predict = predict(net,Img_degrade,noiseStep);
    % if noiseStep>1
        % Img_degrade = (1-(noiseStep-1)/numsteps).*Img_predict + (noiseStep-1)/numsteps.*Img_degrade;
        Img_degrade(:,:,:,1:2,:) = 1/(noiseStep).*Img_predict + (1-1/(noiseStep)).*Img_degrade(:,:,:,1:2,:);
        % Img_degrade(:,:,:,1:2,:) = 1/numsteps.*(Img_predict-Img_input) + Img_degrade(:,:,:,1:2,:);
        % noise = randn(size(Img_input))*0.01;
        % Img_degrade=Img_degrade+sqrt(noiseStep/numsteps).*noise;
    % end
    loss(numsteps-noiseStep+1)=l2norm(Img_clean-Img_predict);
    % printf(['loss:' num2str(loss)])

    tLay = tiledlayout("flow");
    title(tLay,"t = "+ noiseStep)
    for ii = 1:1
        nexttile
        % imagesc(array2mosaic(abs(complex(Img_degrade(:,:,:,1,ii),Img_degrade(:,:,:,2,ii)))))
        tmp=Img_input+Img_predict(:,:,:,1:2,:);
        imagesc(array2mosaic(abs(complex(tmp(:,:,:,1,ii),tmp(:,:,:,2,ii)))))
    end
    drawnow
end

% Display final images.
loss(end+1)=l2norm(Img_clean-Img_degrade(:,:,:,1:2,:));
loss(end+1)=l2norm(Img_clean-Img_input*0);
printf(['loss:' num2str(loss)])

end

%%
function Img_degrade = generateAndDisplayImages3(net,param,numsteps)
Img_input = param.image0;
Img_degrade = param.image0;
images0=complex(Img_input(:,:,:,1),Img_input(:,:,:,2));
loss=l2norm(col(param.ref)-col(images0));
printf(['loss:' num2str(loss)])
for noiseStep = numsteps:-1:1
    % Predict the noise using the network.
    Img_art = predict(net,Img_degrade,noiseStep);
    
    Img_predict = Img_degrade(:,:,:,1:2) + Img_art;

    images0=complex(Img_predict(:,:,:,1),Img_predict(:,:,:,2));
    % images0 = regularizedReconstruction_dc(param.Fg,param.rawdata,param.P{:},'maxit',10,'verbose_flag', ...
    %     0,'tol',1e-5,'z0',images0);
    images0 = single(gather(ReconWavFISTA_mreg(param.init, param.Fg, double(0.0*max(abs(col(param.init)))),...
        param.W, double(param.alpha), double(images0), 1, true)));
    Img_predict(:,:,:,1)=real(images0);
    Img_predict(:,:,:,2)=imag(images0);

    Img_degrade(:,:,:,1:2) = 1/numsteps.*(Img_predict-Img_input(:,:,:,1:2)) + Img_degrade(:,:,:,1:2);

    % Display intermediate images.
    imagesc(array2mosaic(gather(abs(complex(Img_predict(:,:,:,1),Img_predict(:,:,:,2)))))); colorbar;colormap gray
    drawnow
    loss=l2norm(col(param.ref)-col(images0));
    printf(['loss:' num2str(loss)])
end
end
%%
% function [Img_art,Img_degrade] = applyNoiseToImage_ds(Img_input,Img_label,noiseStep)
%     Img_input = extractdata(Img_input);
%     Img_label = extractdata(Img_label);
%     noiseStep = extractdata(noiseStep);
% 
%     Img_degrade = Img_input;
%     ss=size(Img_label);
%     Img_art = repmat((permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_label;
%     Img_art_bar = repmat((1-permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_label;
% 
%     Img_degrade(:,:,:,[1 2],:) = Img_art_bar + Img_input(:,:,:,[1 2],:);
% 
%     % Img_degrade(:,:,:,[1 3],:) = Img_art_bar(:,:,:,[2 3],:) + Img_input(:,:,:,[1 3],:);
%     % Img_degrade(:,:,:,[2 4],:) = Img_art_bar(:,:,:,[2 3],:) + Img_input(:,:,:,[2 4],:);
%     % Img_degrade(:,:,:,5,:) = Img_art_bar(:,:,:,1,:)/10 + Img_input(:,:,:,5,:);
% 
%     Img_degrade = dlarray(Img_degrade,'SSSCB');
%     Img_art = dlarray(Img_art,'SSSCB');
%     % noise = dlarray(randn(size(Img_input)),'SSSCB')*0.01;
%     % Img_degrade = Img_degrade+sqrt(noiseStep).*noise;
% end
%%
function [Img_clean,Img_degrade] = applyNoiseToImage_ds(Img_input,Img_label,noiseStep)
    % Img_input = extractdata(Img_input);
    % Img_label = extractdata(Img_label);
    % noiseStep = extractdata(noiseStep);

    Img_degrade = Img_input;
    Img_clean = Img_input(:,:,:,[1 2],:)+Img_label;

    Img_degrade(:,:,:,[1 2],:) = (1-noiseStep).*Img_clean + noiseStep.*Img_input(:,:,:,[1 2],:);
    
    % Img_degrade = dlarray(Img_degrade,'SSSCB');
    % Img_clean = dlarray(Img_art,'SSSCB');
    % noise = dlarray(randn(size(Img_input)),'SSSCB')*0.01;
    % Img_degrade = Img_degrade+sqrt(noiseStep).*noise;
end
%%
function [Img_art,Img_degrade] = applyNoiseToImage(Img_input,Img_label,noiseStep)
    Img_input = extractdata(Img_input);
    Img_label = extractdata(Img_label);
    noiseStep = extractdata(noiseStep);
    Img_degrade = Img_input;
    ss=size(Img_label);
    Img_art = repmat((permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_label;
    Img_art_bar = repmat((1-permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_label;

    Img_degrade(:,:,:,[1 2],:) = Img_art_bar + Img_input(:,:,:,[1 2],:);

    % Img_degrade(:,:,:,[1 3],:) = Img_art_bar(:,:,:,[2 3],:) + Img_input(:,:,:,[1 3],:);
    % Img_degrade(:,:,:,[2 4],:) = Img_art_bar(:,:,:,[2 3],:) + Img_input(:,:,:,[2 4],:);
    % Img_degrade(:,:,:,5,:) = Img_art_bar(:,:,:,1,:)/10 + Img_input(:,:,:,5,:);
    
    % Img_degrade = dlarray(Img_degrade,'SSSCB');
    % Img_art = dlarray(Img_art,'SSSCB');
    % noise = dlarray(randn(size(Img_input)),'SSSCB')*0.01;
    % Img_degrade = Img_degrade+sqrt(noiseStep).*noise;
end
%%
% function [Img_clean,Img_degrade] = applyNoiseToImage(img_input,img_label,noiseStep)
%     img_input = extractdata(img_input);
%     img_label = extractdata(img_label);
%     noiseStep = extractdata(noiseStep);
%     % Img_input = dlarray(img(:,:,:,1:2,:),'SSSCB');
%     % Img_clean = dlarray(img(:,:,:,1:2,:)+img(:,:,:,3:4,:),'SSSCB');
%     % Img_degrade = (1-noiseStep).*Img_clean + noiseStep.*Img_input;
%     Img_input = img(:,:,:,1:2,:);
%     Img_clean = img(:,:,:,5:6,:);
%     ss=size(Img_clean);
%     Img_degrade(:,:,:,1:2,:) = repmat((1-permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_clean + repmat((permute(noiseStep,[1 3 4 5 2])),[ss(1:4) 1]).*Img_input;
%     Img_degrade(:,:,:,3:4,:) = img(:,:,:,3:4,:);
%     Img_degrade = dlarray(Img_degrade,'SSSCB');
%     Img_clean = dlarray(Img_clean,'SSSCB');
%     % noise = dlarray(randn(size(Img_input)),'SSCB')*0.01;
%     % Img_degrade = Img_degrade+sqrt(noiseStep).*noise;
% end
function [loss, gradients] = modelLoss(net,X,Y,T)
% Forward data through the network.
noisePrediction = forward(net,X,Y);

% Compute mean squared error loss between predicted noise and target.
loss = mse(noisePrediction,T);

gradients = dlgradient(loss,net.Learnables);
end
function [data] = preprocessMiniBatch(data)
% Concatenate mini-batch.
data = cat(5,data{:});
% label = cat(5,label{:});

% Rescale the images so that the pixel values are in the range [-1 1].
% X = rescale(X,-1,1,'InputMin',0,'InputMax',255);
end

