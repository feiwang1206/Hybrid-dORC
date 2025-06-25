function param=dc(pname_source0,pname_source,ff,scale)
%% single image 256 
% scale=2;
dim0=[64 64 50];
dim=[64 64 48]/scale;
load(['/home/ubuntu/Documents/work/train/all/data0.mat']);
dt=5e-6;
te=0.002;

trajectory=data.trajectory;
traj = trajectory.trajectory;
traj_idx = trajectory.idx;
for ii=1:3
    traj1{1}(:,ii) = traj{1}(traj_idx{1},ii)*(dim0(ii)/dim(ii));   
    traj1{2}(:,ii) = traj{2}(traj_idx{2},ii)*(dim0(ii)/dim(ii)); 
end
Tt{1}=dt*traj_idx{1}+te;
Tt{2}=dt*traj_idx{2}+te;
Fg0=orc_segm_nuFTOperator_structure2(traj1,dim,dt,10,Tt,0.01);

lengthP = 0;
P = cell(1,lengthP);
counter = 1;
for k=1:1
    operator(1).handle = @identityOperator;
    operator(1).args = {};
    P{counter} = @L2Norm;%recon_details.penalty(k).norm;
    counter = counter + 1;
    P{counter} = 0;
    counter = counter + 1;
    for n=1:length(operator)
        P{counter} = operator(n).handle(operator(n).args{:});
        counter = counter + 1;
    end
end
    
fnames=dir([pname_source '/inputsVal']);
load([pname_source0 '/' fnames(ff+2).name]);
image01=load([pname_source '/inputsVal/' fnames(ff+2).name]);
% param.image0=image0.image;
% norm0=load([pname_source '/norm/' fnames(ff+2).name]);
% norm=norm0.norm;
smaps=load([pname_source0 '/smaps/' num2str(recon{12}) '.mat']);
smaps=imresize4D(smaps.smaps,dim);

field=imresize3D(recon{6},dim);

param.ref=imresize3D(recon{11},dim);
param.Fg=orc_segm_nuFTOperator_multi_savetime_smaps(Fg0,gpuArray(double(smaps)),gpuArray(double(field)));
param.rawdata=gpuArray(double(param.Fg*param.ref));
param.rawdata=awgn(param.rawdata,40); 
param.init=double(gather(param.Fg'*gpuArray(param.rawdata)));
param.W = DWT(3*[1,1,1],dim, true, 'haar');
param.alpha=gather(PowerIteration_mreg2(param.Fg, gpuArray(ones(dim)),5));
image0 = single(gather(ReconWavFISTA_mreg(param.init, param.Fg, double(0.05*max(abs(col(param.init)))),...
    param.W, double(param.alpha), double(param.init), 5, true)));
param.norm=max(abs(col(image0)));
param.image0(:,:,:,1)=real(image0)/param.norm;
param.image0(:,:,:,2)=imag(image0)/param.norm;
param.image0(:,:,:,3)=field/1000;
param.image0(:,:,:,4)=image01.image(:,:,:,4);

% tic;param.image0 = regularizedReconstruction_dc(Fg,rawdata,P{:},'maxit',5,'verbose_flag', 0,'tol',1e-5,'z0',init);toc
% param.P=P;
param.dim=dim;
end
