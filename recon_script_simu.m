% function test_exp(rang,pname)
clear
%%    %% iter1
addpath(genpath('toolbox/attResUnet'));
addpath(genpath('toolbox/mreg_recon_tool'));
addpath(('toolbox/image_reconstruction_toolbox'));
setup

if gpuDeviceCount>0
    gpu_acc=1;
else
    gpu_acc=0;
end
%%
lambda=0.01;

scale=[1 1 1];
scale2=[1 1 1]/1;
iter=20;
folder='simu_dynamic/';
load('data/data_simu.mat');
recon00=data.phantom_simu;
if ~exist(folder,'dir')
    mkdir(folder);
end

dim=[64 64 50];
dim1=[64 64 48];
dt=5e-6;
deadtime(1)=0.002;
deadtime(2)=0.002;
te=0.0341;
traj = data.trajectory.trajectory;
traj_idx = data.trajectory.idx;
traj{1} = traj{1}(traj_idx{1},:);   
traj{2} = traj{2}(traj_idx{2},:); 
dis=traj{1}(:,1).^2+traj{1}(:,2).^2+traj{1}(:,3).^2;
[center]=find(dis==min(dis));

traj1=traj;
traj1{1}(:,1)=traj{1}(:,1);
traj1{1}(:,2)=traj{1}(:,2);
traj1{1}(:,3)=traj{1}(:,3)*dim(3)/dim1(3);
traj1{2}(:,1)=traj{2}(:,1);
traj1{2}(:,2)=traj{2}(:,2);
traj1{2}(:,3)=traj{2}(:,3)*dim(3)/dim1(3);

traj2=traj1;
subset1=(1:length(traj2{1}));
Tt{1}=(subset1+traj_idx{1}(1))*dt+deadtime(1);


subset2=(1:length(traj2{2}));
Tt{2}=(subset2+traj_idx{2}(1))*dt+deadtime(2);

L{1}='l1';
L{2}='tv';
L{3}=1e-5;
clear operator
if strcmp(L{2},'tv')
    operator(1).handle = @finiteDifferenceOperator;
    operator(1).args = {1};
    operator(2).handle = @finiteDifferenceOperator;
    operator(2).args = {2};
    operator(3).handle = @finiteDifferenceOperator;
    operator(3).args = {3};
elseif strcmp(L{2},'id')
    operator.handle = @identityOperator;
    operator.args = {};
elseif strcmp(L{2},'wl')
    operator(1).handle = @waveletDecompositionOperator;
    operator(1).args = {imsizeout,3,'db2'};
end


lengthP = 0;
P = cell(1,lengthP);
counter = 1;

if strcmp(L{1},'l1')
    P{counter} = @L1Norm;
else
    P{counter} = @L2Norm;
end
counter = counter + 1;
%     lambda=1e-4;
P{counter} = L{3};
counter = counter + 1;
for k=1:length(operator)
    P{counter} = operator(k).handle(operator(k).args{:});
    counter = counter + 1;
end

L{1}='l2';
L{2}='tv';
L{3}=1e-5;
clear operator
if strcmp(L{2},'tv')
    operator(1).handle = @finiteDifferenceOperator;
    operator(1).args = {1};
    operator(2).handle = @finiteDifferenceOperator;
    operator(2).args = {2};
    operator(3).handle = @finiteDifferenceOperator;
    operator(3).args = {3};
elseif strcmp(L{2},'id')
    operator(1).handle = @identityOperator;
    operator(1).args = {};
elseif strcmp(L{2},'wl')
    operator(1).handle = @waveletDecompositionOperator;
    operator(1).args = {imsizeout,3,'db2'};
end


lengthP = 0;
P2 = cell(1,lengthP);
counter = 1;

if strcmp(L{1},'l1')
    P2{counter} = @L1Norm;
else
    P2{counter} = @L2Norm;
end
counter = counter + 1;
%     lambda=1e-4;
P2{counter} = L{3};
counter = counter + 1;
for k=1:length(operator)
    P2{counter} = operator(k).handle(operator(k).args{:});
    counter = counter + 1;
end
%% wmap error

smaps=double(data.smaps);
dte=0.003;
Tt2{1}=Tt{1}+dte;
Tt2{2}=Tt{2}+dte;
if gpu_acc==1
    Fg0{1}=orc_segm_nuFTOperator_structure(traj2(1),dim1./scale,gpuArray(imresize4D(smaps,dim1)),dt,10,Tt(1),0.01);
    Fg0{2}=orc_segm_nuFTOperator_structure(traj2(2),dim1./scale,gpuArray(imresize4D(smaps,dim1)),dt,10,Tt(2),0.01);
    Fg0{4}=orc_segm_nuFTOperator_structure(traj2,dim1./scale,gpuArray(imresize4D(smaps,dim1)),dt,10,Tt,0.01);
    Fg0_dte{4}=orc_segm_nuFTOperator_structure(traj2,dim1./scale,gpuArray(imresize4D(smaps,dim1)),dt,10,Tt2,0.01);
else
    Fg0{1}=orc_segm_nuFTOperator_structure(traj2(1),dim1./scale,(imresize4D(smaps,dim1)),dt,10,Tt(1),0.01);
    Fg0{2}=orc_segm_nuFTOperator_structure(traj2(2),dim1./scale,(imresize4D(smaps,dim1)),dt,10,Tt(2),0.01);
    Fg0{4}=orc_segm_nuFTOperator_structure(traj2,dim1./scale,(imresize4D(smaps,dim1)),dt,10,Tt,0.01);
    Fg0_dte{4}=orc_segm_nuFTOperator_structure(traj2,dim1./scale,(imresize4D(smaps,dim1)),dt,10,Tt2,0.01);
end

traj3=traj2;
traj3{1}(:,1)=traj2{1}(:,1)*scale2(1);
traj3{1}(:,2)=traj2{1}(:,2)*scale2(2);
traj3{1}(:,3)=traj2{1}(:,3)*scale2(3);
traj3{2}(:,1)=traj2{2}(:,1)*scale2(1);
traj3{2}(:,2)=traj2{2}(:,2)*scale2(2);
traj3{2}(:,3)=traj2{2}(:,3)*scale2(3);
for shot=1:2
    if gpu_acc==1
        Fg02{shot}=orc_segm_nuFTOperator_structure(traj3(shot),dim1./scale2,gpuArray(imresize4D(smaps,dim1./scale2)),...
            dt,10,{Tt{shot}},0.01);
        Fg02_dte{shot}=orc_segm_nuFTOperator_structure(traj3(shot),dim1./scale2,...
            gpuArray(imresize4D(smaps,dim1./scale2)),dt,10,{Tt2{shot}},0.01);
    else
        Fg02{shot}=orc_segm_nuFTOperator_structure(traj3(shot),dim1./scale2,(imresize4D(smaps,dim1./scale2)),...
            dt,10,{Tt{shot}},0.01);
        Fg02_dte{shot}=orc_segm_nuFTOperator_structure(traj3(shot),dim1./scale2,...
            (imresize4D(smaps,dim1./scale2)),dt,10,{Tt2{shot}},0.01);
    end
end
%%
TR=2;
frame_times=TR:TR:96;
rang=1:length(frame_times);
mshift=gaussmf(rang,[15 24]);
pulse=sin(2*pi*1/8.8*(frame_times-TR)).*mshift*1;
mask_decay=ones(dim1);
image0=imresize3D(recon00{11},dim1);
wmap_fake=imresize3D(recon00{3},dim1)*2;
wmap_err=imresize3D(recon00{6}-recon00{3},dim1);
ana=gather(imresize3D(abs(recon00{21}),dim1));
ana=ana/max(ana(:));
ana=smooth3(ana,'box',5);
mask=ana>0.3;
mask(:,:,1:3)=0;
mask(:,:,46:48)=0;
miter=5;
%%
for n=rang
    disp(num2str(n))
    subfolder='rawdata/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];
    if ~exist(fname,'file')
        wmap_real=smooth3(real(imshift(wmap_fake+wmap_err,[0 0 pulse(n)])));
        image1=imshift(image0,[0 0 pulse(n)]);
        if gpu_acc==1
            for shot=1:2
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg02{shot},gpuArray(imresize3D(wmap_real,dim1./scale2)));
                rawdata{shot}=gather(Fg{shot}*imresize3D(image1,dim1./scale2)).*sqrt(prod(scale2));
            end    
            for shot=1:2
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg02_dte{shot},gpuArray(imresize3D(wmap_real,dim1./scale2)));
                rawdata{shot+2}=gather(Fg{shot}*imresize3D(image1,dim1./scale2)).*sqrt(prod(scale2));
            end
        else
            for shot=1:2
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg02{shot},(imresize3D(wmap_real,dim1./scale2)));
                rawdata{shot}=Fg{shot}*imresize3D(image1,dim1./scale2).*sqrt(prod(scale2));
            end    
            for shot=1:2
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg02_dte{shot},(imresize3D(wmap_real,dim1./scale2)));
                rawdata{shot+2}=Fg{shot}*imresize3D(image1,dim1./scale2).*sqrt(prod(scale2));
            end
        end

        recon{17}=rawdata;
        recon{18}{1}=awgn(recon{17}{1},40);
        recon{18}{2}=awgn(recon{17}{2},40);
        recon{18}{3}=awgn(recon{17}{3},40);
        recon{18}{4}=awgn(recon{17}{4},40);
        recon{6}=wmap_real;
        recon{3}=wmap_fake;
        recon{11}=image1;
        shot=4;
        P{2}=lambda*max(abs(col([recon{18}{1};recon{18}{2}])));
        if gpu_acc==1
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(imresize3D(recon{6},dim1)));
            recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double([recon{18}{1};recon{18}{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5));
        else
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(imresize3D(recon{6},dim1)));
            recon{shot} = regularizedReconstruction(Fg{shot},(double([recon{18}{1};recon{18}{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5);
        end
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('ground truth')
        save(fname,'recon');
    else
        load(fname,'recon');
    end
            
    rawdata=recon{18};
    recon=[];
    %% uncorrect
    subfolder='recon_un/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];
    recon{3}=0*imresize3D(wmap_fake,dim1);
    if ~exist(fname,'file')
        shot=4;
        P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
        if gpu_acc==1
            Fg{shot}=nuFTOperator([traj2{1};traj2{2}],dim1./scale,gpuArray(imresize4D(smaps,dim1)));
            recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5));
        else
            Fg{shot}=nuFTOperator([traj2{1};traj2{2}],dim1./scale,(imresize4D(smaps,dim1)));
            recon{shot} = regularizedReconstruction(Fg{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5);
        end
        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('uncorrected')
        save(fname,'recon');
    end
    %% static
    subfolder='recon_static/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];
    recon{3}=imresize3D(wmap_fake,dim1);
    if ~exist(fname,'file')
        shot=4;
        P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
        if gpu_acc==1
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
            recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5));
        else
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
            recon{shot} = regularizedReconstruction(Fg{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',50,'verbose_flag', 0,'tol',1e-5);
        end

        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('static')
        save(fname,'recon');
    end

    %% static dual-TE phase
    subfolder='recon_static_phase/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];
        truth=load([folder 'rawdata/mat/' num2str(n) '.mat']);
    if ~exist(fname,'file')
        load([folder 'recon_static/mat/' num2str(n) '.mat']);
        wmap0=recon{3};
        shot=4;
        if gpu_acc==1
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
            P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
            recon1 = gather(regularizedReconstruction(Fg{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',10,'verbose_flag', 0,'tol',1e-5));
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0_dte{shot},gpuArray(recon{3}));
            P{2}=lambda*max(abs(col([rawdata{3};rawdata{4}])));
            recon2 = gather(regularizedReconstruction(Fg{shot},gpuArray(double([rawdata{3};rawdata{4}])),P{:},...
                'maxit',10,'verbose_flag', 0,'tol',1e-5));
        else
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
            P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
            recon1 = regularizedReconstruction(Fg{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',10,'verbose_flag', 0,'tol',1e-5);
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0_dte{shot},(recon{3}));
            P{2}=lambda*max(abs(col([rawdata{3};rawdata{4}])));
            recon2 = regularizedReconstruction(Fg{shot},(double([rawdata{3};rawdata{4}])),P{:},...
                'maxit',10,'verbose_flag', 0,'tol',1e-5);
        end

        for iter=1:5
            pmaps(:,:,:,1)=gather(recon1);
            pmaps(:,:,:,2)=gather(recon2);
            wmap = fieldmap(angle(pmaps(:,:,:,2)./pmaps(:,:,:,1)),mask,ana,dte);
            wmap = mri_field_map_reg3D(pmaps,[0 dte],'l2b',-1,'winit',wmap,'mask',mask)+recon{3};
            wmap = wmap0+smooth3(wmap-wmap0,'box',5);
            rmse=l2norm(wmap(mask==1)-recon{3}(mask==1))/sqrt(sum(col(mask)));
            printf(['static dual-TE phase ' num2str(rmse)]);
            if rmse<1
                break;
            end
            % err=l2norm(wmap(mask==1)-truth.recon{6}(mask==1))/sqrt(sum(col(mask)));
            % printf(['error ' num2str(err)]);
            recon{3}=wmap;
            if gpu_acc==1
                Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
                P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
                recon1 = gather(regularizedReconstruction(Fg1{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                    'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',gpuArray(recon1)));
                Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0_dte{shot},gpuArray(recon{3}));
                P{2}=lambda*max(abs(col([rawdata{3};rawdata{4}])));
                recon2 = gather(regularizedReconstruction(Fg1{shot},gpuArray(double([rawdata{3};rawdata{4}])),P{:},...
                    'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',gpuArray(recon2)));
            else
                Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
                P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
                recon1 = gather(regularizedReconstruction(Fg1{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                    'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',recon1));
                Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0_dte{shot},(recon{3}));
                P{2}=lambda*max(abs(col([rawdata{3};rawdata{4}])));
                recon2 = gather(regularizedReconstruction(Fg1{shot},(double([rawdata{3};rawdata{4}])),P{:},...
                    'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',recon2));
            end
        end
        recon{shot}=recon1;
        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('dual-TE phase')
        save(fname,'recon');
    end
    %% static+rp
    subfolder='recon_static_rp/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];
    if ~exist(fname,'file')
        if n>1
            recon1=load([folder 'recon_static/mat/' num2str(1) '.mat']);
            recon1=recon1.recon;
            load([folder 'recon_static/mat/' num2str(n) '.mat']);
            wmap0=recon{3};

            shot=4;
            for iter=1:miter
                pmaps(:,:,:,1)=gather(recon1{shot});
                pmaps(:,:,:,2)=gather(recon{shot});
                wmap = fieldmap(angle(pmaps(:,:,:,2)./pmaps(:,:,:,1)),mask,ana,te);
                wmap = mri_field_map_reg3D(pmaps,[0 te],'l2b',-1,'winit',wmap,'mask',mask)+recon{3};
                wmap = wmap0+smooth3(wmap-wmap0,'box',5);
                rmse=l2norm(wmap(mask==1)-recon{3}(mask==1))/sqrt(sum(col(mask)));
                printf(['static rp ' num2str(rmse)]);
                if rmse<1
                    break;
                end
                recon{3}=wmap;
                if gpu_acc==1
                    Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
                    P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
                    recon{shot} = gather(regularizedReconstruction(Fg1{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                        'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',gpuArray(recon{shot})));
                else
                    Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
                    P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
                    recon{shot} = regularizedReconstruction(Fg1{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                        'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',recon{shot});
                end
            end
        else
            load([folder 'recon_static/mat/' num2str(n) '.mat']);            
        end
        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('rp')
        save(fname,'recon');
    end
    %% recon static+jmbir
    subfolder='recon_static_jmbir/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    wmap0=imresize3D(wmap_fake,dim1);

    shot=4;
    fname=[folder subfolder num2str(n) '.mat'];
    if ~exist(fname,'file')
        load([folder 'recon_static/mat/' num2str(n) '.mat']);
        rawdata4=[rawdata{1};rawdata{2}];
        recon{shot}=zeros(dim1);
        for step=1:5
            P{2}=lambda*max(abs(col(rawdata4)));
            if gpu_acc==1
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
                recon{shot}=double(gather(regularizedReconstruction(Fg{shot},gpuArray(double(rawdata4)),P{:},...
                    'maxit',5,'verbose_flag',0,'z0',recon{shot})));
                Fg_wmap=orc_segm_nuFTOperator_wmap_multi_savetime(Fg0{shot},gpuArray(recon{3}),gpuArray(recon{shot}),...
                        gpuArray(double(rawdata4)));
                rawdata1=rawdata4-Fg{shot}*recon{shot};
                wmap=double(gather(regularizedReconstruction_wmap(Fg_wmap,gpuArray(double(rawdata1)),...
                    'maxit',5,'verbose_flag',0,'tol',1e-5)))+recon{3};
            else
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
                recon{shot}=double(regularizedReconstruction(Fg{shot},(double(rawdata4)),P{:},...
                    'maxit',5,'verbose_flag',0,'z0',recon{shot}));
                Fg_wmap=orc_segm_nuFTOperator_wmap_multi_savetime(Fg0{shot},(recon{3}),recon{shot},...
                        (double(rawdata4)));
                rawdata1=rawdata4-Fg{shot}*recon{shot};
                wmap=double(regularizedReconstruction_wmap(Fg_wmap,(double(rawdata1)),...
                    'maxit',5,'verbose_flag',0,'tol',1e-5))+recon{3};
            end
            
            wmap=wmap0+smooth3(wmap-wmap0);
            rmse=l2norm(wmap(mask==1)-recon{3}(mask==1))/sqrt(sum(col(mask)));
                printf(['static jmbir ' num2str(rmse)]);
            if rmse<0.2
                break;
            end
            recon{3}=wmap;

       end
        recon{17}=[];
        recon{18}=[];
        save(fname,'recon');
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('jmbir')
    end
    %% jmodl
    subfolder='recon_jmodl/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];

    recon{3}=imresize3D(wmap_fake,dim1);
    if ~exist(fname,'file')
        if gpu_acc==1
            Fg{1}=orc_segm_nuFTOperator_multi_savetime(Fg0{1},gpuArray(imresize3D(recon{3},dim1)));
            Fg{2}=orc_segm_nuFTOperator_multi_savetime(Fg0{2},gpuArray(imresize3D(recon{3},dim1)));
            
            for shot=1:2
                P{2}=lambda*max(abs(col(rawdata{shot})));
                recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double(rawdata{shot})),P{:},'maxit',5,...
                    'verbose_flag', 0,'tol',1e-5));
            end
        else
            Fg{1}=orc_segm_nuFTOperator_multi_savetime(Fg0{1},(imresize3D(recon{3},dim1)));
            Fg{2}=orc_segm_nuFTOperator_multi_savetime(Fg0{2},(imresize3D(recon{3},dim1)));
            
            for shot=1:2
                P{2}=lambda*max(abs(col(rawdata{shot})));
                recon{shot} = regularizedReconstruction(Fg{shot},(double(rawdata{shot})),P{:},'maxit',5,...
                    'verbose_flag', 0,'tol',1e-5);
            end
        end
        input{1}=imresize3D(recon{1},dim1);
        input{2}=imresize3D(recon{2},dim1);
        norm=max(col(abs(input{1}+input{2})/2));
        for nn=1:4
            input{1}=imresize3D(recon{1},dim1);
            input{2}=imresize3D(recon{2},dim1);
            input_field=imresize3D(recon{3},dim1);
            image=zeros([dim1 5]);
            image(:,:,:,1)=real(gather(input{1}))/norm;
            image(:,:,:,2)=imag(gather(input{1}))/norm;
            image(:,:,:,3)=real(gather(input{2}))/norm;
            image(:,:,:,4)=imag(gather(input{2}))/norm;
            image(:,:,:,5)=gather(input_field)/1000;
            load(['net/net_final' num2str(nn) '.mat'],'net');
            if gpu_acc==1
                tmp=gather(double(predict(net,gpuArray(image)))); 
            else
                tmp=double(predict(net,image)); 
            end
            recon{3} = tmp(:,:,:,1)*100+image(:,:,:,5)*1000;
            recon0=imresize3D((tmp(:,:,:,2)+1i*tmp(:,:,:,3))*norm+(input{1}+input{2})/2,dim1);

            if nn<4
                for shot=1:2
                    P{2}=lambda*max(abs(col(rawdata{shot})));
                    if gpu_acc==1
                        Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(imresize3D(recon{3},dim1)));
                        recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double(rawdata{shot})),...
                            'maxit',5,'verbose_flag', 0,'tol',1e-5,'z0',recon0));
                    else
                        Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(imresize3D(recon{3},dim1)));
                        recon{shot} = regularizedReconstruction(Fg{shot},(double(rawdata{shot})),...
                            'maxit',5,'verbose_flag', 0,'tol',1e-5,'z0',recon0);
                    end
                end
            end
        end
        shot=4;
        P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
        if gpu_acc==1
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(imresize3D(recon{3},dim1)));
            recon{shot} = gather(regularizedReconstruction(Fg{shot},gpuArray(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',20,'verbose_flag', 0,'tol',1e-5,'z0',recon0));
        else
            Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(imresize3D(recon{3},dim1)));
            recon{shot} = regularizedReconstruction(Fg{shot},(double([rawdata{1};rawdata{2}])),P{:},...
                'maxit',20,'verbose_flag', 0,'tol',1e-5,'z0',recon0);
        end
        recon{1}=[];
        recon{2}=[];

        recon{17}=[];
        recon{18}=[];

        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('jmodl')
        save(fname,'recon');
    end

    %% jmodl+rp
    subfolder='recon_jmodl_rp/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];

    recon{3}=imresize3D(wmap_fake,dim1);
    shot=4;
    if ~exist(fname,'file')
        if n>1
            recon1=load([folder 'recon_jmodl/mat/' num2str(1) '.mat']);
            recon1=recon1.recon;
            load([folder 'recon_jmodl/mat/' num2str(n) '.mat']);
            wmap0=recon{3};
            for iter=1:5
                tmp0=recon1{shot};
                tmp1=recon{shot};

                pmaps(:,:,:,1)=gather(tmp0);
                pmaps(:,:,:,2)=gather(tmp1);
                wmap = fieldmap(angle(pmaps(:,:,:,2)./pmaps(:,:,:,1)),mask,ana,te);
                wmap = mri_field_map_reg3D(pmaps,[0 te],'l2b',-1,'winit',wmap,'mask',mask)+recon{3};
                wmap = wmap0+smooth3(wmap-wmap0,'box',5);
                rmse=l2norm(wmap(mask==1)-recon{3}(mask==1))/sqrt(sum(col(mask)));
                printf(['jmodl rp ' num2str(rmse)]);
                if rmse<1
                    break;
                end
                recon{3}=wmap;
                P{2}=lambda*max(abs(col([rawdata{1};rawdata{2}])));
                if gpu_acc==1
                    Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
                    recon{shot} = gather(regularizedReconstruction(Fg1{shot},gpuArray(double([rawdata{1};rawdata{2}])),...
                        P{:},'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',gpuArray(recon{shot})));
                else
                    Fg1{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
                    recon{shot} = regularizedReconstruction(Fg1{shot},(double([rawdata{1};rawdata{2}])),...
                        P{:},'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',recon{shot});
                end
            end
        else
            load([folder 'recon_jmodl/mat/' num2str(n) '.mat']);
        end


        recon{1}=[];
        recon{2}=[];
        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('jmodl+rp')
        
        save(fname,'recon');
    end
    %% modl field image rp jmbir
    subfolder='recon_jmodl_rp_jmbir/mat/';
    if ~exist([folder subfolder],'dir')
        mkdir([folder subfolder]);
    end
    fname=[folder subfolder num2str(n) '.mat'];

    if ~exist(fname,'file')
        recon_raw=load([folder 'rawdata/mat/' num2str(n) '.mat']);
        recon_raw=recon_raw.recon;
        load([folder 'recon_jmodl_rp/mat/' num2str(n) '.mat']);
        % recon_static=load([folder 'recon_static/' num2str(1) '.mat']);
        % error_image_jmbir(1)=l2norm((recon_raw{4}-recon{4}).*mask)/sqrt(sum(col(mask)));
        % error_field_jmbir(1)=l2norm((recon_raw{6}-recon{3}).*mask)/sqrt(sum(col(mask)));
        % printf([num2str(error_image_jmbir(1)) ' ' num2str(error_field_jmbir(1))])

        rawdata4=[rawdata{1};rawdata{2}];
        wmap0=recon{3};
        for step=1:5
            P{2}=lambda*max(abs(col(rawdata4)));
            if gpu_acc==1
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},gpuArray(recon{3}));
                recon{shot}=double(gather(regularizedReconstruction(Fg{shot},gpuArray(double(rawdata4)),P{:},...
                    'maxit',5,'verbose_flag',0,'z0',recon{shot})));
                Fg_wmap=orc_segm_nuFTOperator_wmap_multi_savetime(Fg0{shot},gpuArray(recon{3}),gpuArray(recon{shot}),...
                        gpuArray(double(rawdata4)));
    
                rawdata1=rawdata4-Fg{shot}*recon{shot};
                wmap=double(gather(regularizedReconstruction_wmap(Fg_wmap,gpuArray(double(rawdata1)),...
                    'maxit',5,'verbose_flag',0,'tol',1e-5)))+recon{3};
            else
                Fg{shot}=orc_segm_nuFTOperator_multi_savetime(Fg0{shot},(recon{3}));
                recon{shot}=double(regularizedReconstruction(Fg{shot},(double(rawdata4)),P{:},...
                    'maxit',5,'verbose_flag',0,'z0',recon{shot}));
                Fg_wmap=orc_segm_nuFTOperator_wmap_multi_savetime(Fg0{shot},(recon{3}),recon{shot},...
                        (double(rawdata4)));
    
                rawdata1=rawdata4-Fg{shot}*recon{shot};
                wmap=double(regularizedReconstruction_wmap(Fg_wmap,(double(rawdata1)),...
                    'maxit',5,'verbose_flag',0,'tol',1e-5))+recon{3};
            end
            wmap=wmap0+smooth3(wmap-wmap0);

            rmse=l2norm(wmap(mask==1)-recon{3}(mask==1))/sqrt(sum(col(mask)));
                printf(['jmodl rp jmbir ' num2str(rmse)]);
            if rmse<0.2
                break;
            end
            recon{3}=wmap;
            % error_image_jmbir(step+1)=l2norm((recon_raw{4}-recon{4}).*mask)/sqrt(sum(col(mask)));
            % error_field_jmbir(step+1)=l2norm((recon_raw{6}-recon{3}).*mask)/sqrt(sum(col(mask)));
            % printf([num2str(error_image_jmbir(step+1)) ' ' num2str(error_field_jmbir(step+1))])
        end

        recon{1}=[];
        recon{2}=[];
        recon{17}=[];
        recon{18}=[];
        figure,imagesc(array2mosaic(abs(abs(recon{4}))));axis equal;colormap gray;colorbar,title('jmodl+rp+jmbir')

        save(fname,'recon');
    end    

end