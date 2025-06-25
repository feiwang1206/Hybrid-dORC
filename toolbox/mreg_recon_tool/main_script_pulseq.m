addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
%% set work time frames and path
clear
run('/home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox/setup');
% example setting
rawdata_path=pwd;
dirs=dir('*.dat');
if dirs(1).name(25)=='p'
    mreg_filename=dirs(1).name;%mreg rawdata file name, e.g. meas_MID00133_FID01338_MREG.dat
else
    mreg_filename=dirs(2).name;%mreg rawdata file name, e.g. meas_MID00133_FID01338_MREG.dat
end
recon_path=pwd;%
% twx = mapVBVD(mreg_filename);
% load your trajectory and correct gradient delay
mreg_path=which('mreg_recon_tool');
% Set reconstruction parameters
recon_details.recon_resolution=[64 64 16];%desired reconstrution spatial resolution
recon_details.fov=[0.192 0.192 0.048];%desired reconstrution spatial resolution
recon_details.voxel_size=recon_details.fov./recon_details.recon_resolution;%mm
recon_details.dt = 5e-6;
recon_details.te = 0.0016;
recon_details.rawdata_filename=mreg_filename;
% recon_details.refname=[rawdata_path '/' ref_filename];
recon_details.DORK_frequency=zeros(1,1000);
recon_details.DORK_phi_offset=zeros(1,1000);

% recon_details.penalty.norm_string='L2-norm';
% recon_details.penalty.norm=@L2Norm;
% recon_details.penalty.lambda=0.2;
recon_details.penalty.norm_string='L1-norm';
recon_details.penalty.norm=@L1Norm;
recon_details.penalty.lambda=5e-6;
% 
% recon_details.penalty.operator_string='identity';
% recon_details.penalty.operator.handle = @identityOperator;
% recon_details.penalty.operator.args = {};
% recon_details.pname=[recon_path '/' recon_details.penalty.norm_string '_' ...
%     num2str(recon_details.penalty.lambda)];
recon_details.pname=[recon_path];

recon_details.penalty.operator_string='TV';
recon_details.penalty.operator(1).handle = @finiteDifferenceOperator;
recon_details.penalty.operator(1).args = {1};
recon_details.penalty.operator(2).handle = @finiteDifferenceOperator;
recon_details.penalty.operator(2).args = {2};
recon_details.penalty.operator(3).handle = @finiteDifferenceOperator;
recon_details.penalty.operator(3).args = {3};
% recon_details.pname=[pathname recon_details.penalty.norm_string '_' ...
%     num2str(recon_details.penalty.lambda)];

recon_details.offresonance_correction_flag=1;%off-resonance correction
recon_details.max_iterations=20;
recon_details.recon_output_format='mat';
recon_details.global_frequency_shift=0;
recon_details.tolerance=1e-5;
recon_details.timeframes=1:length(recon_details.DORK_frequency);
recon_details.coil_compression=0;%coil array compression
recon_details.num_coil=16;
recon_details.acceleration_rate=0.1;%tPCR acceleration rate, 0.1 means 10 times acceleration,..
%0.01 means 100 times acceleration

mkdir(recon_details.pname);
save([recon_details.pname '/recon_details.mat'],'recon_details');

%% calculate the sensitivity map and field map
% if ~exist([recon_details.pname '/data.mat'],'file')
%     [twx]=mapVBVD('meas_MID00081_FID198812_gre_field_mapping.dat');
%     rawdata=permute(rawdata,[1 3 2]);
%     rawdata=reshape(rawdata,[64 64 8 2 20]);
%     rawdata=permute(rawdata,[1 2 3 5 4]);
%     cmaps=ifff(ifff(ifff(rawdata,1),2),3);
%     % recon=imrotate(recon,90);
%     % recon=flip(recon);
%     % calculate sos and mask
%     anatomical = sqrt(sum(abs(cmaps(:,:,:,:,1)).^2,4));
%     mask      = anatomical > 0.05*max(anatomical(:));
% 
%     % calculate coil sensitivites using adapt3D
%     smaps = coilSensitivities(cmaps(:,:,:,:,1),'adapt');
%     %wwmap
%     te = [10 12]*1e-3;
%     delta_te = (te(2)-te(1)); %[s]
%     pmaps = phasemaps(cmaps);
%     wmap = fieldmap(angle(pmaps(:,:,:,2)./pmaps(:,:,:,1)),mask,anatomical,delta_te);
%     wmap = mri_field_map_reg3D(pmaps,te,'l2b',-1,'winit',wmap,'mask',mask);
%     % 3d inital image
%     [rawdata]=loadData('meas_MID00473_FID146187_pulseq.dat');
%     rawdata=permute(rawdata,[1 3 2]);
%     rawdata=reshape(rawdata,[64 64 64 20]);
%     % smaps=flip(data.smaps);
%     % smaps=imrotate(smaps,-90);
%     F=FTOperator_c2(ones(data.dim),gpuArray(data.smaps));
%     recon_iter = gather(regularizedReconstruction(F,gpuArray(double(rawdata)),'maxit',20,...
%         'verbose_flag', 0,'tol',1e-5));
%     % recon_iter=imrotate(recon_iter,90);
%     % recon_iter=flip(recon_iter);
%     data.recon_full=recon_iter.*exp(-1i*data.wmap*0.0361);
% 
%     data.anatomical=anatomical;
%     data.smaps=smaps;
%     data.wmap=wmap;
%     data.trajectory=traj;
%     data.mask=mask;
%     data.dim=size(anatomical);
%     save([recon_details.pname '/data'],'data');
% else
%     load([recon_details.pname '/data'],'data');
%     save([recon_details.pname '/data'],'data');
% end
%%
% if ~exist([recon_details.pname '/data.mat'],'file')
%     dirs=dir('*.dat');
%     if dirs(1).name(25)=='M'
%         ref_filename=dirs(2).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
%     else
%         ref_filename=dirs(1).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
%     end
% 
%     data=sensitivity_field_map(header.rawdata_filename,ref_filename,recon_details,traj);
%     save([recon_details.pname '/data'],'data');
% else
%     load([recon_details.pname '/data'],'data');
%     data.trajectory =traj;
%     save([recon_details.pname '/data'],'data');
% end

%%
if ~exist([recon_details.pname '/data.mat'],'file')
    dirs=dir('*.dat');
    if dirs(1).name(25)=='M'
        ref_filename=dirs(2).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
    else
        ref_filename=dirs(1).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
    end
    reference = loadReference(ref_filename);
    reference.raw.raw = reference.raw;
    reference = reference.raw;
    % calculate sos and mask
    reference.anatomical = sqrt(sum(abs(reference.cmaps(:,:,:,:,1)).^2,4));
    reference.mask      = reference.anatomical > 0.05*max(reference.anatomical(:));
    
    % calculate coil sensitivites using adapt3D
    reference.smaps = coilSensitivities(reference.cmaps(:,:,:,:,1),'adapt');
    %reference.SENSEreco = squeeze(sum(reference.cmaps.*repmat(conj(reference.smaps),[1 1 1 1 size(reference.cmaps,5)]),4));
    
    % w-map
    % calculate sos and mask and phase difference map from big Reference image
    
    
    % Get delta_te from Siemens header
    te = cell2mat(reference.raw.header.MeasYaps.alTE);
    te = te(1:size(reference.cmaps,5));
    
    %delta_te = (reference.raw.header.MeasYaps.alTE{2}-reference.raw.header.MeasYaps.alTE{1})/1e6; %[s]
    delta_te = (te(2)-te(1))/1e6; %[s]
    
    pmaps = phasemaps(reference.cmaps);
    
    try
        reference.wmap = fieldmap(angle(pmaps(:,:,:,2)./pmaps(:,:,:,1)),reference.mask,reference.anatomical,delta_te);
        
        if strcmp(reference.mode,'3d') || strcmp(reference.mode,'multi_slice') % 3D
            reference.wmap = mri_field_map_reg3D(pmaps,te/1e6,'l2b',-1,'winit',reference.wmap,'mask',reference.mask);
        else                                                                   % 2D
            reference.wmap = mri_field_map_reg(squeeze(pmaps),[reference.raw.header.MeasYaps.alTE{1}, reference.raw.header.MeasYaps.alTE{2}]/1e6,'l2b',-1,'winit',reference.wmap,'mask',reference.mask);
        end
    catch
        warning('Wasn''t able to calculate start value for fieldmap. Maybe FSL Toolbox is not installed or the path is not set correctly. Fessler toolbox is started without a startvalue. Unwrapping of the fieldmap might be inaccurate.')
        
        if strcmp(reference.mode,'3d') || strcmp(reference.mode,'multi_slice') % 3D
            reference.wmap = mri_field_map_reg3D(pmaps,[reference.raw.header.MeasYaps.alTE{1}, reference.raw.header.MeasYaps.alTE{2}]/1e6,'l2b',-1,'mask',reference.mask);
        else                                                                   % 2D
            reference.wmap = mri_field_map_reg(squeeze(pmaps),[reference.raw.header.MeasYaps.alTE{1}, reference.raw.header.MeasYaps.alTE{2}]/1e6,'l2b',-1,'mask',reference.mask);
        end
    end
    
    reference.wmap = smooth3(reference.wmap, 'gaussian', 5, 0.8);

    reference.cmap=[];
    reference.raw=[];
    data=reference;
    load([recon_details.pname '/ktraj']);
    % dk=1/(0.003*2);
    % ktraj{1}=(ktraj{1})'/dk*pi;
    % data.trajectory.trajectory{1}=ktraj(1:length(t_adc)/2,:);
    % data.trajectory.idx{1}=1:length(t_adc)/2;
    % data.trajectory.trajectory{2}=ktraj((1+length(t_adc)/2):end,:);
    % data.trajectory.idx{2}=1:length(t_adc)/2;
    % data.t{1}=t_adc(1:length(t_adc)/2);
    % data.t{2}=t_adc(1:length(t_adc)/2);
    data.trajectory.trajectory=ktraj;
    data.trajectory.idx=idx;
    data.t=T;
    save([recon_details.pname '/data'],'data');
else
    load([recon_details.pname '/data']);
    load([recon_details.pname '/ktraj']);
    data.trajectory.trajectory=ktraj;
    data.trajectory.idx=idx;
    data.t=T;
    save([recon_details.pname '/data'],'data');
    
end

%% Standard Reconstruction
% recon_correction_phase_dp_fista3(1,'');
% recon_pulseq(1,'',1:1);
% sr_recon(recon_details.timeframes,recon_details.pname);
% 
% %when you want to use slurm to accelerate
% % slurm_submit_mreg(recon_details.timeframes,1,recon_details.pname,'job_slurm',50,1,'');
% 
% %% combine 3d images to 4d
% tpcr_combination(recon_details.timeframes,recon_details.pname,1,'sr');
% 
% 
% 
% %% tPCR
% 
% %% step 1: decomposition
% n_seg=recon_details.timeframes(1:1000:end);
% for n=1:length(n_seg)
%     tframe=recon_details.timeframes(n:min(n+999,length(recon_details.timeframes)));
%     subfolder1=[recon_details.pname '/tpcr/' num2str(n) '/'];
%     if ~exist([subfolder1 'SV/Vt.mat'],'file')
%         tpcr_decomposition(tframe,subfolder1);
%       %when you want to use slurm to accelerate
% %         slurm_submit_mreg([n recon_details.timeframes(end)],1,subfolder1,'job_slurm',2,3,'ram');
%     end
% end
% 
% %% step 2: reconstruct principal components - segmented
% n_seg=recon_details.timeframes(1:1000:end);
% for n=1:length(n_seg)
%     tframe=recon_details.timeframes(n:min(n+999,length(recon_details.timeframes)));
%     subfolder1=[recon_details.pname '/tpcr/' num2str(n) '/'];
%     while ~exist([subfolder1 'kbase/rawlevel_' num2str(floor(length(tframe)*recon_details.acceleration_rate)+1) '.mat'],'file')
%         printf('wait...')
%         pause(180);
%     end
%     tpcr_recon(1:max([1,length(tframe)*recon_details.acceleration_rate]),subfolder1);
%   %when you want to use slurm to accelerate
% %     slurm_submit_mreg(1:max([1,length(tframe)*accelerate]),1,subfolder1,'job_slurm2',2,2,'ram');
% end
% 
% %% step 3: recombination
%  
% fname = fullfile([recon_details.pname '/tpcr/4D_' num2str(recon_details.acceleration_rate*100) '%.nii']);
% if ~exist(fname,'file')
%     n_seg=recon_details.timeframes(1:1000:end);
%     for n=1:length(n_seg)
%         tframe=recon_details.timeframes(n:min(n+999,length(recon_details.timeframes)));
%         subfolder1=[recon_details.pname '/tpcr/' num2str(n) '/'];
%         tpcr_combination(tframe,subfolder1,recon_details.acceleration_rate,'tpcr')
%     end
% 
%     for n=1:length(n_seg)
%         subfolder1=[recon_details.pname '/tpcr/' num2str(n) '/nifti/'];
%         if n==1
%             temp=load_nii([subfolder1 '4D_' num2str(recon_details.acceleration_rate*100) '%.nii']);
%             delete([subfolder1 '4D_' num2str(recon_details.acceleration_rate*100) '%.nii']);
%             image4d=temp.img;
%         else
%             temp=load_nii([subfolder1 '4D_' num2str(recon_details.acceleration_rate*100) '%.nii']);
%             delete([subfolder1 '4D_' num2str(recon_details.acceleration_rate*100) '%.nii']);
%             image4d(:,:,:,(end+1):(end+size(temp.img,4)))=temp.img;
%         end
%     end
%     trad=make_nii(single(image4d),[3 3 3]);
%     save_nii(trad,fname);
% end

