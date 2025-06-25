%% set work time frames and path
clear
addpath(genpath('/home/ubuntu/Documents/MATLAB/mfiles'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/mreg_recon_tool'));
addpath(genpath('/home/ubuntu/Documents/MATLAB/MRIsimu&recon-keep'));
run('/home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox/setup');
% example setting
rawdata_path=pwd;
dirs=dir('*.dat');
if dirs(1).name(25)=='M'
    mreg_filename=dirs(1).name;%mreg rawdata file name, e.g. meas_MID00133_FID01338_MREG.dat
else
    mreg_filename=dirs(2).name;%mreg rawdata file name, e.g. meas_MID00133_FID01338_MREG.dat
end
recon_path=pwd;%
[~,header]=loadData(mreg_filename,1);
trajname=header.hdr.MeasYaps.sWipMemBlock.tFree{1};
% twx = mapVBVD(mreg_filename);
% load your trajectory and correct gradient delay
mreg_path=which('mreg_recon_tool');
traj=loadTrajectory([mreg_path(1:(end-33)) 'grad_files/' trajname],[],[-3 -3 -3.7]);%trajectory files
% traj=loadTrajectory([mreg_path(1:(end-33)) 'grad_files/' trajname{1}]);%trajectory files

% mreg_filename='meas_MID00474_FID146188_pulseq.dat';

% Set reconstruction parameters
[~,header]=loadData([rawdata_path '/' mreg_filename],1);
recon_details.header=header;
% recon_details.recon_resolution=[64 64 64];%desired reconstrution spatial resolution
% recon_details.fov=[0.192 0.192 0.192];%desired reconstrution spatial resolution
recon_details.recon_resolution=[64 64 16];%desired reconstrution spatial resolution
recon_details.fov=[0.192 0.192 0.048];%desired reconstrution spatial resolution
recon_details.voxel_size=recon_details.fov./recon_details.recon_resolution;%mm
recon_details.dt = 5e-6;
recon_details.te = header.te(1);
recon_details.rawdata_filename=header.rawdata_filename;
% recon_details.refname=[rawdata_path '/' ref_filename];
[~,idx_k0]=min(makesos(traj.trajectory{1}(traj.idx{1},:),2));
DORK_k0 = traj.idx{1}(idx_k0);
recon_details.DORK_k0=DORK_k0;
recon_details.DORK_frequency=[];
[recon_details.DORK_frequency,recon_details.DORK_phi_offset,recon_details.dw,recon_details.dphi,...
    recon_details.dw0,recon_details.dphi0] = DORK_frequency...
    (recon_details.rawdata_filename,recon_details.DORK_k0,length(traj.trajectory));

% linear_trend = [(1:length(recon_details.DORK_frequency))' ones(length(recon_details.DORK_frequency),1)];
% beta = linear_trend(1:length(recon_details.DORK_frequency),:) \ recon_details.DORK_frequency(1:length(recon_details.DORK_frequency));
% temp = linear_trend(1:length(recon_details.DORK_frequency),:)*beta;
% sd=sqrt((recon_details.DORK_frequency-temp).^2);
% idx=find(sd>4*mean(sd));
% recon_details.DORK_frequency(idx)=recon_details.DORK_frequency(idx+1);
% linear_trend = [(1:length(recon_details.DORK_phi_offset))' ones(length(recon_details.DORK_phi_offset),1)];
% beta = linear_trend(1:length(recon_details.DORK_phi_offset),:) \ recon_details.DORK_phi_offset(1:length(recon_details.DORK_phi_offset));
% temp = linear_trend(1:length(recon_details.DORK_phi_offset),:)*beta;
% sd=sqrt((recon_details.DORK_phi_offset-temp).^2);
% idx=find(sd>4*mean(sd));
% recon_details.DORK_phi_offset(idx)=recon_details.DORK_phi_offset(idx+1);


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

% [~,header]=loadData([rawdata_path '/' mreg_filename],1);
% recon_details.rawdata_filename2=header.rawdata_filename;
% [rawdata,header]=loadData(mreg_filename,1);
% trajname=header.hdr.MeasYaps.sWipMemBlock.tFree;
% traj=loadTrajectory([mreg_path(1:(end-33)) 'grad_files/' trajname{1}],[],[-3 -3 -3.7]);%trajectory files
% [~,idx_k0]=min(makesos(traj.trajectory{1}(traj.idx{1},:),2));
% DORK_k0 = traj.idx{1}(idx_k0);
% recon_details.DORK_k0=DORK_k0;
% [recon_details.DORK_frequency2,recon_details.DORK_phi_offset2,recon_details.dw2,recon_details.dphi2,...
%     recon_details.dw02,recon_details.dphi02] = DORK_frequency2...
%     (recon_details.rawdata_filename2,recon_details.DORK_k0,length(traj.trajectory));

mkdir(recon_details.pname);
save([recon_details.pname '/recon_details.mat'],'recon_details');

%% calculate the sensitivity map and field map
% for ii=1:length(traj.trajectory)
%     for jj=1:size(traj.trajectory{1},2)
%         traj.trajectory{ii}(:,jj)=traj.trajectory{ii}(:,jj)*(recon_details.voxel_size(jj)/0.003);
%     end
% end
if ~exist([recon_details.pname '/data.mat'],'file')
    dirs=dir('*.dat');
    if dirs(1).name(25)=='M'
        ref_filename=dirs(2).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
    else
        ref_filename=dirs(1).name;%reference rawdata file name, e.g. meas_MID00100_FID01305_gre_field_mapping.dat
    end
        
    data=sensitivity_field_map(header.rawdata_filename,ref_filename,recon_details,traj);
    save([recon_details.pname '/data'],'data');
else
    load([recon_details.pname '/data'],'data');
    data.trajectory =traj;
    save([recon_details.pname '/data'],'data');
end


%% Standard Reconstruction
    addpath(genpath('/home/ubuntu/Documents/work/train/diffisionnet'));
    addpath /home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox
% recon_correction_phase_dp_fista3(1,'');
recon_diff(1,'',[1],'',0);
recon_diff(1,'',[1:2],'',0);
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

