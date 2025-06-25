function recon=sr_recon_mshot(tframes,pathname,filename,seg,shot,norm,lambda)
% reconstruction with SR
%     tframes: the time frames to reconstruct
%     pathname: the folder where the rawdatas store and the reconstructed images save
% 19.11.2020
% Fei Wang

    
    
    % load data which includes the sensitivity map, field map and
    % trajectory
    data=load([pathname '/data.mat']);
    data=data.data;
    recon_details=load([pathname '/recon_details.mat']);
    recon_details=recon_details.recon_details;
    dt=5e-6;
    ss=size(data.anatomical);
    if recon_details.coil_compression==1
        %% coil compression
        smaps_mean=mean(data.smaps,4);
        data.smaps=data.smaps-repmat(smaps_mean,[1 1 1 size(data.smaps,4)]);
        smaps2d=reshape(data.smaps,[prod(ss) size(data.smaps,4)]);
        [Vt,~]=eig(smaps2d'*smaps2d);
        Vt=Vt(:,end:-1:(end-recon_details.num_coil+1));
        Ut=smaps2d*Vt;    
        data.smaps=reshape(Ut,[ss(1) ss(2) ss(3) recon_details.num_coil])+repmat(smaps_mean,[1 1 1 recon_details.num_coil]);
    end
    % make folders for storation
    filename_save=[recon_details.pname '/sr'];
    if strcmp(recon_details.recon_output_format,'mat')
        if ~exist([filename_save '/mat'],'dir')
            mkdir([filename_save '/mat']);
        end
    end
    if strcmp(recon_details.recon_output_format,'nifti')
        mkdir([filename_save '/nifti']);
    end
    
    for i=1:length(filename)
        [rawdata,header]=loadData([pathname '/' filename{i}],1);
        trajname=header.hdr.MeasYaps.sWipMemBlock.tFree;
%         trajectory=loadTrajectory(trajname{1},[],[-3.7 -3 -3]);%trajectory files
        trajectory=loadTrajectory(trajname{1},[],[0 0 0]);%trajectory files
        traj0{i}=trajectory.trajectory;
        traj_idx0{i}=trajectory.idx;
        if i==1
            traj=trajectory.trajectory(shot);
            traj_idx = trajectory.idx(shot);
        else
            traj(end+(1:length(trajectory.trajectory)))=trajectory.trajectory(shot);
            traj_idx(end+(1:length(trajectory.idx)))=trajectory.idx(shot);
        end
    end
    if isfield(header,'inPlaneRot') && abs(header.inPlaneRot-pi/2)<1e-5
        for ii=1:length(traj)
            traj{ii} = traj{ii}(:,[2 1 3]);
            traj{ii}(:,2) = -traj{ii}(:,2);
            recon_details.DeltaT(ii) = header.te(1) + traj_idx{ii}(1)*dt;
        end
    end
    n=0;
    for ii=1:length(traj)
        n=n+1;
        K{ii}=traj{ii}(traj_idx{ii},:);
        T{ii}=header.te(1)+traj_idx{ii}*dt;
        if n==1
            K1=K{ii};
        else
            K1(end+(1:length(K{ii})),:)=K{ii};
        end
    end

    if gpuDeviceCount
        data.smaps=gpuArray(single(data.smaps));
%         data.wmap=gpuArray(single(data.wmap));
    end
    % make forward operator
    if recon_details.offresonance_correction_flag==1
%         Fg=orc_segm_nuFTOperator_multi(K,size(data.wmap),data.smaps,data.wmap,5e-6,2,recon_details.DeltaT);
        Fg=orc_segm_nuFTOperator_multi_sub(K,size(data.wmap),data.smaps,data.wmap,dt,seg,T);
    else
        Fg=nuFTOperator(K1,size(data.anatomical),data.smaps);
    end


    % assemble arguments for regularizedReconstruction
%     lengthP = 0;
%     for k=1:length(recon_details.penalty)
%         lengthP = lengthP + length(recon_details.penalty(k).operator) + 2;
%     end
% 
%     P = cell(1,lengthP);
%     counter = 1;
%     for k=1:length(recon_details.penalty)
%         P{counter} = recon_details.penalty(k).norm;
%         counter = counter + 1;
%         P{counter} = recon_details.penalty(k).lambda;
%         counter = counter + 1;
%         for n=1:length(recon_details.penalty(k).operator)
%             P{counter} = recon_details.penalty(k).operator(n).handle(recon_details.penalty(k).operator(n).args{:});
%             counter = counter + 1;
%         end
%     end


    image       = data.anatomical;
    image(isnan(image))=0;
    image=image/max(abs(image(:)));

    sz=size(image);

    g = Gradient3D_forward_constant_unitstep(image);
    ng = Norm_n(g,0,4);
    eta = 1e-2;
    id_mask=ones(sz)*0.0;
    t1=smooth3(image,'box',3);
    id_mask(t1>0.05*max(abs(t1(:))))=1;
%     id_mask=smooth3(id_mask,'box',3);

    lengthP = 0;
    P = cell(1,lengthP);
    counter = 1;

%     operator(1).handle = @finiteDifferenceOperator;
%     operator(1).args = {1};
%     operator(2).handle = @finiteDifferenceOperator;
%     operator(2).args = {2};
%     operator(3).handle = @finiteDifferenceOperator;
%     operator(3).args = {3};
    if norm==1
        P{counter} = @L1Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        P{counter}=TV3d(sz);
    elseif norm==2
        P{counter} = @L1Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
%         for n=1:length(operator)
%             P{counter} = operator(n).handle(operator(n).args{:});
%             counter = counter + 1;
%         end

        sideinfo0 = eta./sqrt(ng.^2 + eta^2);
        sideinfo=col(sideinfo0);
        P{counter}=wTV3d(sz,sideinfo);
    elseif norm==3
        P{counter} = @L1Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        sideinfo0 = Vec3D_mult(g,1./sqrt(ng.^2 + eta.^2));
        sideinfo=cell(length(sz),1);
        for jj=1:length(sz)
            sideinfo{jj}=col(sideinfo0(:,:,:,jj));
        end
        P{counter}=dTV3d(sz,sideinfo);
    elseif norm==4
        P{counter} = @L1Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        sideinfo0 = Vec3D_mult(g,1./sqrt(ng.^2 + eta.^2));
        sideinfo=cell(length(sz),1);
        for jj=1:length(sz)
            sideinfo{jj}=col(sideinfo0(:,:,:,jj));
        end
        P{counter}=dTV3d(sz,sideinfo);
        
        counter = counter + 1;
        P{counter} = @L1Norm2;
        counter = counter + 1;
        P{counter} = lambda(2);
        counter = counter + 1;
        sideinfo0 = eta./sqrt(id_mask.^2 + eta^2);
        sideinfo=col(sideinfo0);
        P{counter}=wID(sz,sideinfo);
    elseif norm==5
        P{counter} = @L1Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        P{counter}=TV3d(sz);

        counter = counter + 1;
        P{counter} = @L1Norm2;
        counter = counter + 1;
        P{counter} = lambda(2);
        counter = counter + 1;
        sideinfo0 = eta./sqrt(id_mask.^2 + eta^2);
        sideinfo=col(sideinfo0);
        P{counter}=wID(sz,sideinfo);
    elseif norm==6
        P{counter} = @L2Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        P{counter}=ID(sz);
    elseif norm==7
        P{counter} = @L2Norm2;%recon_details.penalty(k).norm;
        counter = counter + 1;
        P{counter} = lambda(1);
        counter = counter + 1;
        P{counter}=ID(sz);

        counter = counter + 1;
        P{counter} = @L2Norm2;
        counter = counter + 1;
        P{counter} = lambda(2);
        counter = counter + 1;
        sideinfo0 = eta./sqrt(id_mask.^2 + eta^2);
        sideinfo=col(sideinfo0);
        P{counter}=wID(sz,sideinfo);
    end


    % reconstruction  
    for k=tframes
        n=0;
        for i=1:length(traj)
%             for ii=1:length(traj{i})%length(traj)
                n=n+1;
                [~,header]=loadData([pathname '/' filename{1}],1);
                recon_details.rawdata_filename=header.rawdata_filename;
%                 DORK_k0 = 82;%traj.idx{1}(idx_k0);
%                 recon_details.DORK_k0=DORK_k0;
%                 recon_details.DORK_frequency=[];
%                 [recon_details.DORK_frequency,recon_details.DORK_phi_offset,recon_details.dw,recon_details.dphi,...
%                     recon_details.dw0,recon_details.dphi0] = DORK_frequency...
%                     (recon_details.rawdata_filename,recon_details.DORK_k0,length(traj0{i}));

                interleaf_idx=n;
                kk=k+shot(mod(i-1,length(shot))+1)-1;
                [rawdata, header] = loadData([pathname '/' filename{floor((i-1)/length(shot))+1}],kk);
                shift_raw_data
                %% coil compression
                if recon_details.coil_compression==1
                    rawdata_mean=mean(rawdata,2);
                    rawdata=(rawdata-repmat(rawdata_mean,[1 size(rawdata,2)]))*Vt;
                    rawdata=rawdata+repmat(rawdata_mean,[1 recon_details.num_coil]);
                end
                if n==1
                    rawdata2d=rawdata;
                else
                    rawdata2d((end+1):(end+length(rawdata)),:)=rawdata;
                end
%             end
        end
%         if gpuDeviceCount
%             rawdata2d=gpuArray(single(rawdata2d));
%         end
        if strcmp(recon_details.recon_output_format,'mat')
            fname0=[filename_save '/mat/' num2str(k) '.mat'];
%                 if ~exist(fname0,'file')
                k
%                 [recon] = gather(regularizedReconstruction(Fg,double(rawdata2d),P{:}, ...
%                     'tol',recon_details.tolerance, ...
%                     'maxit',recon_details.max_iterations, ...
%                     'verbose_flag', 1));

                recon=gather(regularizedReconstruction2(Fg,rawdata2d,P{:},'maxit',recon_details.max_iterations,'verbose_flag', 1,'tol',0e-6,'z0',[]));

            fname = fullfile([filename_save '/mat/' num2str(k) '.mat']);
                save(fname,'recon');
%                 end
        else
            fname0=[filename_save '/nifti/real/' num2str(k) '.nii'];
            if ~exist(fname0,'file')
                k
                [recon] = gather(regularizedReconstruction(Fg,rawdata2d(:),P{:}, ...
                    'tol',recon_details.tolerance, ...
                    'maxit',recon_details.max_iterations, ...
                    'verbose_flag', 1));

                if ~exist([filename_save '/nifti/real'],'dir')
                    mkdir([filename_save '/nifti/real']);
                end
                if ~exist([filename_save '/nifti/imag'],'dir')
                    mkdir([filename_save '/nifti/imag']);
                end
                fname = fullfile([filename_save '/nifti/real/' num2str(k) '.nii']);
                trad=make_nii(single(real(recon)),[3 3 3]);
                save_nii(trad,fname);
                fname = fullfile([filename_save '/nifti/imag/' num2str(k) '.nii']);
                trad=make_nii(single(imag(recon)),[3 3 3]);
                save_nii(trad,fname);
            end
        end
    end

    function shift_raw_data

    %% DORK and off-resonance correction
    t = header.te(1)*ones(size(rawdata,1),1) + (0:size(rawdata,1)-1)'*header.dwelltime;
    % which is caused by heating of the system if without frequency adjustment,
    %here 60 is estimated by eyes
    if isempty(recon_details.DORK_frequency)
        freq_offset = recon_details.global_frequency_shift;
        phi_offset = 0;
    else
        freq_offset = recon_details.DORK_frequency(kk) + recon_details.global_frequency_shift;
        phi_offset = recon_details.DORK_phi_offset(kk);
    end
%     rawdata = rawdata .* repmat(exp(-1i*(phi_offset+freq_offset.*t)), [1 size(rawdata, 2)]);

    %% Adding additional Phase to the data for data shifting
    rawdata = rawdata(traj_idx{interleaf_idx},:) .* repmat(exp(1i*traj{interleaf_idx}(traj_idx{interleaf_idx},[2 1 3])...
        *(data.shift)'), [1 size(rawdata, 2)]);%from 0 to 0.5
    end
end
