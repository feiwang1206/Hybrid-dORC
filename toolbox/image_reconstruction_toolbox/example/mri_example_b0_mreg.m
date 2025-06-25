% mri_example_b0.m
% Example of field-corrected MR reconstruction for a spiral k-space trajectory.
% Copyright 2005-7-29, Jeff Fessler, The University of Michigan

% zmap
clear
% load('/home/ubuntu/Documents/work/train/01/data.mat');
% omega=data.trajectory.trajectory{1}(data.trajectory.idx{1},:);
% fmap=data.wmap;
% dt=5e-6;
load('/home/ubuntu/Documents/work/mreg_example_data.mat');
mask=true(param.dim);
% if ~isvar('zmap'), printm 'read fieldmap'
% % 	[fmap mask_orig] = mri_phase_denoise('test');
% 	fmap = fmap / (2*pi); % 2ms echo time diff.
% 	[nx ny nz] = size(fmap);
% % 	mask = ellipse_im(nx, ny, [0 0 27 35 0 1], 'dx', 1, 'oversample', 3) > 0;
%     mask=(ones(nx,ny,nz)==1);
%     
% 
% 	zmap = 0 + (2i*pi) * fmap;
% % prompt
% end

% if ~isvar('xtrue'), printm 'setup object'
% 
% 	ig = image_geom('nx', 64, 'nz', 48, ...
% 		'offsets', 'dsp', ... % [-n/2:n/2-1] for mri
% 		'fov', [19.2 19.2 15]); % 20 cm transaxial FOV
% % 	xs = mri_objects('fov', ig.fovs, 'test4'); % object model
% % 	xtrue = xs.image(ig.xg, ig.yg, ig.zg); % samples of continuous-space
%     xtrue=imresize3D(data.anatomical,ig.dim);
% % prompt
% end

ig.fovs=[19.2 19.2];
ig.dim=[64 64];
% kspace trajectory
	f.traj = 'cartesian';
% 	f.traj = 'spiral3';
% 	f.traj = 'radial';
% 	f.dens = {'voronoi'};
	f.dens = {};
% 	cpu etic
	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, ig.dim, ig.fovs, f.dens);
% omega=param.traj{1}(param.traj_idx{1},:);
% kspace = zeros(size(omega));
% for id=1:length(param.dim)
%     dx = ig.fovs(id) / param.dim(id);
%     kspace(:,id) = omega(:,id) / (2*pi) / dx;
% end
% kspace=[kspace(:,2) -kspace(:,1) kspace(:,3)];
% ti=param.T{1};
% 
N=[64 64];
mask=true(N);
	f.nufft = {N, [5 5], 2*N, N/2, 'minmax:kb'};
	Gn = Gmri(kspace, mask, 'fov', ig.fovs, 'basis', {'dirac'}, ...
		'nufft', f.nufft);

% base NUFFT
if ~isvar('Gn'), printm 'Gn'
% 	f.nufft = {param.dim, [5 5 5], 2*param.dim, param.dim/2, 'table', 2^12, 'minmax:kb'};
	f.nufft = {param.dim, [5 5 5], 2*param.dim, param.dim/2, 'minmax:kb'};
	Gn = Gmri(kspace, mask, 'fov', ig.fovs, 'basis', {'dirac'}, ...
		'nufft', f.nufft);
end

yi=param.rawdata;
%%
% penalty
%
if ~isvar('R'), printm 'R'
	% scale beta by fov^4 since A'A and 2D.
	f.beta = 2^-28 * size(omega,1) * ig.fov(1)^4;
	R = Robject(mask, 'potential', 'quad', 'beta', f.beta);

end
%%
% system
% if ~isvar('Gm'), printm 'Gmri'
	L = 9;
	Gm = feval(Gn.arg.new_zmap, Gn, ti, -1i*param.field, L);
% end

% if 0 && ~isvar('Tm'), printm 'Tm: Gmri gram'
% if ~isvar('Tm'), printm 'Tm: Gmri gram'
	Tm = build_gram(Gm, 1);
% return
% end
% %%
% smaps=imresize4D(data.smaps,ig.dim);
% size_s=size(smaps);
% coil_num=8;
% smaps2d=reshape(smaps,[prod(size_s(1:3)) size(smaps,4)]);
% [Vt,~]=eig(smaps2d'*smaps2d);
% Vt=Vt(:,end:-1:(end-coil_num+1));
% Ut=smaps2d*Vt;    
% smaps=reshape(Ut,[size(data.wmap) coil_num]);

% % if ~isvar('Gb'), printm 'Gb object with sense maps within'
	for ic=1:size(param.smaps,4)
		tmp = param.smaps(:,:,:,ic);
		Gc{ic} = Gm * diag_sp(tmp(mask)); % cascade
%         Tc = diag_sp(conj(tmp(ig.mask)))*(build_gram(Gm,1)* diag_sp(tmp(ig.mask)));
        Tc = diag_sp(conj(tmp(mask)))*(Tm* diag_sp(tmp(mask)));
        if ic>1
            Tb=fatrix_plus(Tb,Tc);
        else
            Tb=Tc;
        end
	end
	Gb = block_fatrix(Gc, 'type', 'col'); % [G1; G2; ... ]
% % end

%%
% if ~isvar('xcg1'), printm 'xcg1 iterative'
	f.niter = 10;
%     yi=Gb*xtrue(mask);
    
	xinit = gpuArray(zeros(param.dim));%xcp0/prod(ig.dim);
	cpu tic
	tic;xcg1 = qpwls_pcg1(xinit(mask), Gb, 1, yi(:), R.C, 'niter', f.niter);toc
	xcg1 = embed(xcg1(:,end), mask);
%     nrms(xcg1, xtrue)
figure,imagesc(array2mosaic(abs(gather(reshape(xcg1,param.dim)))));colormap gray;colorbar
%%
    bb = Gb' * yi(:);
    tic;xcg2 = qpwls_pcg2(xinit(mask), Tb, bb, R.C, 'niter', f.niter);toc
    xcg2 = embed(xcg2(:,end), mask);
figure,imagesc(array2mosaic(abs(gather(reshape(xcg2,param.dim)))));colormap gray;colorbar
%%
F=orc_segm_nuFTOperator_multi_sub({omega},param.dim,gpuArray(param.smaps),gpuArray(param.field),param.dt,1,{param.T{1}});
FT=orc_segm_nuFTOperator_multi_sub_T({omega},param.dim,gpuArray(param.smaps),gpuArray(param.field),param.dt,1,{param.T{1}});
%%
tic;recon{1} = regularizedReconstruction(F,yi(:),'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',(zeros(param.dim)));toc
figure,imagesc(array2mosaic(abs(gather(reshape(recon{param.shot},param.dim)))));colormap gray;colorbar
% %%
bb=gather(F'*param.rawdata(:));
tic;recon{1} = regularizedReconstruction_toep(FT,bb,'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',(zeros(param.dim)));toc
figure,imagesc(array2mosaic(abs(gather(reshape(recon{param.shot},param.dim)))));colormap gray;colorbar
%%
tic;param.alpha = gather(PowerIteration_mreg(F, gpuArray(ones(param.dim))));toc
init=double(reshape(gather(F'*param.rawdata(:)),param.dim));
param.W = DWT(3*[1,1,1],param.dim, true, 'haar');
%%
tic;recon{param.shot} = gather(ReconWavFISTA_mreg(init, F, 0.01*max(abs(col(init))), param.W, param.alpha, init, 50, true));toc
figure,imagesc(array2mosaic(abs(gather(reshape(recon{param.shot},param.dim)))));colormap gray;colorbar
% %%
% %%
% tic;param.alpha = gather(PowerIteration_mreg_toep(FT, gpuArray(ones(param.dim))));toc
% init=double(reshape(gather(F'*param.rawdata(:)),param.dim));
% param.W = DWT(3*[1,1,1],param.dim, true, 'haar');

tic;recon{param.shot} = gather(ReconWavFISTA_mreg_toep(init, FT, 0.01*max(abs(col(init))), param.W, param.alpha, init, 50, true));toc
figure,imagesc(array2mosaic(abs(gather(reshape(recon{param.shot},param.dim)))));colormap gray;colorbar


