% mri_example_3d.m
% Example illustrating 3D regularized iterative reconstruction for MRI
% from nonuniform k-space samples.
% (This example does not include field inhomogeneity or relaxation.)
% Copyright 2004-12-16, Jeff Fessler, The University of Michigan

% this is an "honest" simulation where the Fourier data is calculated
% analytically from a continuous-space object model, even though the
% reconstructions are all discrete-space.
clear
load('/home/ubuntu/Documents/work/train/01/data.mat');
omega=data.trajectory.trajectory{1}(data.trajectory.idx{1},:);
dt=5e-6;

if ~isvar('xtrue'), printm 'setup object'

	ig = image_geom('nx', 64, 'nz', 48, ...
		'offsets', 'dsp', ... % [-n/2:n/2-1] for mri
		'fov', [19.2 19.2 15]); % 20 cm transaxial FOV

	xs = mri_objects('fov', ig.fovs, 'test4'); % object model
% 	xtrue = xs.image(ig.xg, ig.yg, ig.zg); % samples of continuous-space
    xtrue=imresize3D(data.anatomical,ig.dim);
% prompt
end

fmap=imresize3D(data.wmap,ig.dim);
f.dt=5e-6;
ti=dt*(1:length(omega));

smaps=imresize4D(data.smaps,ig.dim);
size_s=size(smaps);
coil_num=3;
smaps2d=reshape(smaps,[prod(size_s(1:3)) size(smaps,4)]);
[Vt,~]=eig(smaps2d'*smaps2d);
Vt=Vt(:,end:-1:(end-coil_num+1));
Ut=smaps2d*Vt;    
smaps=reshape(Ut,[size(data.wmap) coil_num]);
smaps=ones([ig.dim 1])/1;
f.ncoil=size(smaps,4);
%
% k-space trajectory, k-space data
%
if ~isvar('yi'), printm 'trajectory'
	f.traj = 'cartesian';
	f.traj = 'spiral3';
	f.traj = 'radial';
	f.dens = {'voronoi'};
	f.dens = {};
	cpu etic
% 	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, ig.dim, ig.fovs, f.dens);
    kspace = zeros(size(omega));
    for id=1:length(ig.dim)
        dx = ig.fovs(id) / ig.dim(id);
        kspace(:,id) = omega(:,id) / (2*pi) / dx;
    end
    
end


% create Gmri or Gnufft class object
if ~isvar('Gm'), printm 'system model'
	printm 'setup G objects'
	N = ig.dim;
	nufft_args = {N, 6*ones(size(N)), 2*N, N/2, 'table', 2^10, 'minmax:kb'};
	mask = true(N);
	clear N
%	f.basis = {'rect'}; % fix: using this has scaling issues
	f.basis = {'dirac'};
	Gm = Gmri(kspace, mask, ...
		'fov', ig.fovs, 'basis', f.basis, 'nufft', nufft_args);
    Gm = Gm.arg.new_zmap(Gm, ti, 1i*fmap, 6);
    Tm=Gmri_gram(Gm,1,1);
    Tm = Tm.arg.new_zmap(Tm, ti, 1i*fmap, 6);
end

if ~isvar('Gb'), printm 'Gb object with sense maps within'
	for ic=1:f.ncoil
		tmp = smaps(:,:,:,ic);
		Gc{ic} = Gm * diag_sp(tmp(ig.mask)); % cascade
        Tc = diag_sp(conj(tmp(ig.mask)))*(build_gram(Gm,1)* diag_sp(tmp(ig.mask)));
        % Tc = diag_sp(conj(tmp(ig.mask)))*(Tm* diag_sp(tmp(ig.mask)));
        if ic>1
            Tb=fatrix_plus(Tb,Tc);
        else
            Tb=Tc;
        end
	end
	Gb = block_fatrix(Gc, 'type', 'col'); % [G1; G2; ... ]
end
% Gbz = Gb.arg.new_zmap(Gb, ti, 1i*fmap,2);
% Tbz = Tb.arg.new_zmap(Tb, ti, 1i*fmap,2);
%%
ytrue = Gb * xtrue(ig.mask);

% add noise
randn('state', 0)
yi = ytrue + 0 * randn(size(ytrue));
% todo: visualize data...

xcp = Gb' * (yi)/prod(ig.dim);
xcp = embed(xcp, mask);
if ~isvar('R'), printm 'regularizer'
	beta = 2^-7 * size(omega,1);	% good for quadratic
	R = Reg1(ig.mask, 'beta', beta);
	if 0 % check resolution: [1.1 1.1 1]
		qpwls_psf(Gb, R, 1, ig.mask, 1, 'fwhmtype', 'profile');
	return
	end
end
%
	niter = 10;
% 	ytmp = yi;% / abs(ig.dx*ig.dy*ig.dz); % trick: analytical data vs DSFT
% 	tic;xpcg = qpwls_pcg1(1*xcp(ig.mask), Gb, 1, ytmp(:), R.C, 'niter', niter);toc
% % 	tic;xpcg = regularizedReconstruction(Gb,yi(:),'maxit',niter,'verbose_flag', 0,'tol',1e-5,'z0',xcp(ig.mask));toc
% 	xpcg = ig.embed(xpcg);


%
%     ytmp = yi / abs(ig.dx*ig.dy*ig.dz); % trick: analytical data vs DSFT
%     bb = Gb' * yi;%ytmp(:);
% % 	if ~isvar('xcg2'), printm 'xcg2'
% 		tic;xcg2 = qpwls_pcg2(1*xcp(ig.mask), Tb, bb, R.C, 'niter', niter);toc
% %         tic;xcg2 = regularizedReconstruction_toep(Gb,Tb,bb(:),'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',xcp(ig.mask));toc
% 		xcg2 = embed(xcg2(:,end), mask);
% % 	end
% 
% nrms(xcp/prod(ig.dim), xtrue)
% nrms(xpcg, xtrue)
% nrms(xcg2, xtrue)
%%
% Fg=orc_segm_nuFTOperator_multi_sub({omega},ig.dim,gpuArray(double(smaps)),gpuArray(double(fmap)),dt,6,{ti});
% rawdata=Fg*xtrue;
% tic;xcg3 = regularizedReconstruction(Fg,rawdata,'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',[]);toc
% nrms(xcg3, xtrue)
% %%
% bb=Fg'*rawdata;
% FgT=orc_segm_nuFTOperator_multi_sub_T({omega},ig.dim,gpuArray(double(smaps)),gpuArray(double(fmap)),dt,6,{ti});
% tic;xcg4 = regularizedReconstruction_toep(FgT,bb,'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',[]);toc
% nrms(xcg4, xtrue)

%%
N = [64 64];
f.fov = 22; % cm
f.traj = 'spiral3'; 
f.dens = {'voronoi'};
f.niter=50;
[kspace omega wi_traj] = mri_trajectory(f.traj, {}, N, f.fov, f.dens);
omega2d=omega;
mask=true(ig.dim(1:2));
f.beta = 2^-28 * size(omega,1) * f.fov^4;
R = Robject(mask, 'potential', 'quad', 'beta', f.beta);

    N = ig.dim(1:2);
%     nufft_args = {N, 5*ones(size(N)), 2*N, N/2, 'table', 2^10, 'minmax:kb'};
    nufft_args = {N, 5*ones(size(N)), 2*N, N/2, 'minmax:kb'};
    mask = true(N);
    %	f.basis = {'rect'}; % fix: using this has scaling issues
    f.basis = {'dirac'};
        fovs=[22];
    kspace = zeros(size(omega2d));
    for id=1:size(omega2d,2)
        dx = fovs / ig.dim(id);
        kspace(:,id) = omega2d(:,id) / (2*pi) / dx;
    end
% omega2d=omega(6201:7200,1:2);
smaps2d=squeeze(smaps(:,:,24,:));
fmap2d=fmap(:,:,24);
ti2d=dt*(1:length(omega2d));
%     s.Gm = Gmri(kspace, mask, 'basis', f.basis, 'nufft', nufft_args);
%     s.Gm = s.Gm.arg.new_zmap(s.Gm, ti, 1i*gather(wmap), Ns);
    
	f.nufft = {N, [5 5], 2*N, N/2, 'table', 2^12, 'minmax:kb'};
	s.Gm = Gmri(kspace, mask, 'fov', fovs, 'basis', {'rect'}, ...
		'nufft', f.nufft);
    s.Gm = feval(s.Gm.arg.new_zmap, s.Gm, ti2d, 1i*gather(fmap2d), 6);
    
    s.Tm=build_gram(s.Gm,1);

xtrue2d=double(xtrue(:,:,24));

Fg2d=orc_segm_nuFTOperator_multi_sub({omega2d},ig.dim(1:2),(double(smaps2d)),(double(fmap2d)),dt,6,{ti2d});
rawdata2d=double(Fg2d*xtrue2d);
tic;xcg3 = regularizedReconstruction(Fg2d,rawdata2d(:),'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',col(zeros(ig.dim(1:2))));toc
% yi=s.Gm*xtrue2d(mask);
% tic;xcg3 = qpwls_pcg1(0*xtrue2d(mask), s.Gm, 1, yi(:), R.C, 'niter', f.niter);toc
nrms(xcg3, xtrue2d)

bb=Fg2d'*rawdata2d(:);
FgT2d=orc_segm_nuFTOperator_multi_sub_T({omega2d},ig.dim(1:2),(double(smaps2d)),(double(fmap2d)),dt,6,{ti2d});
tic;xcg4 = regularizedReconstruction_toep(FgT2d,bb,'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',col(zeros(ig.dim(1:2))));toc
% bb=s.Gm'*yi(:);
% tic;xcg4 = qpwls_pcg2(0*xtrue2d(mask), FgT2d, bb, R.C, 'niter', f.niter);toc
nrms(xcg4, xtrue2d)

