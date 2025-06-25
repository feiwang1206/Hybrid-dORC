% mri_example_b0.m
% Example of field-corrected MR reconstruction for a spiral k-space trajectory.
% Copyright 2005-7-29, Jeff Fessler, The University of Michigan
clear
load('/home/ubuntu/Documents/work/train/01/data.mat');
omega=data.trajectory.trajectory{1}(6201:7200,1:2);
fmap=data.wmap(:,:,24);
dt=5e-6;
% zmap
if ~isvar('zmap'), printm 'read fieldmap'
% 	[fmap mask_orig] = mri_phase_denoise('test');
	fmap = fmap / (2*pi); % 2ms echo time diff.
	[nx ny] = size(fmap);
	mask = ellipse_im(nx, ny, [0 0 27 35 0 1], 'dx', 1, 'oversample', 3) > 0;

	if 1 & im % picture of field map
		t = imdilate(mask, strel('square', 3)) - mask;
		t = t .* (1 + (-1).^[1:nx]' * (-1).^[1:ny])/2;
		t = t * max(fmap(:));
		if im
			im clf, im(fmap + t)
			title 'Fieldmap and mask outline'
			h = colorbar; %subplot(h); 
            ylabel 'Hz'; clear h
		end
	end

	zmap = 0 + (2i*pi) * fmap;
prompt
end

% xtrue
if ~isvar('xtrue'), printm 'xtrue'
	f.eparam = [
		[0 0 23 29 0 1];
		[0 16 10 10 0 1];
		[0 -16 8 8 0 1];
		[-8 -1 4 8 0 -1];
		[+8 -1 4 8 0 -1];
	];
	xtrue = dsingle(ellipse_im(nx, ny, f.eparam, 'dx', 1, 'oversample', 3));
	xtrue(48:49, 32+[-10:10]) = 2;
	xtrue(16, 32+[-2:2]) = 2;
	xtrue(end/2+1, end/2+1) = 2;
xtrue=data.anatomical(:,:,24);
mask=true(size(xtrue));
	f.clim = [0 2.5];
	if 1 & im
		t = imdilate(mask, strel('square', 3)) - mask;
		im clf, im(xtrue + 2.0*t, f.clim), cbar
		title('Image and support outline')
	prompt, clear t
	end
end


% kspace trajectory
if ~isvar('kspace'), printm 'kspace (slow due to voronoi)'
	N = [nx ny];
	f.fov = 19.2; % cm
	f.traj = 'spiral3'; f.dens = {'voronoi'};
	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, N, f.fov, f.dens);
%     kspace = zeros(size(omega));
%     for id=1:2
%         dx = f.fov / nx;
%         kspace(:,id) = omega(:,id) / (2*pi) / dx;
%     end
% 	if 1 & im
% 		plot(omega(:,1), omega(:,2), '.')
% 		axis(pi*[-1 1 -1 1]), axis_pipi, axis square
% 	prompt
% 	end
end

% sample times, starting 0
if ~isvar('ti')
	f.dt = 5e-6;
	ti = dsingle(([1:length(kspace)]-1)*f.dt);
%	ti = ti + .03; warning 'ti shift!' % shift echo for sangwoo test
	f.daq = max(ti) + f.dt;
	printm('readout time: %g ms', 1000 * f.daq)
end


% "exact" system for generating the data
if ~isvar('Ge_zmap'), printm 'Ge_zmap'
	Ge_ft = Gmri(kspace, mask, 'exact', 1, 'n_shift', N/2, ...
		'fov', f.fov, 'basis', {'rect'});
	% trick! adjust wi's to undo the basis effect for CP
% 	wi_basis = wi_traj ./ Ge_ft.arg.basis.transform;

	Ge_zmap = feval(Ge_ft.arg.new_zmap, Ge_ft, ti, zmap, {});
	clear yi
end


% base NUFFT
if ~isvar('Gn'), printm 'Gn'
	f.nufft = {N, [6 6], 2*N, N/2, 'table', 2^12, 'minmax:kb'};
	Gn = Gmri(kspace, mask, 'fov', f.fov, 'basis', {'rect'}, ...
		'nufft', f.nufft);
end


% data
if ~isvar('yi'), printm 'yi'
	yt = Ge_ft * xtrue(mask);
	% add noise
	randn('state', 0)
	yn = randn(size(yt)) + 1i * randn(size(yt));
	f.snr_db = 50;
	f.snr_db = inf;
	f.scale_noise = norm(yt) / norm(yn) / exp(f.snr_db / 20);
	yi = yt + f.scale_noise * yn;
	printm('data rmse = %g', rms(yi-yt)')
	clear xup0 xcp0 yn %xe
% prompt
end


%
% penalty
%
if ~isvar('R'), printm 'R'
	% scale beta by fov^4 since A'A and 2D.
	f.beta = 2^-28 * size(omega,1) * f.fov^4;
	R = Robject(mask, 'potential', 'quad', 'beta', f.beta);

	if 1 % explore resolution
		[psf var] = qpwls_psf(Ge_ft, R.C, 1, mask);
		im(psf), cbar
		printm('stddev = %g', sqrt(var * prod(N)))
% 	prompt
	end
end

% system
if ~isvar('Gm'), printm 'Gmri'
	L = 6;
	Gm = feval(Gn.arg.new_zmap, Gn, ti, zmap, L);
end

% if 0 && ~isvar('Tm'), printm 'Tm: Gmri gram'
if ~isvar('Tm'), printm 'Tm: Gmri gram'
	Tm = build_gram(Gm, 1);
% return
end


if ~isvar('xcg1'), printm 'xcg1 iterative'
	f.niter = 15;

	cpu tic
	tic;xcg1 = qpwls_pcg1(0*xtrue(mask), Gm, 1, yi(:), R.C, 'niter', f.niter);toc
	xcg1 = embed(xcg1(:,end), mask);
	im clf, im([xtrue; xcg1], 'xcg1'), cbar

% 	if 0 & ~isvar('bb'), printm 'bb'
	if ~isvar('bb'), printm 'bb'
		bb = Gm' * yi(:);
		im clf, im(embed(bb,mask), 'bb'), cbar
		prompt
	end

% 	if 0 & ~isvar('xcg2'), printm 'xcg2'
	if ~isvar('xcg2'), printm 'xcg2'
		tic;xcg2 = qpwls_pcg2(0*xtrue(mask), Tm, bb, R.C, 'niter', f.niter);toc
		xcg2 = embed(xcg2(:,end), mask);
		im clf, im(xcg2, 'xcg2'), cbar
		prompt
	end
prompt
end

nrms(xcg1, xtrue)
nrms(xcg2, xtrue)

%%
% smaps=imresize4D(data.smaps,ig.dim);
% size_s=size(smaps);
% coil_num=3;
% smaps2d=reshape(smaps,[prod(size_s(1:3)) size(smaps,4)]);
% [Vt,~]=eig(smaps2d'*smaps2d);
% Vt=Vt(:,end:-1:(end-coil_num+1));
% Ut=smaps2d*Vt;    
% smaps=reshape(Ut,[size(data.wmap) coil_num]);
% smaps2d=squeeze(smaps(:,:,24,:));
ig.dim=[64 64];
smaps=ones([ig.dim 1])/1;

omega1=[-omega(:,2) omega(:,1)];
Fg2d=orc_segm_nuFTOperator_multi_sub({omega1},ig.dim,(double(smaps)),(double(fmap)),dt,6,{ti});
% rawdata2d=double(Fg2d*xtrue);
tic;xcg3 = regularizedReconstruction(Fg2d,yi(:),'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',(zeros(ig.dim(1:2))));toc
% yi=s.Gm*xtrue2d(mask);
% tic;xcg3 = qpwls_pcg1(0*xtrue(mask), Gm, 1, yi(:), R.C, 'niter', f.niter);toc
nrms(xcg3, xtrue)
figure,imagesc(abs(xcg3));colorbar
%%
% Fg2d=nuFTOperator(omega1,ig.dim,(double(smaps)));
% yi1=Fg2d*xtrue;
bb=Fg2d'*yi(:)*prod(ig.dim);
FgT2d=orc_segm_nuFTOperator_multi_sub_T({omega1},ig.dim(1:2),(double(smaps)),(double(fmap)),dt,6,{ti});
% FgT2d=nuFTOperator_T(omega1,ig.dim(1:2),(double(smaps)),2,[5 5],'kaiser',0.5);
tic;xcg4 = regularizedReconstruction_toep(FgT2d,bb,'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',(zeros(ig.dim(1:2))));toc
% bb=s.Gm'*yi(:);
% tic;xcg4 = qpwls_pcg2(0*xtrue(mask), Tm, bb, R.C, 'niter', f.niter);toc
nrms(xcg4, xtrue)
figure,imagesc(abs(xcg4));colorbar

