% mri_sense_demo1.m
% Example illustrating regularized iterative reconstruction for parallel MRI
% (sensitivity encoding imaging or SENSE), from nonuniform k-space samples.
% (These examples do not include field inhomogeneity or relaxation.)
% Copyright 2006-4-18, Jeff Fessler, The University of Michigan

if ~isvar('smap'), printm 'sense maps'
	ig = image_geom('nx', 64, 'fov', 250); % 250 mm FOV
	ig.mask = ig.circ(ig.dx * (ig.nx/2-2), ig.dy * (ig.ny/2-1)) > 0;
	f.ncoil = 4;
	smap = mri_sensemap_sim('nx', ig.nx, 'ny', ig.ny, 'dx', ig.dx, ...
		'rcoil', 120, 'ncoil', f.ncoil);
prompt
end

% true object (discrete because of discrete sense map)
if 0 | ~isvar('xtrue'), printm 'true object'
	xtrue = ellipse_im(ig, 'shepplogan-emis', 'oversample', 2);
	clim = [0 8];
	im clf, im pl 3 2
	im(1, ig.x, ig.y, xtrue, 'x true', clim), cbar
	im(4, ig.mask, 'mask')
prompt
end

%
% trajectory
%
if ~isvar('Gm'), printm 'G objects'
	f.traj = 'spiral1';
	f.dens = 'voronoi';

	N = [ig.nx ig.ny];
	[kspace omega wi] = mri_trajectory(f.traj, {}, ...
		N, ig.fov, {f.dens});

	% create Gnufft class object
	J = [6 6];
	nufft_args = {N, J, 2*N, N/2, 'table', 2^10, 'minmax:kb'};
	Gn = Gnufft(ig.mask, {omega, nufft_args{:}});
	Gm = Gmri(kspace, ig.mask, ...
		'fov', ig.fov, 'basis', {'rect'}, 'nufft', nufft_args);

	if im
		im subplot 2
		plot(omega(1:5:end,1), omega(1:5:end,2), '.')
		title(sprintf('%s: %d', f.traj, size(omega,1)))
%		axis(pi*[-1 1 -1 1]), 
		axis_pipi
	end
end

if ~isvar('Gb'), printm 'Gb object with sense maps within'
	for ic=1:f.ncoil
		tmp = smap(:,:,ic);
		Gc{ic} = Gm * diag_sp(tmp(ig.mask)); % cascade
        Tc = build_gram(Gm,1)* diag_sp(tmp(ig.mask));
        if ic>1
            Tb=fatrix_plus(Tb,Tc);
        else
            Tb=Tc;
        end
	end
	Gb = block_fatrix(Gc, 'type', 'col'); % [G1; G2; ... ]
end

if ~isvar('yi'), printm 'data yi'
	ytrue = Gb * xtrue(ig.mask);

	% add noise
	randn('state', 0)
	yi = ytrue + 0 * randn(size(ytrue));
	% todo: visualize data...
end

if ~isvar('xcp'), printm 'conj. phase reconstruction'
	y4 = reshape(yi, [], f.ncoil);
	for ic=1:f.ncoil
		tmp = Gn' * (wi .* y4(:,ic));
		xcp(:,:,ic) = ig.embed(tmp);
	end

	im(2, abs(xcp), 'Conj. Phase Recon for each coil'), cbar
prompt
end

if ~isvar('xsos'), printm 'sum-of-squares reconstruction'
	xsos = sqrt(sum(abs(xcp).^2, 3));
	im(3, abs(xsos), 'sum-of-squares recon'), cbar
prompt
end

if 0 | ~isvar('R'), printm 'regularizer'
	f.beta = 2^11;
	R = Robject(ig.mask, 'beta', f.beta, 'potential', 'quad');
	if 1
		qpwls_psf(Gb, R, 1, ig.mask, 1, 'chat', 0, 'offset', [0 0]);
	end
prompt
end

if ~isvar('xpcg'), printm 'PCG with quadratic penalty'
	f.niter = 10;
	xpcg = qpwls_pcg(xsos(ig.mask), Gb, 1, yi(:), 0, R.C, 1, f.niter);
	xpcg = ig.embed(xpcg(:,end)); % convert last vector to image for display

	im(4, abs(xpcg), '|x| pcg quad', clim), cbar, drawnow
%prompt
end

bb = Gb' * col(wi.*reshape(yi, [], f.ncoil));
% if ~isvar('xcg2'), printm 'xcg2'
    tic;xcg2 = qpwls_pcg2(xsos(ig.mask), Tb, bb, R.C, 'niter', 2*f.niter);toc
    xcg2 = ig.embed(xcg2(:,end));
	im(5, abs(xcg2), '|x| pcg quad', clim), cbar, drawnow
% end
