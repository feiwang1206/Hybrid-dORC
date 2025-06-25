% radon_example
% This m-file was based on one written by Gianni Schena at Univ. Trieste.
% It is a natural place to start because it directly compares matlab's
% radon and iradon routines with an algorithm provided in this suite.
clear
figure('units','normalized','outerposition',[0 0 1 1]);
if ~isvar('xmat')
	ig = image_geom('nx', 64*2, 'ny', 64*2, 'dx', 1);
	xtrue =0.03* phantom(ig.nx)'; % make phantom
	xtrue = max(xtrue,0);
    metal=zeros(size(xtrue));
    metal((ig.nx/4-5):(ig.nx/4+5),(ig.nx/2-5):(ig.nx/2+5))=1;
    xtrue=xtrue+metal*max(col(xtrue))*10;
	im clf, im pl 3 4
	im(1, xtrue, 'phantom'), cbar

	% project phantom using matlab way
	na = 360/2;
	angle = [0:(na-1)]/na*360;
	[yi_mat, rad] = radon(xtrue', angle);	% trick: transpose!
    bi=9.0e2;
    yb = bi .* exp(-yi_mat);
    % yb = poisson(yb/1e12)*1e12;
%     yb=abs(yb+1e-1*randn(size(yb)));
    yb=imnoise(yb,'Poisson');
    yi_mat = -log((yb) ./ bi);
    yi_mat(yi_mat <= 0) = 0; 

    printm('ray spacing = %g,%g', min(diff(rad)), max(diff(rad)))

	sg = sino_geom('par','orbit_start',angle(1),'orbit',angle(end), 'nb', size(yi_mat,1), 'na', size(yi_mat,2), ...
		'dr', rad(2)-rad(1));
%     sg = sino_geom('fan', 'orbit_start',angle(1),'orbit',angle(end),'nb', size(yi_mat,1), 'na', size(yi_mat,2), ...
%     'ds', 1,'dsd',349,'dso',241,'dod',108,'dfs',0);

    im(2, yi_mat, '"radon" sino'), cbar

	% matlab reconstruction from sinogram
prompt
end

%
% make system model: use Gtomo2_dsc if possible,
% otherwise fall back to Gtomo2_strip
%
if ~isvar('G'),	printm 'G'

% 	if has_mex_jf
% % 		G = Gtomo2_wtmex(sg, ig, 'pairs', {'strip_width', sg.dr'});
% 		G = Gtomo2_wtmex(sg, ig, 'pairs', {'strip_width', sg.ds'});
% 	else
		warning 'you really should get wtfmex!'
		printm 'making Gtomo2_strip: be patient!'
		G = Gtomo2_strip(sg, ig, 'strip_width', sg.dr);
% 		G = Gtomo2_strip(sg, ig, 'strip_width', sg.ds);
% 	end
prompt
end

% if ~isvar('yi'), printm 'yi' % make sinogram
% 	yi = G * xtrue; % forward projector, aka DRR
%     bi=1e6;
%     yb = bi .* exp(-yi);
%     yi=abs(yb+1e-100*randn(size(yb)));
%     yi = -log((yi) ./ bi);
%     yi(yi <= 0) = 0; 
    yi=yi_mat;

    xmat = iradon(yi_mat, sg.ad, 'linear', 'Ram-Lak', 1, ig.nx)'; % transpose!
%     xmat = iradon(yi_mat, sg.ad, 'hann', ig.nx)'; % transpose!
	im(3, xmat, '"iradon" recon'), cbar
	im(4, abs(xmat-xtrue), '|error|'), cbar
% 	im(7, yi, 'sinogram'), cbar
% 	im(8, abs(yi-yi), 'sino |diff|'), cbar
prompt
% end

% reconstruct using em_fbp FBP algorthm
if ~isvar('xfbp1'), printm 'xfbp1'
	fbp_kernel = [1];	% noiseless data, so no filtering
	xfbp1 = em_fbp(sg, ig, max(yi,0), [], [], 'kernel', fbp_kernel);
	im(5, xfbp1, 'em\_fbp'), cbar
	im(6, abs(xfbp1-xtrue), 'error'), cbar
prompt
end

if ~isvar('Rq'), printm 'Rq'
	l2b = 2;	% log_2(beta)
%	C = C2sparse('leak', ig.mask, 8, 0);
	Rq = Robject(ig.mask, 'beta', 2^l2b);

	if 1
		psf = qpwls_psf(G, Rq, 1, ig.mask);
		printm('expected FWHM = %g', fwhm2(psf))
	end
prompt
end

% fix: Caller must do G = G(:,mask(:)) for masked reconstructions.

% CG iterative reconstruction
if ~isvar('xcg'), printm 'xcg'
	wi = ones(size(yi));
	W = diag_sp(wi(:));
	niter = 10;
	xcg = qpwls_pcg(xfbp1(ig.mask), G, W, yi(:), 0, ...
		Rq.C, 1, niter, ig.mask, 1);
	xcg = ig.embed(xcg(:,end));
	im(7, xcg, 'CG'), cbar
	im(8, abs(xcg-xtrue), 'error'), cbar
prompt
end

%
% run the block-iterative SPS-OS algorithm
%
if 1, printm 'xsps'
	nblock = 10;
	Gb = Gblock(G, nblock, 0);

	if ~isvar('R'),	printm 'R'
		R = Robject(ig.mask, 'type_denom', 'matlab', ...
			'potential', 'huber', 'beta', 2^l2b, 'delta', 0.1);
	end

	xsps = pwls_sps_os(xfbp1(ig.mask), yi, [], Gb, R, niter, 2, [], [], 1, 1);
	xsps = ig.embed(xsps(:,end));
	im(9, xsps, 'SPS-OS'), cbar
	im(10, abs(xsps-xtrue), 'error'), cbar
end
