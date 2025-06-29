 function [T, reuse] = nufft_gram(ob, W, reuse)
%function [T, reuse] = nufft_gram(ob, W, reuse)
% construct Toeplitz matrix object T = G'WG for a Gnufft object
% in
%	ob	Fatrix		a [M,np] Gnufft object
%	W	[M,M]		W = diag_sp(wi) (often simply "1" or [])
%	reuse	struct		stuff from the last call that can be reused
% out
%	T	[np,np]		Fmatrix object
%	reuse	struct		stuff that can be reused on future calls
%
% Copyright 2004-6-29, Jeff Fessler & Hugo Shi, The University of Michigan

if nargin == 1 & streq(ob, 'test'), nufft_gram_test, return, end
if nargin < 2 | nargin > 3, help(mfilename), error(mfilename), end

if ~isvar('reuse'), reuse = []; end

% extract
if isnumeric(W) % includes case where W is empty
	wi = W;
elseif isa(W, 'Fatrix') && streq(W.caller, 'diag_sp')
	wi = W.arg.diag;
else
	error 'nufft_gram requires W to be diag_sp or wi array'
end
clear W

if isreal(wi)
	back = @nufft_gram_mult_fun; % trick: because Hermitian!
else
	back = []; % do not provide T' if wi is complex
end

arg = ob.arg;
[arg.ffttoep arg.reuse] = nufft_gram_init(arg, wi, reuse);
T = Fatrix([1 1]*arg.dim(2), arg, 'caller', 'nufft_gram', ...
	'forw', @nufft_gram_mult_fun, 'back', back);


%
% nufft_gram_init()
% construct kernel of circulant matrix into which T is embedded
% and take its DFT to prepare for multiplication.
% out:
% ffttoep [[2Nd]]	kernel of circulant matrix used in FFT
%
function [ffttoep, reuse] = nufft_gram_init(arg, wi, reuse)

if ~isvar('wi') | isempty(wi)
	wi = ones(arg.dim(1), 1); % the usual unweighted case
end

Nd = size(arg.mask);
if ~any(length(Nd) == [2 3])
	error 'only 2D and 3D cases done'
end

if length(Nd) == 2
	if 0
		g1 = build_Gmod1(arg);
		g2 = build_Gmod2(arg);
		s1 = g1.arg.st;
		s2 = g2.arg.st;
	%	max_percent_diff(s1.phase_shift, s2.phase_shift)
		printm('scaling factors: mpd=%g%%', max_percent_diff(s1.sn, s2.sn))
	%	max_percent_diff(s1.alpha{1}, s2.alpha{1})
		printm('table1: mpd=%g%%', max_percent_diff(s1.h{1}, s2.h{1}))
	%	keyboard
	end

	if 0
		toep = fftshift(reshape(g2' * wi, g2.arg.st.Nd));
		ffttoep = fftn_fast(toep);

	return

		t = toep;
		%t = t ./ Gmod.st.sn;
		t(end/2+1,:) = 0;
		t(:,end/2+1) = 0;
	end

	% what follows is left over for comparison with Hugo's way.


	if isempty(reuse)
		[reuse.G1 reuse.G2] = build_G1_G2(arg);
	end

	N1 = Nd(1);
	N2 = Nd(2);

	block1 = reshape(reuse.G1' * real(wi), Nd); % kludge
	block2 = reshape(reuse.G2' * real(wi), Nd);

	if 0 % hugo way
		toep = zeros(2*Nd);
		toep(1:N1,1:N2) = block1;
		toep(N1+2:2*N1, N2+2:2*N2) = ...
			fliplr(flipud(conj(block1(2:N1, 2:N2))));
		toep(N1+2:2*N1, 1:N2) = flipud(block2(2:N1, :));
		toep(1:N1, N2+2:2*N2) = ...
			conj(fliplr([block1(1,2:N2); block2(2:N1,2:N2)]));
	end

	z1 = zeros(N1,1);
	z2 = zeros(N1-1,1);
	toep = [
	[block1 z1 conj(fliplr([block1(1,2:N2); block2(2:N1,2:N2)]))];
	zeros(1,2*N2);
	[flipud(block2(2:N1,:)) z2 fliplr(flipud(conj(block1(2:N1, 2:N2))))]
	]; % [(2Nd)]

	% deal with complex wi if needed, which it rarely should be.
	if ~isreal(wi)
		persistent warned
		if isempty(warned), warned = 0; end
		if ~warned
			printm('Are you *sure* you want complex wi values?')
			warned = 1;
		end

		block1 = reshape(reuse.G1' * imag(wi), Nd);
		block2 = reshape(reuse.G2' * imag(wi), Nd);

		itoep = [
		[block1 z1 conj(fliplr([block1(1,2:N2); block2(2:N1,2:N2)]))];
		zeros(1,2*N2);
		[flipud(block2(2:N1,:)) z2 fliplr(flipud(conj(block1(2:N1, 2:N2))))]
		];

		toep = toep + 1i * itoep; % since everything is linear
	end

	%clf, im(real([toep; t; t-toep]))
	%clf, im(imag([toep; t; t-toep]))
	%printm('me vs hugo %g%%', max_percent_diff(t, toep))
	%keyboard

	ffttoep = fftn_fast(toep);

else % 3d
	if isempty(reuse)
		[reuse.G1 reuse.G2 reuse.G3 reuse.G4 ] = build_G1_G2_G3_G4(arg);
	end
	N1 = Nd(1);
	N2 = Nd(2);
	N3 = Nd(3);

	Filtblk1 = reshape(reuse.G1' * real(wi), Nd);
	Filtblk2 = reshape(reuse.G2' * real(wi), Nd);
	Filtblk3 = reshape(reuse.G3' * real(wi), Nd);
	Filtblk4 = reshape(reuse.G4' * real(wi), Nd);
% tmp=zeros(prod(Nd),1);
% tmp(1)=1;
% 	Filtblk1 = reshape(reuse.G1' * real(wi.*(reuse.G1*tmp)), Nd);
% 	Filtblk2 = reshape(reuse.G2' * real(wi.*(reuse.G2*tmp)), Nd);
% 	Filtblk3 = reshape(reuse.G3' * real(wi.*(reuse.G3*tmp)), Nd);
% 	Filtblk4 = reshape(reuse.G4' * real(wi.*(reuse.G4*tmp)), Nd);

	tblk1 = Filtblk1;
	tblk2 = Filtblk2(2:N1,:,:);	%remove the duplicated part with Filtblk1
	tblk3 = Filtblk3(2:N1,2:N2,:);	%remove the duplicated part with block 1, 2, 4
	tblk4 = Filtblk4(:,2:N2,:);	%remove the duplicated part with block 1

	z1 = zeros(N1,1,N3);
	z2 = zeros(N1-1,1,N3);

	%top half of the 3D filter
	toep_top = [
		tblk1 z1 flipdim(tblk4,2);			% Upper block
		zeros(1,2*N2,N3);				% Zero padding in the middle
		flipdim(tblk2,1) z2 flipdim(flipdim(tblk3,1),2) % lower block
	];

	%construct the bottom half now
	bblk1 = flipdim(conj(Filtblk3(:,:,2:N3)),3);
	bblk2 = flipdim(conj(Filtblk4(:,:,2:N3)),3);
	bblk2 = bblk2(2:N1,:,:);
	bblk3 = flipdim(conj(Filtblk1(:,:,2:N3)),3);
	bblk3 = bblk3(2:N1,2:N2,:);
	bblk4 = flipdim(conj(Filtblk2(:,:,2:N3)),3);
	bblk4 = bblk4(:,2:N2,:);

	z4 = zeros(N1,1,N3-1);
	z5 = zeros(N1-1,1,N3-1);
	toep_bottom = [
		bblk1 z4 flipdim(bblk4,2);
		zeros(1,2*N2,N3-1);
		flipdim(bblk2,1) z5 flipdim(flipdim(bblk3,1),2)
		];

	toep = cat(3,toep_top,zeros(2*N1,2*N2,1));
	toep = cat(3,toep,toep_bottom);

	ffttoep = fftn_fast(toep);
end


%
% build a Gnufft object that works [-(N-1),(N-1)] instead of [0,N-1]
%
function G = build_Gmod1(arg)
st = arg.st;
if ~isvar('st.phase_shift'), error 'only table-nufft done', end

st.n_shift = st.Nd;	% shift to center
st.phase_shift = exp(1i * (st.om * st.n_shift(:)));
st.Nd = 2*st.Nd;	% double N
st.Kd = 2*st.Kd;	% double K
dims(2) = prod(st.Nd);
if streq(st.ktype, 'minmax:kb')
	for id = 1:length(st.Nd)
		[st.alpha{id}, st.beta{id}] = ...
		nufft_alpha_kb_fit(st.Nd(id), st.Jd(id), st.Kd(id));
	end
	st.sn = nufft_scale(st.Nd, st.Kd, st.alpha, st.beta);
else
	error 'not done'
end
mask = true(st.Nd); % trick: full mask!
G = Gnufft(mask, st);


%
% build a Gnufft object that works [-(N-1),(N-1)] instead of [0,N-1]
% this seems to require building a new interp table (why?) so it is slower.
%
function G = build_Gmod2(arg)
st = arg.st;
if ~isvar('st.phase_shift'), error 'only table-nufft not done', end
arg = arg.arg; % nufft arguments
Nd = st.Nd;
arg{2} = 2*Nd;		% double N
arg{4} = 2*arg{4};	% double K ???
arg{5} = Nd;		% shift
mask = true(2*Nd);	% double-size mask
G = Gnufft(mask, arg); % todo: can i avoid this?


%
% modified versions of G
% only works for 2D now
% this routine is quite tricky.  nothing new is really rebuilt here,
% but some structure values are changed.
%
function [G1, G2] = build_G1_G2(arg)
st = arg.st;
mask = true(st.Nd); % trick: full mask!

if isvar('st.phase_shift') % table-based nufft
	st = rmfield(st, 'phase_shift'); % trick: eliminate phase shift
	st.n_shift = 0 * st.n_shift;

	% now build two Gnufft objects, one with negative om1.  this is 2D only.
	G1 = Gnufft(mask, st);
	st.om(:,1) = -st.om(:,1);
	G2 = Gnufft(mask, st);

else
%	warning 'non-table nufft may not work'
	G1 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0]});
	st.om(:,1) = -st.om(:,1);
	G2 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0]});
end

% Created by Daehyun Yoon for 3D case
function [G1, G2, G3, G4] = build_G1_G2_G3_G4(arg)
st = arg.st;
mask = true(st.Nd); % trick: full mask!

if isvar('st.phase_shift') % table-based nufft
	st = rmfield(st, 'phase_shift'); % trick: eliminate phase shift
	st.n_shift = 0 * st.n_shift;

	% now build two Gnufft objects, one with negative om1.  this is 2D only.
	G1 = Gnufft(mask, st);
	st.om(:,1) = -st.om(:,1);
	G2 = Gnufft(mask, st);
	st.om(:,2) = -st.om(:,2);
	G3 = Gnufft(mask, st);
	st.om(:,1) = -st.om(:,1);
	G4 = Gnufft(mask,st);

else
%	warning 'non-table nufft may not work'
	G1 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0 0]});
	st.om(:,1) = -st.om(:,1);
	G2 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0 0]});
	st.om(:,2) = -st.om(:,2);
	G3 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0 0]});
	st.om(:,1) = -st.om(:,1);
	G4 = Gnufft(mask, {st.om, st.Nd, st.Jd, st.Kd, [0 0 0]});
end


%
% nufft_gram_mult_fun()
% bridge function to do multiplication
%
function y = nufft_gram_mult_fun(arg, x)
y = nufft_gram_mult(x, arg.ffttoep, arg.mask, arg.dim(2));


%
% nufft_gram_mult()
% multiply an image x by a toeplitz matrix
% by embedding it into a circulant matrix of twice the size and using FFT.
% in
%	x	[*Nd,1] or [np,1]
%	ffttoep	[[2Nd]]
%	mask	[[Nd]]
% out
%	y		[*Nd,1] or [np,1]
function y = nufft_gram_mult(x, ffttoep, mask, np)

N2 = size(ffttoep);
Nd = N2 / 2;

if size(x,1) == np
	x = embed(x, mask); % [np,(R)] to [(Nd),(R)]
end

LL = size(x,1+length(N2));
% y = (zeros([Nd LL]));
y = 0*x;

if length(N2) == 2
	for ll=1:LL
		tmp = ifftn_fast(ffttoep .* fftn_fast(x(:,:,ll), N2));
		y(:,:,ll) = tmp(1:Nd(1), 1:Nd(2));
	end

elseif length(N2) == 3
	for ll=1:LL
        tmp = ifftn_fast(ffttoep .* fftn_fast(x(:,:,:,ll), N2));
		y(:,:,:,ll) = tmp(1:Nd(1), 1:Nd(2), 1:Nd(3));
	end

else
	error 'only 2D and 3D cases implemented'
end

% we must always make output column
y = reshape(y, [], LL);
y = y(mask(:),:);


%
% test this object and compare its speed to Gnufft approach
%
function nufft_gram_test

N = [32 28];
J = [6 5];
K = 2*N;
fov = N;
[kspace omega wi] = mri_trajectory('spiral0', {}, N, fov, {'voronoi'});
nufft_args = {N, J, K, N/2, 'table', 2^11, 'minmax:kb'};
mask = ellipse_im(N(1), N(2), [0 0 14 15 0 1], 'oversample', 3) > 0;
%mask = true(N);
% todo: need to build DSFT G here for comparison!
G = Gnufft(mask, {omega, nufft_args{:}});
%wi = ones(size(omega,1),1);
wi = [1:size(omega,1)]';
randn('state', 0)
wi = wi + 1i*randn(size(wi)); % stress it with complex case!
T = build_gram(G, diag_sp(wi));
x = 0.*mask; x(end/2,end/2+1) = 2;
tic
for ii=1:50, b1 = embed(G' * (wi .* (G * x(mask))), mask); end
t1 = toc;
tic
for ii=1:50, b2 = embed(T * x(mask), mask); end
t2 = toc;
d = max_percent_diff(b1, b2);
printm('time: G''Gx = %g, Tx=%g', t1, t2)
printm('max percent diff between Tx and G''Gx = %g%%', d)
im clf, im(211, abs(stackup(b1, b2)), 'G''G and T'), cbar
im(212, abs(b1-b2), 'diff'), cbar
if d > 0.1
	error 'bug?'
else
	printm 'nufft_gram appears to be working'
end

% todo: see Gnufft_test0.m for possible residual error?
