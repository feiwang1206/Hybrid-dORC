% mri_example_3d.m
% Example illustrating 3D regularized iterative reconstruction for MRI
% from nonuniform k-space samples.
% (This example does not include field inhomogeneity or relaxation.)
% Copyright 2004-12-16, Jeff Fessler, The University of Michigan

% this is an "honest" simulation where the Fourier data is calculated
% analytically from a continuous-space object model, even though the
% reconstructions are all discrete-space.
clear
    addpath /home/ubuntu/Documents/MATLAB/image_reconstruction_toolbox
    setup
load('/home/ubuntu/Documents/work/train/01/data.mat');
omega=data.trajectory.trajectory{1}(data.trajectory.idx{1},1:2);
dt=5e-6;

if ~isvar('xtrue'), printm 'setup object'

	ig = image_geom('nx', 64, 'nz', 1, ...
		'offsets', 'dsp', ... % [-n/2:n/2-1] for mri
		'fov', [19.2 19.2 15]); % 20 cm transaxial FOV
    ig.dim=ig.dim(1:2);
	xs = mri_objects('fov', ig.fovs, 'test4'); % object model
% 	xtrue = xs.image(ig.xg, ig.yg, ig.zg); % samples of continuous-space
    xtrue=imresize(data.anatomical(:,:,32),ig.dim);
% prompt
end


smaps=data.smaps;
size_s=size(smaps);
coil_num=16;
smaps2d=reshape(smaps,[prod(size_s(1:3)) size(smaps,4)]);
[Vt,~]=eig(smaps2d'*smaps2d);
Vt=Vt(:,end:-1:(end-coil_num+1));
Ut=smaps2d*Vt;    
smaps=reshape(Ut,[size_s(1:3) coil_num]);
smaps=squeeze(smaps(:,:,32,:));
% smaps=ones([ig.dim 1])/1;
f.ncoil=size(smaps,3);
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
	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, ig.dim, ig.fovs, f.dens);
% 	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, [ig.dim(1) ig.dim(2)/2], [ig.fovs(1) ig.fovs(2)/2], f.dens);
%     kspace = zeros(size(omega));
%     for id=1:length(ig.dim)
%         dx = ig.fovs(id) / ig.dim(id);
%         kspace(:,id) = omega(:,id) / (2*pi) / dx;
%     end
    
end
% omega=omega(1:3500,:);
% kspace=kspace(1:3500,:);
% wi_traj=wi_traj(1:3500);
% wi_traj = mri_density_comp(kspace, 'voronoi');

omega=omega(1:8:end,:);
fmap=imresize(data.wmap(:,:,32),ig.dim);
f.dt=5e-6;
ti=dt*(1:length(omega));


%%
% omega=omega+[2*pi/64/2 2*pi/64/2];
xg=repmat((-32:31)',[1 64]);
yg=repmat((-32:31),[64 1]);
% xg=repmat((0:63)',[1 64]);
% yg=repmat((0:63),[64 1]);
clear yt
omega1=[omega(:,2) -omega(:,1)];
for c=1:f.ncoil
    for i=1:length(omega1)
        phase=-1i*(omega1(i,1)*xg+omega1(i,2)*yg);
        signal=xtrue.*exp(phase).*smaps(:,:,c);
        yt(i,c)=sum(col(signal));
    end
end
% yt=col(fft2(xtrue));
%%
lambda=0.1;
dim=[64 64];
% F=nuFTOperator(omega,dim,ones(dim),2,[5 5],'minmax:kb');
% Fos=nuFTOperator_os(omega,dim,ones(dim),2,[5 5]);
% FT=nuFTOperator_T(omega,dim,ones(dim),2,[5 5]);
F=nuFTOperator(omega,dim,smaps,2,[5 5],'minmax:kb');
Fos=nuFTOperator_os(omega,dim,smaps,2,[6 6],'minmax:kb',lambda^2);
%%
% FT=nuFTOperator_T2(omega,dim,smaps,2,[6 6],'minmax:kb',0.3);
% bb=Fos'*(yt);%+repmat(xtrue,[1 1 16]).*smaps;
% % bb=sum((Fos'*double(yt)).*conj(smaps),3);
% % bb=sum(bb,3);
% nrms(sum(bb(1:64,1:64).*conj(smaps),3), xtrue)
% tic;xx=FT'*bb;
% xx0=xx;
% nrms(sum(xx(1:64,1:64).*conj(smaps),3), xtrue)
% for i=1:5
% %     xx=xx+FT'*(bb-xx);
%     xx=xx+xx0-FT'*(FT*xx);
%     nrms(sum(xx(1:64,1:64).*conj(smaps),3), xtrue)
% end
% toc
% % cc=sum(xx(1:64,1:64,:).*conj(smaps),3);
% 
% % figure,imagesc(((abs(sum(xx(1:64,1:64).*conj(smaps),3)))));colorbar
% figure,imagesc((sqrt(abs(sum(conj(xx(1:64,1:64,:)).*xx(1:64,1:64,:),3)))));colorbar
%%
lambda=0.4;
scale=[2 2];
for i=1:size(smaps,3)
    smaps2(:,:,i)=padding(smaps(:,:,i),scale);
end
FT=nuFTOperator_T2(omega,dim,smaps,2,[6 6],'minmax:kb',lambda^2);
% bb=Fos'*(yt);%+repmat(xtrue,[1 1 16]).*smaps;
bb=sum((Fos'*double(yt)).*conj(smaps),3);
nrms(bb(1:64,1:64), xtrue)
tic;xx=FT'*((1)*bb);
xx0=xx;
% nrms(xx(1:64,1:64), xtrue)
for i=1:10
%     xx=xx+FT'*(bb-xx);
    xx=xx+xx0-FT'*((1)*(FT*xx));
    nrms(xx(1:64,1:64), xtrue)
end
toc
% cc=sum(xx(1:64,1:64,:).*conj(smaps),3);

figure,imagesc(((abs(xx(1:64,1:64)))));colorbar
% figure,imagesc((sqrt(abs(sum(conj(xx(1:64,1:64,:)).*xx(1:64,1:64,:),3)))));colorbar
%%
lambda=0.1;
Finv=nuFTOperator_inv(omega,dim,smaps,2,[6 6],'minmax:kb',lambda^2);
tic;bb=Finv'*(yt);%/prod(dim*2);%+repmat(xtrue,[1 1 16]).*smaps;
nrms(bb, xtrue)
cc=Finv'*(yt-Finv*bb)+bb;%+repmat(xtrue,[1 1 16]).*smaps;
nrms(cc, xtrue)
for i=1:10
    cc=Finv'*(yt-Finv*cc)+cc;%+repmat(xtrue,[1 1 16]).*smaps;
    nrms(cc, xtrue)
end
toc
% cc=sum(cc(1:64,1:64,:).*conj(smaps),3);
figure,imagesc(((abs(cc))));colorbar,title('inverse')
nrms(cc, xtrue)
%%
% Finv=nuFTOperator_inv(omega,dim,smaps,2,[6 6],'minmax:kb',0.01);
% tic;bb=Finv'*(wi_traj.*yt);%/prod(dim*2);%+repmat(xtrue,[1 1 16]).*smaps;
% cc=Finv'*(wi_traj.*(yt-Finv*bb))+bb;%+repmat(xtrue,[1 1 16]).*smaps;
% for i=1:10
%     cc=Finv'*(yt-Finv*cc)+cc;%+repmat(xtrue,[1 1 16]).*smaps;
% end
% toc
% % cc=sum(cc(1:64,1:64,:).*conj(smaps),3);
% figure,imagesc(((abs(cc))));colorbar,title('inverse')
% nrms(cc, xtrue)
%%
% Finv=nuFTOperator_inv(omega,dim,smaps,2,[6 6],'minmax:kb',lambda^2/10^3);
% tic;bb=(Finv'*yt);%/prod(dim*2);%+repmat(xtrue,[1 1 16]).*smaps;
% cc=Finv'*(yt-Finv*bb)+bb;%+repmat(xtrue,[1 1 16]).*smaps;
% for i=1:20
%     cc=Finv'*(yt-Finv*cc)+cc;%+repmat(xtrue,[1 1 16]).*smaps;
% end
% toc
% cc=sum(cc(1:64,1:64,:).*conj(smaps),3);
% figure,imagesc(((abs(cc))));colorbar,title('inverse')
% nrms(cc, xtrue)
%%
% for i=1:f.ncoil
%     smaps2(:,:,i)=padding(conj(smaps(:,:,i)),[2 2]);
% end
FT=nuFTOperator_T2(omega,dim,smaps,2,[6 6],'minmax:kb',lambda^2);
bb=sum((Fos'*double(yt)).*conj(smaps),3);
% bb=sum((Fos'*double(yt)).*smaps2,3);
% bb=F'*double(yt);
tic;[~,xx] = regularizedReconstruction_toep(FT,bb,'maxit',10,'verbose_flag', 0,'tol',0e-5,'z0',bb);toc
for i=1:length(xx)
    nrms(xx{1}{i}, xtrue)
end
figure,imagesc(((abs(xx{1}{end}(1:64,1:64)))));colorbar
%%
    lengthP = 0;
    P = cell(1,lengthP);
    counter = 1;
    operator.handle = @identityOperator;
    operator.args = {};
    
    P{counter} = @L2Norm;%recon_details.penalty(k).norm;
    counter = counter + 1;
%     lambda=1;
    P{counter} = 0.1;
    counter = counter + 1;
    for k=1:length(operator)
        P{counter} = operator(k).handle(operator(k).args{:});
        counter = counter + 1;
    end
Finv=nuFTOperator_inv(omega,dim,smaps,2,[6 6],'minmax:kb',10^-13);
tic;[~,xx] = regularizedReconstruction(Finv,yt,P{:},'maxit',10,'verbose_flag', 0,'tol',0e-5,'z0',(zeros(dim)));toc
for i=1:length(xx)
    nrms(xx{1}{i}, xtrue)
end
figure,imagesc(((abs(xx{1}{end}(1:64,1:64)))));colorbar
% for i=1:length(xx)
%     nrms(xx{i}, xtrue)
% end
% figure,imagesc(((abs(xx{end}(1:64,1:64)))));colorbar
% %%
% % create Gmri or Gnufft class object
% % if ~isvar('Gm'), printm 'system model'
% 	printm 'setup G objects'
% 	N = ig.dim;
% 	nufft_args = {N, 6*ones(size(N)), 2*N, N/2, 'minmax:kb'};
% 	mask = true(N);
% % 	clear N
% %	f.basis = {'rect'}; % fix: using this has scaling issues
% 	f.basis = {'dirac'};
% 	Gn = Gmri([kspace(:,2) -kspace(:,1)], mask, ...
% 		'fov', 19.2, 'basis', f.basis, 'nufft', nufft_args);
% %     Gm = Gn.arg.new_zmap(Gn, ti, 1i*fmap, 6);
%     Tn=Gmri_gram(Gn,1,1);
% % end
% % if ~isvar('Gb'), printm 'Gb object with sense maps within'
% 	for ic=1:f.ncoil
% 		tmp = smaps(:,:,ic);
% 		Gc{ic} = Gn * diag_sp(tmp(ig.mask)); % cascade
% %         Tc = diag_sp(conj(tmp(ig.mask)))*(build_gram(Gm,1)* diag_sp(tmp(ig.mask)));
%         Tc = diag_sp(conj(tmp(ig.mask)))*(Tn* diag_sp(tmp(ig.mask)));
%         if ic>1
%             Tb=fatrix_plus(Tb,Tc);
%         else
%             Tb=Tc;
%         end
% 	end
% 	Gb = block_fatrix(Gc, 'type', 'col'); % [G1; G2; ... ]
% % end
% % Gbz = Gb.arg.new_zmap(Gb, ti, 1i*fmap,2);
% % Tbz = Tb.arg.new_zmap(Tb, ti, 1i*fmap,2);
% %%
% % yt = Ge_zmap * xtrue(ig.mask);
% 
% % add noise
% randn('state', 0)
% yi = yt + 0 * randn(size(yt));
% % todo: visualize data...
% 
% xcp = Gb' * (yi(:))/prod(ig.dim);
% xcp = embed(xcp, mask);
% if ~isvar('R'), printm 'regularizer'
% 	beta = 2^-7 * size(omega,1);	% good for quadratic
% 	R = Reg1(ig.mask, 'beta', beta);
% 	if 0 % check resolution: [1.1 1.1 1]
% 		qpwls_psf(Gb, R, 1, ig.mask, 1, 'fwhmtype', 'profile');
% 	return
% 	end
% end
% %%
% 	niter = 20;
% 	ytmp = yi;% / abs(ig.dx*ig.dy*ig.dz); % trick: analytical data vs DSFT
% 	tic;xpcg = qpwls_pcg1(1*xcp(ig.mask), Gb, 1, ytmp(:), R.C, 'niter', niter);toc
% % 	tic;xpcg = regularizedReconstruction(Gb,yi(:),'maxit',niter,'verbose_flag', 0,'tol',1e-5,'z0',xcp(ig.mask));toc
% 	xpcg = ig.embed(xpcg);
% figure,imagesc(((abs(xpcg(1:64,1:64)))));colorbar
% 
% %%
% %     ytmp = yi / abs(ig.dx*ig.dy); % trick: analytical data vs DSFT
% 	niter = 20;
%     bb = Gb' * yi(:);%ytmp(:);
% % 	if ~isvar('xcg2'), printm 'xcg2'
% 		tic;xcg2 = qpwls_pcg2(1*xcp(ig.mask), Tb, bb, [], 'niter', niter);toc
% %         tic;xcg2 = regularizedReconstruction_toep(Gb,Tb,bb(:),'maxit',10,'verbose_flag', 0,'tol',1e-5,'z0',xcp(ig.mask));toc
% 		xcg2 = embed(xcg2(:,end), mask);
% % 	end
% figure,imagesc(((abs(xcg2(1:64,1:64)))));colorbar
% 
% nrms(xcp/prod(ig.dim), xtrue)
% nrms(xpcg, xtrue)
% nrms(xcg2, xtrue)
% %%
% Fg=orc_segm_nuFTOperator_multi_sub({omega},ig.dim,gpuArray(double(smaps)),gpuArray(double(fmap)),dt,10,{ti});
% tic;xcg3 = regularizedReconstruction(Fg,yi/sqrt(prod(ig.dim)),'maxit',50,'verbose_flag', 0,'tol',0e-5,'z0',[]);toc
% nrms(xcg3, xtrue)
% %%
% dim=[64 64];
% F=nuFTOperator(omega,dim,ones(dim),2,[5 5],'minmax:kb');
% Fos=nuFTOperator_os(omega,dim,ones(dim),2,[5 5],'minmax:kb');
% FT=nuFTOperator_T(omega,dim,ones(dim),2,[5 5],'minmax:kb');
