clear
load('/home/ubuntu/Documents/work/train/01/data.mat');
omega=data.trajectory.trajectory{1}(7001:8000,1:2);
dt=5e-6;

if ~isvar('xtrue'), printm 'setup object'

	ig = image_geom('nx', 64, 'nz', 1, ...
		'offsets', 'dsp', ... % [-n/2:n/2-1] for mri
		'fov', [19.2 19.2 15]); % 20 cm transaxial FOV
    ig.dim=ig.dim(1:2);
	xs = mri_objects('fov', ig.fovs, 'test4'); % object model
% 	xtrue = xs.image(ig.xg, ig.yg, ig.zg); % samples of continuous-space
    xtrue=imresize(data.anatomical(:,:,32),ig.dim);
    xtrue=phantom(64);
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
smaps=ones([ig.dim 1])/1;

% smaps=ones([ig.dim 2])/1;
% for i=1:64
%     smaps(:,i,1)=((64-i)/64);
%     smaps(:,i,2)=sqrt(1-smaps(:,i,1).^1);
% end
f.ncoil=size(smaps,3);
%
% k-space trajectory, k-space data
%
if ~isvar('yi'), printm 'trajectory'
	f.traj = 'cartesian';
	f.traj = 'spiral3';
	f.traj = 'radial';
	f.dens = {'voronoi'};
% 	f.dens = {};
	cpu etic
% 	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, ig.dim, ig.fovs, f.dens);
% 	[kspace omega wi_traj] = mri_trajectory(f.traj, {}, [ig.dim(1) ig.dim(2)/2], [ig.fovs(1) ig.fovs(2)/2], f.dens);
%     kspace = zeros(size(omega));
%     for id=1:length(ig.dim)
%         dx = ig.fovs(id) / ig.dim(id);
%         kspace(:,id) = omega(:,id) / (2*pi) / dx;
%     end
    
end
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
%%
lambda=1;
dim=[64 64];
F=nuFTOperator(omega,dim,smaps,2,[5 5],'minmax:kb');
FT=nuFTOperator_T(omega,dim,smaps,2,[5 5],'kaiser',lambda);
Fos=nuFTOperator_os(omega,dim,smaps,2,[5 5],'kaiser',0.1);
tic;bb=(1+lambda)*(Fos'*yt)*4;toc
bb=bb(1:64,1:64,:);
figure,imagesc((sqrt(abs(sum(conj(bb).*bb,3)))));colorbar,title('direct inverse')
% bb=Fos'*(yt)*4+lambda*repmat(xtrue,[1 1 size(smaps,3)]).*smaps;
% bb=sum(bb,3);
tic;xx=FT'*bb;toc
% figure,imagesc(((abs(sum(xx,3)))));colorbar,title('direct inverse')
figure,imagesc(((abs(sum(conj(xx).*xx,3)))));colorbar,title('direct inverse')
%%
bb=F'*double(yt);
% bb=Fos'*(yt)*4+lambda*repmat(xtrue,[1 1 size(smaps,3)]).*smaps;
tic;xx = regularizedReconstruction_toep(FT,bb,'maxit',20,'verbose_flag', 0,'tol',0e-5,'z0',bb);toc
figure,imagesc(((abs(xx(1:64,1:64)))));colorbar;title('iterative')
% figure,imagesc(((abs(xtrue(1:64,1:64)))));colorbar;title('true')

%%
xx=F'*(wi_traj.*yt);
figure,imagesc(((abs(xx(1:64,1:64)))));colorbar;title('iterative')

