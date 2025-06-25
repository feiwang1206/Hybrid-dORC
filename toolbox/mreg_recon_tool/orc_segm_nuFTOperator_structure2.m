function  s = orc_segm_nuFTOperator_structure2(trajectory, imageDim, dwelltime, Ns, T,lambda)

%% Usage:
%    A = orc_segm_nuFTOperator(trajectory, imageDim, sensmaps, wmap, dwelltime, Ns, DeltaT)
%
% trajectory =  [N_timepoints x N_dimensions]:
%               Arbitrary trajectory in 2D or 3D k-space (obligatory)
%
% imageDim =    [Nx Ny (Nz)]:
%               Dimensions of the image (Only 2 dimensional for 2D; obligatory)
%
% sensmaps =    [Nx Ny (Nz) Ncoils]:
%               Coil sensitivity maps ([Nx Ny Ncoils] for 2D; optional)
%
% wmap =        [Nx Ny (Nz)]: Off-resonance map in rads/s (obligatory)
%
% dwelltime =   Scalar in seconds; wmap * dwelltimes must be in radians
%               (obligatory)
%
% Ns =          Integer; Number of Segments (10 works fine for MREG
%               trajectories)
%
% DeltaT =      Scalar; Time (seconds), where magnetization is in phase
%               GE: DeltaT = t(beginning of trajectory) - t(excitation); positive
%               SE: DeltaT = t(echo) - t(beginning of trajectory); negative
%               (optional); if not set, the magnetization is assumed to be in 
%               phase at beginning of the trajectory. The effect is some
%               additional phase distribution in the image. This should not
%               have any effect, if the absolute value of the image is of
%               interest and a linear reconstruction is used. But e.g. the
%               total variation penalty acts on the complex variation of
%               the image, so phase variations can be counter productive.
%
% In this implementation of the off-resonance corrected nuFFT-Operator a
% Hanning-Window interpolation is implemented. The nuFFT is hard-coded with
% no oversampling, 5 neighbors in each direction and a Kaiser-Bessel
% Kernel. All of this can easily be changed in the function. For a Min-Max-
% Interploation (in the temporal domain) please use orc_minmax_nuFTOperator
%
%
% 30.09.2011  Thimo Hugger
% 2010 - 2013 Jakob Asslaender

s.imageDim = imageDim;
s.adjoint = 0;
s.trajectory=trajectory;    

if size(trajectory{1},2) == 3
    s.nufftNeighbors = [5 5 5];
else
    s.nufftNeighbors = [5 5];
end
s.oversampling = s.imageDim*2;
s.nSegments = Ns;

trajectory0=trajectory;
for n=1:length(trajectory0)
    tl(n)=size(trajectory0{n},1);
    if size(trajectory{1},2) == 3
        trajectory0{n} = [trajectory0{n}(:,2), -trajectory0{n}(:,1) , trajectory0{n}(:,3)];
    else
        trajectory0{n} = [trajectory0{n}(:,2), -trajectory0{n}(:,1)];
    end
end
s.trajectory_length = sum(tl);
sta=0;
sta1=0;
for n=1:length(trajectory0)
    trajectory=trajectory0{n};
    tl = size(trajectory,1);
    s.tADC(n) = tl*dwelltime;

%     T = s.tADC(n) * [0:tl-1]/tl;

    sr = floor((tl-Ns-1)/Ns);
    sl = 2*sr + 1;

    si = cell(Ns+1,1);
    si{1} = 1:sr+1;
    for kk=1:Ns-1
        si{kk+1} = kk*(sr+1)+1-sr:kk*(sr+1)+1+sr;
    end
    si{end} = Ns*(sr+1)+1-sr:tl;
    for q=1:Ns+1
        s.segment_index{q+sta} = si{q}+sta1;
    end
    h = hann(sl+2);
    h = h(2:end-1);
    ipf = cell(Ns+1,1);
    ipf{1} = h(sr+1:end);
    for kk=1:Ns-1
        ipf{kk+1} = h;
    end
    ipf{end} = [h(1:sr);ones(length(si{end})-sr,1)];  %the remaining datapoints are not included in the interpolation and are therefore set to the last segment only. This is an approximation.
    s.segment_filter(sta+1:sta+Ns+1) = ipf;
%    
    t = zeros(1,Ns+1);
    t(1) = T{n}(1);
    for k=1:Ns
        t(k+1) = T{n}(k*(sr+1)+1);
    end
    s.t(sta+1:sta+Ns+1)=t;
%     for k = 1:Ns+1
%         if size(trajectory0{1},2) == 3
%             s.wmap(:,:,:,k+sta) = exp(1i*wmap*t(k));
%         else
%             s.wmap(:,:,k+sta) = exp(1i*wmap*t(k));                
%         end
%     end
    
    for k=1:Ns+1
%         nstr = nufft_init(trajectory(si{k},:), s.imageDim, s.nufftNeighbors, s.oversampling, s.imageDim/2, 'kaiser');
        nstr = nufft_init_rm(trajectory(si{k},:), s.imageDim, s.nufftNeighbors, s.oversampling, s.imageDim/2, 'kaiser');
        s.interpolation_matrix{k+sta} = nstr.p.arg.G;
        s.interp_filter{k+sta} = spdiags(s.segment_filter{k+sta},-s.segment_index{k+sta}(1)+1,s.trajectory_length,length(s.segment_index{k+sta}))*nstr.p.arg.G;
    end
    
    sta=sta+Ns+1;
    sta1=sta1+tl;
end
trajectory=[];
for n=1:length(trajectory0)
    trajectory=[trajectory;trajectory0{n};];
end
nstr = nufft_init(trajectory, s.imageDim, s.nufftNeighbors, s.oversampling, s.imageDim/2, 'minmax:kb');

s.scaling_factor = nstr.sn; % is the same each time
s.lambda=lambda;



% function wi = mri_dcf_pipe(kspace, G)
% wi = ones(length(kspace), 1);
% P = G.p;
% goal = inf;
% iter = 0;
% saver = zeros(200,1);
% while max(abs(goal-1)) > 0.02
% 	iter = iter + 1;
% 	goal = P * (P' * wi);
% 	wi = wi ./ real(goal);
% 	if iter > 20
% 		warning 'iteration stuck?'
% % 		keyboard
%         break
% 	end
% 	saver(iter) = max(abs(goal-1));
% end
% % printm('pipe ended at iteration %d with %g', iter, max(abs(goal-1)))
% % plot(saver(2:end))
% % keyboard
% 
% % fov = 256;
% % scale = G.sn(end/2,end/2)^(-2) / fov^2 ...
% % 	/ prod(G.Kd) * prod(G.Nd);
% % wi = wi * scale;
% 
% function wi=circu_precond(s,lambda)
% gap=s.oversampling./s.imageDim;
% % scale=prod(s.oversampling).^1.5/prod(s.imageDim).^2;
% scale=1/prod(s.imageDim).^2;
% psf=nufft_adj(ones(s.trajectory_length,1),s.nufftStruct);%/prod(s.oversampling)^1.5;
% wi=0;
% for i=1:length(s.sensmaps)
%     xf=abs(fftshift(fftn(conj(s.sensmaps{i}),s.oversampling))).^2;%/prod(s.oversampling);
%     x=fftshift(ifftn(xf));%*sqrt(prod(s.oversampling));
%     x=x.*psf;
%     wi_i=fftn(fftshift(x));%/sqrt(prod(s.oversampling));
%     if length(size(x))==3
%         wi_i=wi_i(1:gap(1):end,1:gap(2):end,1:gap(3):end);
%     else
%         wi_i=wi_i(1:gap(1):end,1:gap(2):end);
%     end
%     wi_i=wi_i*scale;
%     wi=wi+wi_i;
% end
% wi=wi+lambda;
% wi(wi==0)=1;
% wi=1./wi;    

