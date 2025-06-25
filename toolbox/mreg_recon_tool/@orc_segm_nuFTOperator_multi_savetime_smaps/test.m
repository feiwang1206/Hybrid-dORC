function  wi = orc_segm_nuFTOperator_multi_savetime_smaps_structure(s, sensmaps)
wi=0;
% if isempty(sensmaps)
%     s.numCoils = 1;
% else
%     if size(s.trajectory{1},2) == 3 && length(size(sensmaps))== 4
%         s.numCoils = size(sensmaps, length(size(sensmaps)));
%     end
%     if size(s.trajectory{1},2) == 3 && length(size(sensmaps))== 3
%         s.numCoils = 1;
%     end
%     if size(s.trajectory{1},2) == 2 && length(size(sensmaps))== 3
%         s.numCoils = size(sensmaps, length(size(sensmaps)));
%     end
%     if size(s.trajectory{1},2) == 2 && length(size(sensmaps))== 2
%         s.numCoils = 1;
%     end
% end
% if isempty(sensmaps)
%     s.sensmaps{1} = 1;
% else
%     for k=1:s.numCoils
%         if size(s.trajectory{1},2) == 3
%             if s.numCoils > 1
%                 s.sensmaps{k} = sensmaps(:,:,:,k);
%             else
%                 s.sensmaps{1}=sensmaps;
%             end
%         else
%              s.sensmaps{k} = sensmaps(:,:,k);
%         end
%     end
% end
% s.sensmaps_scale = bsxfun(@times,s.scaling_factor,sensmaps);
% 
% 
% s1=s;
% trajectory=[];
% for n=1:length(s.trajectory)
%     trajectory=[trajectory;s.trajectory{n};];
% end
% s1.nufftStruct= nufft_init(trajectory, s.imageDim*2, s.nufftNeighbors, round(4*s.imageDim), ceil(s.imageDim), 'minmax:kb');
% wi=kspace_precond(s1,s.sensmaps,s.lambda);


% function wi=kspace_precond(s,sensmaps,lambda)
% scale=prod(s.oversampling).^1.5/prod(s.imageDim);
% psf=nufft_adj(ones(s.trajectory_length,1),s.nufftStruct);
% wi=[];
% for i=1:length(sensmaps)
%     snorm=l2norm(gather(col(sensmaps{i})))^2;
%     xf=0;
%     for j=1:length(sensmaps)
%         xf=xf+abs((fftn((sensmaps{i}.*conj(sensmaps{j})),s.oversampling))).^2;
%     end
%     x=fftshift(ifftn((xf)))/sqrt(prod(s.oversampling));
%     x=x.*psf;
%     wi_i=nufft(x,s.nufftStruct)/prod(s.oversampling);
%     wi_i=wi_i*scale/snorm;
%     wi(:,i)=gather(col(wi_i));
% end
% wi=(abs(wi)+lambda)/(1+lambda);
% wi(wi==0)=1;
% wi=1./wi;        
    

    

