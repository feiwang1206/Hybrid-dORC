function [z,z_iter] = conjugateGradient2(A,A2,b,tol,maxit,z0,verbose_flag, newfigure)

% function z = conjugateGradient(A,b,tol,maxit,z0)
%
% Solves: A*z=b
% A must be hermitian and positive definite.
%
% tol = tolerance of the method (default: 1e-6)
% maxit = maximum number of iterations (default: 20)
% z0 = initial guess (default: 0)
%
% 30.09.2011
% Thimo Hugger
% 21.10.2019
% Fei Wang

persistent h;

start_at_zero = 1;

if nargin<=2 || isempty(tol)
    tol = 1e-6;
end
if nargin<=3 || isempty(maxit)
    maxit = 20;
end
if nargin<=4 || isempty(z0)
    if isnumeric(b)
        z = zeros(size(b));
    else
        eval(['z = ',class(b),'(size(b));']);
    end
elseif isnumeric(z0) && isscalar(z0) && z0==0
    z = zeros(size(b));
else
    z = z0;
    start_at_zero = 0;
end
if nargin<=5
    verbose_flag = 1;
end
if nargin<=6
    newfigure=0;
end

%nb = sqrt(sum(col(conj(b).*b)));
%tolb = tol * nb;
if start_at_zero
    d = b;
else
    d = b - A*z;
end
r = d;
reference_norm=sum(col(conj(b) .* b));
normrr0 = sum(col(conj(r) .* r));
normrr = normrr0;
if nargout==2
    z_iter = cell(1,(1));
end

for iter=1:maxit(1)
    Ad = A*d;
    Ad2 = A2*d;
    tmp = conj(d) .* Ad;
    alpha = normrr/sum(col(tmp));
    z = z + alpha*d;
    r = r - alpha*Ad;
    normrr2 = sum(col(conj(r) .* r));
	beta = normrr2/normrr;
    normrr = normrr2;
	d = r + beta*d;
    
    if nargout==2
        if mod(iter,10)==0
            z_iter{iter} = z;
        end
    end
    
    if verbose_flag==1
        if isempty(h) || ~ishandle(h) || ~strcmp(get(h,'Tag'),'cg_figure') || (newfigure && iter==1)
            h = figure('Tag','cg_figure');
        end
        if gcf~=h
            set(0,'CurrentFigure',h);
        end
        if isnumeric(z) && isvector(z)
            plot(abs(z));
        elseif isnumeric(z) && is2Darray(z)
            imagesc(abs(z));
            colormap gray;
        elseif isnumeric(z) && length(size(z))==3
            imagesc(abs(array2mosaic(gather(z))));
            colormap gray;
        else
            imagesc(z);
            colormap gray;
        end
        axis off;
        title(['Iteration #', num2str(iter),': current tolerance = ', num2str(sqrt(normrr/normrr0)), '.']);
        drawnow;
    elseif verbose_flag==2
        fprintf(['Iteration #', num2str(iter),': current tolerance = ', num2str(sqrt(normrr/normrr0)), '\n']);
    end
    update1=sqrt(normrr/normrr0);
    z_iter{3}(iter)=update1;
    update2=sqrt(normrr/reference_norm);
    z_iter{5}(iter)=update2;

%     for ii=1:length(maxit)
%         if iter==maxit(ii)
%             z_keep{1}{ii}=z;
%             z_keep{2}(ii)=maxit(ii);
%         end
%     end
    % z_keep{iter}=z;
%     if iter>1
        if z_iter{5}(iter) < tol
            break
        end
%     end
    
end
% if iter<maxit
    z_keep{1}{1}=z;
    z_keep{2}(1)=iter;
% end
z_iter{1}=z_keep;