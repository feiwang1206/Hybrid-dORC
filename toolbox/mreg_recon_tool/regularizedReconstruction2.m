function [z, z_iter] = regularizedReconstruction2(A,b,varargin)

% usage:
%   [z, z_iter] = regularizedReconstruction(A,b,varargin)
%
% A = forward operator or matrix
% b = right hand side of forward model
%
% varargin contains the handle to the pernalty followed by the
% input arguments of that constraint.
%
% example:
%   z = nonlinearRegularization(A,b,@L1Norm,lambda);
%   (performs a regularization with lambda*norm(z,1) as constraint)
%
% more constraints are also possible, e.g.:
%   z = nonlinearRegularization(A,b,@L2Norm,lambda1,@L1Norm,lambda2,Dx,Dy);
%   constraint = lambda1*z'*z + lambda2* (norm(Dx*z,1) + norm(Dy*z,1))
%
% and so on ...
%
% varargin can also contain optional string keywords followed by some value, e.g.
%   z = nonlinearRegularization(A,b,@L1Norm,lambda,'tol',1e-5,'cg_method','pr');
%
% allowed string keywords are:
%  'tol' to set the stopping tolerance
%  'maxit' to set the maximum iterations
%  'verbose' use 0 to switch to silent mode
%  'cg_method' to set the CG-Method
%  'z0' to set a starting point
%
% 30.09.2011
% Thimo Hugger

tol = 1e-6;
maxit = 50;
verbose_flag = 1;
z0 = [];
cg_method = 'fr+pr';
b = double(b); % since it is loaded as single
newfigure = 0;

I_all = [];
for k=1:length(varargin)
    if ischar(varargin{k}) || strcmp(class(varargin{k}), 'function_handle')
        I_all(end+1) = k;
    end
end
I_all(end+1)=length(varargin)+1;

for k=1:length(varargin)
    if ischar(varargin{k})
        keyword = varargin{k};
        switch keyword
            case 'tol'
                tol = varargin{k+1};
            case 'maxit'
                maxit = varargin{k+1};
            case 'verbose_flag'
                verbose_flag = varargin{k+1};
            case 'z0'
                z0 = varargin{k+1};
                if isempty(z0) || prod(size(z0))==1
                    z0 = [];
                else
                    gamma = max(col(abs(A*z0))) / max(abs(b(:)));
                    z0 = z0 / (gamma+1e-15); % adjust the amplitude of the initial guess according to the data 'b'
                end
            case 'cg_method'
                cg_method = varargin{k+1};
            case 'newfigure'
                newfigure = 1;
            otherwise
                if ~strcmp(varargin{k-1}, 'cg_method')
                    error(['regularizedReconstruction: Variable ''', keyword, ''' is unknown']);
                end
        end
    end
end

I_fhandle = [];
for k=1:length(varargin)
    if strcmp( class(varargin{k}), 'function_handle' )
        I_fhandle(end+1) = k;
    end
end

linear_flag = 1;
for k=1:length(I_fhandle)
    linear_flag = linear_flag * strcmp(func2str(varargin{I_fhandle(k)}),'L2Norm');
end

if linear_flag==0 % use nonlinear conjugate gradient
    R = {};
    dR = {};
    for k=1:length(I_fhandle)
        n = find(I_fhandle(k)==I_all);
        [tmp, dtmp] = varargin{I_all(k)}(varargin{I_all(n)+1:I_all(n+1)-1});
        R{k} = tmp;
        dR{k} = dtmp;
    end
    
    if isempty(z0)
        z0=zeros([size(A,2), 1]);    
    end
    reference_norm = l2norm(A'*b);
    
    [f,df] = cost_function;
    
    if nargout==1
        [z] = nonlinearConjugateGradient1(f,df,z0,tol,maxit,cg_method,verbose_flag,reference_norm);
    else
        [z, z_iter] = nonlinearConjugateGradient1(f,df,z0,tol,maxit,cg_method,verbose_flag,reference_norm);
    end
    
else % use normal conjugate gradient
    R{1} = @(x) A'*(A*x);
    counter = 2;
    for k=1:length(I_fhandle)
        n = find(I_fhandle(k)==I_all);
        if I_all(n+1)==I_all(n)+2;
            R{counter} = @(z) varargin{I_fhandle(k)+1}^2*z;
            counter = counter + 1;
        else
            for l=2:(I_all(n+1)-I_all(n)-1)
                R{counter} = @(z) varargin{I_fhandle(k)+1}^2*( varargin{I_all(n)+l}' * (varargin{I_all(n)+l} * z) );
                counter = counter + 1;
            end
        end
    end
    if isempty(z0)
        z0=zeros([size(A,2), 1]);    
    end
    
    f = left_hand_side;
    S = virtualMatrix(f,size(A,2));
    
    q = A'*b;
    
    if nargout==1
        z = conjugateGradient(S,q,tol,maxit,z0,verbose_flag,newfigure);
    else
        [z,z_iter] = conjugateGradient(S,q,tol,maxit,z0,verbose_flag,newfigure);
    end    
end


    function [hLS, hdLS] = residualNorm

        function [y1, re] = LS(z,alpha,d,update_flag) % nargin>=2 is meant for efficient line search
            persistent Azb Ad;
            if nargin==1
                Azb = A*z-b;
                re=Azb;
                y1 = sum(col(conj(Azb).*Azb));
            else
                if update_flag==1
                    Ad = A*d;
                end
                re=Azb;
                Ayb = Azb + alpha * Ad;
                y1 = sum(col(conj(Ayb).*Ayb));
            end
        end
        
        
        function y2 = dLS(re)
                y2 = 2*(A' * re);
        end

        hLS = @LS;
        hdLS = @dLS;
    end


    % assemble the complete cost function and return the function handle to it and its derivative
    function [hf,hdf] = cost_function
        
        [LS, dLS] = residualNorm;
        
        function [y1, re,reR] = f(z,alpha,d,update_flag) % nargin>=2 is meant for efficient line search
            if nargin==1
                [y1, re] = LS(z);
                for m=1:length(R)
                    [yR,reR{m}]=R{m}(z);
                    y1 = y1 + yR;
                end
            else
                [y1, re] = LS(z,alpha,d,update_flag);
                for m=1:length(R)
                    [yR,reR{m}]=R{m}(z,alpha,d,update_flag);
                    y1 = y1 + yR;
                end
            end
        end

        function [y2] = df(z,re,reR)
            y2 = dLS(re);
            for m=1:length(dR)
                y2 = y2 + dR{m}(z,reR{m});
            end
        end
        hf = @f;
        hdf = @df;     
    end

    % assemble the complete cost function and return the function handle to it and its derivative

    % function handle to left hand side of equation in the linear case
    function hl = left_hand_side
        
        function y = f(z)
            y = R{1}(z);
            for m=2:length(R)
                y = y + R{m}(z);
            end
        end
        
        hl = @f;
    end

end