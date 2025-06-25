function [hf, hdf] = L1Norm2(lambda,varargin)

% usage:
%   [hf, hdf] = L1Norm(lambda,varargin)
%
% varargin enthaelt Operatoren die auf 'z' wirken vor der
% Bildung der Norm, z.B.:
%   Dx = finiteDifferenceOperator(1);
%   Dy = finiteDifferenceOperator(2);
%   varargin = {Dx, Dy} fuer Berechnung der TV in 2D.
%
% 30.09.2011
% Thimo Hugger

mu = 1e-15;

if nargin==1
    varargin{1} = 1;
end

    function [y1,reR] = R(z,alpha,d,update_flag) % nargin>=2 is meant for efficient line search
        z=double(z);
        sz=size(z);
        persistent Vz Vd;
        if nargin>1 && update_flag==1
            Vz = cell(1,length(sz));
            Vd = cell(1,length(sz));
        end
        y1 = 0;
        for k=1:length(varargin{1})
            if nargin==1
                Vz{k}=varargin{1}{k}*col(z);
                reR{k}=Vz{k};
                y1 = y1 + sum(abs(Vz{k} + mu));
            else
%                 if update_flag==1
                    Vz{k} = varargin{1}{k}*col(z);
                    Vd{k} = varargin{1}{k}*col(double(d));
%                 end
                reR{k}=Vz{k};
                y1 = y1 + sum((abs(Vz{k} + alpha * Vd{k} + mu)));
            end
        end
        y1 = lambda * y1;
    end

    function y2 = dR(z,reR)
        sz=size(z);
        persistent Vz;

        y2 = 0;
        for k=1:length(varargin{1})
            Vz = reR{k};
            y2 = y2 + reshape((varargin{1}{k})'*( Vz./ sqrt(conj(Vz).*Vz + mu)),size(z));
        end
        y2 = lambda * y2;
    end

    hf = @R;
    hdf = @dR;
    
end