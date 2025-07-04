classdef SpatialFlattenLayer3d < nnet.layer.Layer & nnet.layer.Formattable
    % SPATIALFLATTENLAYER  Custom layer that reshapes the spatial
    % dimensions of a 3-D image to a single spatial dimension.
    
    %   Copyright 2023 The MathWorks, Inc. 

    methods
        function layer = SpatialFlattenLayer3d(opts)
           arguments
               opts.Name = "flatten"
           end
            layer.Name = opts.Name;
        end

        function out = predict(~, in)
            % Validate the input
            if ~strcmp(dims(in), 'SSSCB')
                error("Input must be a dlarray object with three spatial dimensions, one channel dimension, and one batch dimension.");
            elseif size(in,1) ~= size(in,2)
                error("Input must be a square image.");
            end

            inSize = size(in);
            outSize = [inSize(1)*inSize(2)*inSize(3) inSize(4) inSize(5)];
            out = reshape(in, outSize);
            out = dlarray(out, "SCB");
        end
    end
end