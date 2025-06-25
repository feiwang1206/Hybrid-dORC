classdef SpatialUnflattenLayer3d < nnet.layer.Layer & nnet.layer.Formattable
    % SPATIALUNFLATTENLAYER  Custom layer that converts a dlarray with one
    % spatial dimension into a dlarray with two spatial dimensions.
    % This layer is intendended to be used in conjunction with
    % SpatialFlattenLayer to reconstruct the correct image dimensions.
    
    %   Copyright 2023 The MathWorks, Inc.
    properties
        flatSize
    end
    
    methods
        function layer = SpatialUnflattenLayer3d(opts)
           arguments
               opts.Name = "unflatten"
               opts.flatSize=[];
           end
            layer.Name = opts.Name;
            layer.flatSize=opts.flatSize;
        end

        function out = predict(this, in)
            % Validate the input
            if ~strcmp(dims(in), "SCB")
                error("Input must be a dlarray with one spatial, one channel, and one batch dimension");
            % elseif sqrt(size(in,1)) ~= floor(sqrt(size(in,1)))
            %     error("Size of the spatial dimension of the input must be a square number.");
            end

            inSize = size(in);
            outSize = [this.flatSize inSize(2) inSize(3)];
            out = reshape(in, outSize);
            out = dlarray(out, "SSSCB");
        end
    end
end