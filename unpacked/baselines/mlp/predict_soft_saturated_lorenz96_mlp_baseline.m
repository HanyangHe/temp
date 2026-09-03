function Y = predict_soft_saturated_lorenz96_mlp_baseline(result, X)
%PREDICT_SOFT_SATURATED_LORENZ96_MLP_BASELINE Evaluate the validation-selected Python MLP.
%
% The Python adapter exports affine-layer weights/biases and train-split
% standardization statistics.  This evaluator makes the selected sweep model
% callable at arbitrary states, including inside ode15s rollout.

    if isempty(X)
        Y = zeros(0,0);
        return;
    end
    if ~isfield(result,'modelParameters') || isempty(result.modelParameters)
        error('Selected Python MLP parameters were not exported.');
    end
    if isfield(result,'normalization') && isstruct(result.normalization)
        normInfo = result.normalization;
    elseif isfield(result,'pyResult') && isfield(result.pyResult,'normalization')
        normInfo = result.pyResult.normalization;
    else
        error('Selected Python MLP normalization statistics are unavailable.');
    end

    xMean = reshape(double(normInfo.x_mean),1,[]);
    xStd = reshape(double(normInfo.x_std),1,[]);
    yMean = reshape(double(normInfo.y_mean),1,[]);
    yStd = reshape(double(normInfo.y_std),1,[]);
    xStd(~isfinite(xStd) | xStd==0) = 1;
    yStd(~isfinite(yStd) | yStd==0) = 1;

    Z = (double(X)-xMean)./xStd;
    weights = result.modelParameters.weights;
    biases = result.modelParameters.biases;
    nLayers = sequence_length_local(weights);
    activation = lower(strtrim(char(result.modelParameters.activation)));

    for iLayer = 1:nLayers
        W = double(sequence_item_local(weights,iLayer));
        b = reshape(double(sequence_item_local(biases,iLayer)),1,[]);
        Z = Z*W.' + b;
        if iLayer < nLayers
            switch activation
                case 'tanh'
                    Z = tanh(Z);
                case 'relu'
                    Z = max(Z,0);
                case {'silu','swish'}
                    Z = Z./(1+exp(-Z));
                otherwise
                    error('Unsupported exported MLP activation: %s',activation);
            end
        end
    end
    Y = Z.*yStd+yMean;
end

function n = sequence_length_local(value)
    if iscell(value)
        n = numel(value);
    elseif isnumeric(value)
        if ismatrix(value)
            n = 1;
        else
            n = size(value,1);
        end
    elseif isstruct(value)
        n = numel(value);
    else
        error('Unsupported exported parameter container type: %s',class(value));
    end
end

function item = sequence_item_local(value,index)
    if iscell(value)
        item = value{index};
    elseif isnumeric(value)
        if ismatrix(value)
            if index~=1; error('Parameter sequence index is out of range.'); end
            item = value;
        else
            item = squeeze(value(index,:,:));
        end
    elseif isstruct(value)
        item = value(index);
    else
        error('Unsupported exported parameter container type: %s',class(value));
    end
end
