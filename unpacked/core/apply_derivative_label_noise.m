function [YtrainNoisy,YvalNoisy,info] = apply_derivative_label_noise(Ytrain,Yval,opt)
%APPLY_DERIVATIVE_LABEL_NOISE Add paired relative Gaussian derivative noise.
%
% The noise scale for output j is std(Ytrain(:,j)) computed from the clean
% training derivatives.  A single RNG seed generates one standard-normal
% realization for training and validation.  Reusing the same seed while
% changing relativeStd therefore gives a paired noise-amplitude sweep.
%
% Clean test/OOD targets are deliberately outside this helper.

    if nargin < 3 || isempty(opt); opt = struct(); end
    enable = getfield_default_local(opt,'enable',false);
    rho = double(getfield_default_local(opt,'relativeStd',0));
    seed = double(getfield_default_local(opt,'seed',1));
    applyTrain = logical(getfield_default_local(opt,'applyToTrain',true));
    applyVal = logical(getfield_default_local(opt,'applyToValidation',true));
    scaleMode = lower(strtrim(char(getfield_default_local( ...
        opt,'scaleMode','training_derivative_std'))));
    distribution = lower(strtrim(char(getfield_default_local( ...
        opt,'distribution','gaussian'))));

    YtrainNoisy = Ytrain;
    YvalNoisy = Yval;
    info = struct();
    info.enabled = logical(enable) && rho > 0;
    info.requestedEnable = logical(enable);
    info.relativeStd = rho;
    info.seed = seed;
    info.applyToTrain = applyTrain;
    info.applyToValidation = applyVal;
    info.scaleMode = scaleMode;
    info.distribution = distribution;

    if rho < 0 || ~isfinite(rho)
        error('derivativeLabelNoise.relativeStd must be finite and nonnegative.');
    end
    if ~strcmp(scaleMode,'training_derivative_std')
        error('Unsupported derivative-label noise scaleMode: %s',scaleMode);
    end
    if ~strcmp(distribution,'gaussian')
        error('Unsupported derivative-label noise distribution: %s',distribution);
    end

    scale = std(Ytrain,0,1);
    scale(~isfinite(scale)) = 0;
    info.perCoordinateScale = scale;
    info.cleanTrainStd = scale;
    info.cleanTrainRMS = sqrt(mean(Ytrain.^2,1));

    % Keep the caller RNG stream unchanged.  This makes the noise realization
    % independent of later PhDN/MLP optimization randomness.
    callerState = rng;
    cleanup = onCleanup(@() rng(callerState)); %#ok<NASGU>
    rng(seed,'twister');
    epsilonTrain = randn(size(Ytrain));
    epsilonVal = randn(size(Yval));

    if enable && rho > 0
        if applyTrain
            YtrainNoisy = Ytrain + rho .* epsilonTrain .* scale;
        end
        if applyVal
            YvalNoisy = Yval + rho .* epsilonVal .* scale;
        end
    end

    info.trainNoiseRMSEPerCoordinate = sqrt(mean((YtrainNoisy-Ytrain).^2,1));
    if isempty(Yval)
        info.validationNoiseRMSEPerCoordinate = zeros(1,size(Ytrain,2));
    else
        info.validationNoiseRMSEPerCoordinate = sqrt(mean((YvalNoisy-Yval).^2,1));
    end
    info.pairedRealizationPolicy = ...
        'same seed within a round; relativeStd only rescales fixed epsilon';
end

function value = getfield_default_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end
