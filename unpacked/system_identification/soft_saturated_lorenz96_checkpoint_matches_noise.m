function [matches,storedNoiseLevel] = soft_saturated_lorenz96_checkpoint_matches_noise(methodPath,noiseLevel)
%SOFT_SATURATED_LORENZ96_CHECKPOINT_MATCHES_NOISE
% Return true only when an existing method checkpoint belongs to the
% requested derivative-noise level. This prevents an old checkpoint in a
% reused rho folder from being mistaken for the new low-noise case.
%
% The small sibling noise_context.mat is checked first so normal resume does
% not need to load a potentially large PhDN checkpoint merely to inspect rho.

matches = false;
storedNoiseLevel = NaN;
if exist(methodPath,'file') ~= 2
    return;
end

methodDir = fileparts(methodPath);
cellDir = fileparts(methodDir);
contextPath = fullfile(cellDir,'noise_context.mat');
if exist(contextPath,'file') == 2
    try
        payload = load(contextPath,'context');
        if isfield(payload,'context') && isstruct(payload.context)
            c = payload.context;
            if isfield(c,'samplingPlan') && isstruct(c.samplingPlan) && ...
                    isfield(c.samplingPlan,'noiseLevel')
                storedNoiseLevel = double(c.samplingPlan.noiseLevel);
            elseif isfield(c,'noiseProtocol') && isstruct(c.noiseProtocol) && ...
                    isfield(c.noiseProtocol,'relativeStd')
                storedNoiseLevel = double(c.noiseProtocol.relativeStd);
            end
        end
    catch
        storedNoiseLevel = NaN;
    end
end

if isscalar(storedNoiseLevel) && isfinite(storedNoiseLevel)
    matches = abs(storedNoiseLevel-double(noiseLevel)) <= 1e-14;
    return;
end

% Fallback for older checkpoints that predate noise_context.mat.
try
    loaded = load(methodPath,'result');
catch
    return;
end
if ~isfield(loaded,'result') || ~isstruct(loaded.result) || isempty(loaded.result)
    return;
end
result = loaded.result;
if isfield(result,'noiseRobustness') && isstruct(result.noiseRobustness)
    if isfield(result.noiseRobustness,'relativeStd')
        storedNoiseLevel = double(result.noiseRobustness.relativeStd);
    elseif isfield(result.noiseRobustness,'noiseLevel')
        storedNoiseLevel = double(result.noiseRobustness.noiseLevel);
    end
end
if ~isfinite(storedNoiseLevel) && isfield(result,'data') && isstruct(result.data) && ...
        isfield(result.data,'derivativeLabelNoise') && ...
        isstruct(result.data.derivativeLabelNoise)
    d = result.data.derivativeLabelNoise;
    if isfield(d,'relativeStd')
        storedNoiseLevel = double(d.relativeStd);
    end
end
if isscalar(storedNoiseLevel) && isfinite(storedNoiseLevel)
    matches = abs(storedNoiseLevel-double(noiseLevel)) <= 1e-14;
end
end
