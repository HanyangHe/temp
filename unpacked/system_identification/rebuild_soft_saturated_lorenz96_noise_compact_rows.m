function [noiseRows,info] = rebuild_soft_saturated_lorenz96_noise_compact_rows( ...
    sourceRoot,noiseLevels,numRounds,nTrain,varargin)
%REBUILD_SOFT_SATURATED_LORENZ96_NOISE_COMPACT_ROWS
% Rebuild the compact robustness row table directly from rho-specific method
% checkpoints. This deliberately never falls back to the large aggregate
% allResults variable. Each PhDN checkpoint is loaded once, reduced to one
% compact PhDN row plus its collected Stage-0 SR row, then immediately cleared.
%
% This is intended as a one-time recovery/migration path when the aggregate
% noiseRows variable is stale after an interrupted long robustness sweep.

p = inputParser;
addParameter(p,'IncludeStage0SRAblations',true,@(x)islogical(x)||isnumeric(x));
addParameter(p,'IncludeMLP',true,@(x)islogical(x)||isnumeric(x));
addParameter(p,'RequireCompletePhDN',true,@(x)islogical(x)||isnumeric(x));
addParameter(p,'Verbose',true,@(x)islogical(x)||isnumeric(x));
parse(p,varargin{:});
includeSR = logical(p.Results.IncludeStage0SRAblations);
includeMLP = logical(p.Results.IncludeMLP);
requireCompletePhDN = logical(p.Results.RequireCompletePhDN);
verbose = logical(p.Results.Verbose);

noiseRows = struct([]);
missingPhDN = strings(0,1);
missingMLP = strings(0,1);
loadedPhDN = 0;
loadedSR = 0;
loadedMLP = 0;
wallTimer = tic;

phdnLabels = {'PhDN-G1','PhDN-G2','PhDN-G3'};
phdnFields = {'phdn_g1','phdn_g2','phdn_g3'};
srLabels = {'Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3'};

for iRound = 1:numRounds
    roundKey = sprintf('round_%02d',iRound);
    for iNoise = 1:numel(noiseLevels)
        noiseLevel = noiseLevels(iNoise);
        noiseKey = soft_saturated_lorenz96_noise_key(noiseLevel);
        methodDir = fullfile(sourceRoot,roundKey,noiseKey,'method_results');

        for iG = 1:3
            methodPath = fullfile(methodDir,[phdnFields{iG} '.mat']);
            if exist(methodPath,'file') ~= 2
                missingPhDN(end+1,1) = string(methodPath); %#ok<AGROW>
                continue;
            end

            if verbose
                fprintf('Compact-row rebuild: %s | round=%d | noise=%.1f%%\n', ...
                    phdnLabels{iG},iRound,100*noiseLevel);
            end
            loadTimer = tic;
            loaded = load(methodPath,'result');
            loadSeconds = toc(loadTimer);
            if ~isfield(loaded,'result') || ~isstruct(loaded.result) || isempty(loaded.result)
                warning(['Malformed PhDN checkpoint ignored: ' methodPath]);
                missingPhDN(end+1,1) = string(methodPath); %#ok<AGROW>
                clear loaded;
                continue;
            end

            resultPhdn = loaded.result;
            clear loaded;
            [noiseMatches,storedNoiseLevel] = ...
                soft_saturated_lorenz96_checkpoint_matches_noise(methodPath,noiseLevel);
            if ~noiseMatches
                warning(['Ignoring stale PhDN checkpoint from a different rho: ' methodPath]);
                if verbose && isfinite(storedNoiseLevel)
                    fprintf('  stored rho=%g, requested rho=%g; checkpoint will not be relabeled.\n', ...
                        storedNoiseLevel,noiseLevel);
                end
                missingPhDN(end+1,1) = string(methodPath); %#ok<AGROW>
                clear resultPhdn;
                continue;
            end
            noiseRows = append_system_identification_result_row( ...
                noiseRows,nTrain,iRound,phdnLabels{iG},resultPhdn,[]);
            noiseRows(end).noiseLevel = noiseLevel;
            noiseRows(end).noisePercent = 100*noiseLevel;
            if isfield(resultPhdn,'noiseRobustness') && ...
                    isfield(resultPhdn.noiseRobustness,'noiseSeed')
                noiseRows(end).noiseSeed = resultPhdn.noiseRobustness.noiseSeed;
            else
                noiseRows(end).noiseSeed = NaN;
            end
            loadedPhDN = loadedPhDN+1;

            if includeSR
                try
                    [~,resultStage0SR] = attach_soft_saturated_lorenz96_stage0_sr_ablation(resultPhdn);
                catch ME
                    warning(['Could not recover Stage-0 SR from ' methodPath ': ' ME.message]);
                    resultStage0SR = [];
                end
                if ~isempty(resultStage0SR) && ...
                        (~isfield(resultStage0SR,'available') || logical(resultStage0SR.available))
                    noiseRows = append_system_identification_result_row( ...
                        noiseRows,nTrain,iRound,srLabels{iG},resultStage0SR,[]);
                    noiseRows(end).noiseLevel = noiseLevel;
                    noiseRows(end).noisePercent = 100*noiseLevel;
                    if isfield(resultPhdn,'noiseRobustness') && ...
                            isfield(resultPhdn.noiseRobustness,'noiseSeed')
                        noiseRows(end).noiseSeed = resultPhdn.noiseRobustness.noiseSeed;
                    else
                        noiseRows(end).noiseSeed = NaN;
                    end
                    loadedSR = loadedSR+1;
                end
                clear resultStage0SR;
            end

            clear resultPhdn;
            if verbose
                fprintf('  direct checkpoint load %.2f s; compact rows retained only.\n',loadSeconds);
            end
        end

        if includeMLP
            mlpPath = fullfile(methodDir,'mlp.mat');
            if exist(mlpPath,'file') == 2
                [noiseMatches,storedNoiseLevel] = ...
                    soft_saturated_lorenz96_checkpoint_matches_noise(mlpPath,noiseLevel);
                if ~noiseMatches
                    if verbose && isfinite(storedNoiseLevel)
                        fprintf('Ignoring stale MLP checkpoint: stored rho=%g, requested rho=%g.\n', ...
                            storedNoiseLevel,noiseLevel);
                    end
                    missingMLP(end+1,1) = string(mlpPath); %#ok<AGROW>
                else
                    loaded = load(mlpPath,'result');
                    if isfield(loaded,'result') && isstruct(loaded.result) && ~isempty(loaded.result)
                        resultMlp = loaded.result;
                        noiseRows = append_system_identification_result_row( ...
                            noiseRows,nTrain,iRound,'MLP',resultMlp,[]);
                        noiseRows(end).noiseLevel = noiseLevel;
                        noiseRows(end).noisePercent = 100*noiseLevel;
                        if isfield(resultMlp,'noiseRobustness') && ...
                                isfield(resultMlp.noiseRobustness,'noiseSeed')
                            noiseRows(end).noiseSeed = resultMlp.noiseRobustness.noiseSeed;
                        else
                            noiseRows(end).noiseSeed = NaN;
                        end
                        loadedMLP = loadedMLP+1;
                        clear resultMlp;
                    end
                    clear loaded;
                end
            else
                missingMLP(end+1,1) = string(mlpPath); %#ok<AGROW>
            end
        end
    end
end

% Sort and guard against accidental duplicate keys.
if ~isempty(noiseRows)
    methods = string({noiseRows.method});
    rounds = [noiseRows.roundIndex];
    levels = [noiseRows.noiseLevel];
    [~,order] = sortrows([rounds(:),levels(:),double(categorical(methods(:)))],[1 2 3]);
    noiseRows = noiseRows(order);
end

expectedPhDN = 3*numRounds*numel(noiseLevels);
if requireCompletePhDN && loadedPhDN ~= expectedPhDN
    preview = strjoin(cellstr(missingPhDN(1:min(6,end))),newline);
    error(['Compact-row rebuild found %d/%d PhDN checkpoints. Missing examples:\n%s\n', ...
        'No large aggregate allResults fallback was attempted.'], ...
        loadedPhDN,expectedPhDN,preview);
end

info = struct();
info.loadedPhDN = loadedPhDN;
info.loadedStage0SR = loadedSR;
info.loadedMLP = loadedMLP;
info.expectedPhDN = expectedPhDN;
info.missingPhDN = missingPhDN;
info.missingMLP = missingMLP;
info.wallSeconds = toc(wallTimer);
info.usedAggregateAllResults = false;

if verbose
    fprintf(['Compact-row rebuild complete: PhDN=%d/%d, Stage0-SR=%d, MLP=%d, ', ...
        'wall=%.2f min.\n'],loadedPhDN,expectedPhDN,loadedSR,loadedMLP,info.wallSeconds/60);
    if ~isempty(missingMLP)
        fprintf('MLP is optional here: %d MLP checkpoint(s) were absent.\n',numel(missingMLP));
    end
end
end
