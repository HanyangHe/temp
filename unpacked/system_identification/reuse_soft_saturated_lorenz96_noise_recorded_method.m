function [methodResult,replayInfo] = reuse_soft_saturated_lorenz96_noise_recorded_method( ...
    sourceRoot,roundIndex,noiseLevel,methodField,methodLabel,strictMode,task,compactMode,allowLargeAggregateFallback)
%REUSE_SOFT_SATURATED_LORENZ96_NOISE_RECORDED_METHOD Replay one saved noise-case method.
%
% The derivative-noise experiment is keyed by (round, rho), not by
% (round, Ntrain). Dedicated checkpoints are stored as
%   round_XX/<rho-key>/method_results/<methodField>.mat
% by save_soft_saturated_lorenz96_noise_run. Each file contains variable
% "result". This loader intentionally does not use the legacy N_01000
% sample-efficiency replay path.

    if nargin < 9 || isempty(allowLargeAggregateFallback); allowLargeAggregateFallback = false; end
    if nargin < 8 || isempty(compactMode); compactMode = false; end
    if nargin < 7; task = []; end
    if nargin < 6 || isempty(strictMode); strictMode = true; end
    if nargin < 5 || isempty(methodLabel); methodLabel = upper(char(methodField)); end

    methodField = lower(strtrim(char(methodField)));
    methodLabel = char(methodLabel);
    noisePercent = 100*double(noiseLevel);
    roundKey = sprintf('round_%02d',roundIndex);
    noiseKey = soft_saturated_lorenz96_noise_key(noiseLevel);
    methodPath = fullfile(sourceRoot,roundKey,noiseKey,'method_results',[methodField '.mat']);

    replayInfo = struct();
    replayInfo.available = false;
    replayInfo.methodField = methodField;
    replayInfo.methodLabel = methodLabel;
    replayInfo.roundIndex = roundIndex;
    replayInfo.noiseLevel = noiseLevel;
    replayInfo.noisePercent = noisePercent;
    replayInfo.sourcePath = '';
    replayInfo.reportMode = '';
    replayInfo.compactMode = logical(compactMode);
    replayInfo.loadInfo = struct('loadMode','', 'directMethodPath',methodPath, ...
        'aggregatePath','', 'directRecordFound',false);
    methodResult = [];

    loadTimer = tic;
    if exist(methodPath,'file') == 2
        [noiseMatches,storedNoiseLevel] = ...
            soft_saturated_lorenz96_checkpoint_matches_noise(methodPath,noiseLevel);
        if ~noiseMatches
            if isfinite(storedNoiseLevel)
                warning(['Ignoring stale noise checkpoint with stored rho=' num2str(storedNoiseLevel) ...
                    ' while requested rho=' num2str(noiseLevel) ': ' methodPath]);
            end
        else
            loaded = load(methodPath,'result');
            if isfield(loaded,'result') && isstruct(loaded.result) && ~isempty(loaded.result)
                methodResult = loaded.result;
                replayInfo.sourcePath = methodPath;
                replayInfo.loadInfo.loadMode = 'noise_direct_method_result';
                replayInfo.loadInfo.directRecordFound = true;
            else
                warning(['Ignoring malformed noise method checkpoint: ' methodPath]);
            end
        end
    end

    % Robust fallback: use the aggregate noise MAT if the per-rho checkpoint
    % is unavailable. This also supports old completed sweeps that retained
    % allResults but lost one of the small method_results files.
    if isempty(methodResult) && allowLargeAggregateFallback
        aggregatePath = fullfile(sourceRoot,'summary', ...
            'soft_saturated_lorenz96_noise_robustness_results.mat');
        replayInfo.loadInfo.aggregatePath = aggregatePath;
        if exist(aggregatePath,'file') == 2
            aggregate = load(aggregatePath,'allResults');
            if isfield(aggregate,'allResults') && isstruct(aggregate.allResults) && ...
                    isfield(aggregate.allResults,roundKey) && ...
                    isfield(aggregate.allResults.(roundKey),noiseKey) && ...
                    isfield(aggregate.allResults.(roundKey).(noiseKey),methodField)
                candidate = aggregate.allResults.(roundKey).(noiseKey).(methodField);
                if isstruct(candidate) && ~isempty(candidate)
                    methodResult = candidate;
                    replayInfo.sourcePath = aggregatePath;
                    replayInfo.loadInfo.loadMode = 'noise_aggregate_allResults';
                end
            end
        end
    end
    replayInfo.loadInfo.loadSeconds = toc(loadTimer);

    if isempty(methodResult)
        message = sprintf(['Saved %s noise-case result was not found for %s/%s under:\n%s\n', ...
            'Expected dedicated checkpoint:\n%s\n', ...
            'Run the complete noise case once or point RecordedBaselineSourceRoot ', ...
            'to a valid noise-robustness output tree. Large aggregate fallback is disabled by default.'], ...
            methodLabel,roundKey,noiseKey,sourceRoot,methodPath);
        if strictMode
            error('%s',message);
        else
            warning('%s',message);
            return;
        end
    end

    isStage0SR = startsWith(methodField,'stage0sr_') || ...
        startsWith(methodField,'stage0_sr_') || strcmp(methodField,'stage0sr');

    % Stage-0 SR is a collected ablation object rather than a conventional
    % independently trained baseline. It has its own report printer and does
    % not require record_soft_saturated_lorenz96_method_report.
    if isStage0SR
        replayInfo.reportMode = 'saved_stage0_sr_ablation_report';
        methodResult.reusedRecordedBaseline = true;
        methodResult.reusedRecordedMethod = true;
        methodResult.recordedReplaySourcePath = replayInfo.sourcePath;
        methodResult.recordedReplayReportMode = replayInfo.reportMode;
        methodResult.recordedReplayRoundIndex = roundIndex;
        methodResult.recordedReplayNoiseLevel = noiseLevel;
        replayInfo.available = true;

        fprintf('\n============================================================\n');
        fprintf('RECORDED NOISE METHOD REPLAY: %s (training skipped)\n',methodLabel);
        fprintf('Noise key: %s | source: %s\n',noiseKey,replayInfo.sourcePath);
        fprintf('Replay load: %.3f s | mode=%s\n', ...
            replayInfo.loadInfo.loadSeconds,replayInfo.loadInfo.loadMode);
        fprintf('Report mode: %s | compact=%d\n',replayInfo.reportMode,logical(compactMode));
        fprintf('============================================================\n');
        print_stage0_sr_ablation_result(methodResult,logical(compactMode));
        return;
    end

    if isfield(methodResult,'recordedConsoleReport') && ...
            ~isempty(methodResult.recordedConsoleReport)
        reportText = char(methodResult.recordedConsoleReport);
        replayInfo.reportMode = 'exact_saved_report';
    else
        try
            methodResult = record_soft_saturated_lorenz96_method_report( ...
                methodResult,methodField,task);
            reportText = char(methodResult.recordedConsoleReport);
            replayInfo.reportMode = 'saved_result_reconstructed_report';
        catch MEreport
            message = sprintf(['Saved %s noise-case result was loaded from %s, but its ', ...
                'report could not be reconstructed: %s'], ...
                methodLabel,replayInfo.sourcePath,MEreport.message);
            if strictMode
                error('%s',message);
            else
                warning('%s',message);
                return;
            end
        end
    end

    methodResult.reusedRecordedBaseline = true;
    methodResult.reusedRecordedMethod = true;
    methodResult.recordedReplaySourcePath = replayInfo.sourcePath;
    methodResult.recordedReplayReportMode = replayInfo.reportMode;
    methodResult.recordedReplayRoundIndex = roundIndex;
    methodResult.recordedReplayNoiseLevel = noiseLevel;
    replayInfo.available = true;

    fprintf('\n============================================================\n');
    fprintf('RECORDED NOISE METHOD REPLAY: %s (training skipped)\n',methodLabel);
    fprintf('Noise key: %s | source: %s\n',noiseKey,replayInfo.sourcePath);
    fprintf('Replay load: %.3f s | mode=%s\n', ...
        replayInfo.loadInfo.loadSeconds,replayInfo.loadInfo.loadMode);
    fprintf('Report mode: %s | compact=%d\n',replayInfo.reportMode,logical(compactMode));
    fprintf('============================================================\n');

    displayReportText = reportText;
    if logical(compactMode)
        displayReportText = compact_recorded_report_local(reportText,methodField);
    end
    fprintf('%s',displayReportText);
    if ~isempty(displayReportText) && displayReportText(end) ~= sprintf('\n')
        fprintf('\n');
    end
end

function compactText = compact_recorded_report_local(reportText,methodField)
%COMPACT_RECORDED_REPORT_LOCAL Minimal replay display; saved report is untouched.
    compactText = char(reportText);
    if isempty(compactText); return; end
    methodField = lower(strtrim(char(methodField)));
    lines = regexp(compactText,'\r\n|\n|\r','split');
    keep = false(size(lines));
    commonPrefixes = { ...
        'Demo finished for task:', 'Final model operator mode:', 'Best validation MSE', ...
        'Data source', 'Method', 'Protocol', 'Selected parameter count', ...
        'Complete sweep wall time', 'Selected-candidate train time', ...
        'Selected-candidate wall time', 'Training wall time', ...
        'Validation MSE/RMSE', 'In-distribution test MSE/RMSE', 'OOD test MSE/RMSE', ...
        'Train MSE', 'Val   MSE', 'Test  MSE', 'OOD   MSE'};
    if startsWith(methodField,'phdn')
        extraPrefixes = {'true-operator PhDN Test MSE','OOD true-operator PhDN Test MSE', ...
            'active terms','inv / exp / sqrt / cross','identity-cancellation'};
    elseif strcmp(methodField,'mlp')
        extraPrefixes = {'Selected depth','Selected hidden layers','Selected activation'};
    elseif strcmp(methodField,'eql')
        extraPrefixes = {'Selected paper depth L','Official hidden layers L-1', ...
            'Selected sparsity lambda','Selected active units','Parameters / active weights', ...
            'Active coefficients total','Selected model training N'};
    elseif strcmp(methodField,'kan')
        extraPrefixes = {'Selected depth/width','Selected validation-best grid', ...
            'Selected sparsification lambda','Selected structure source', ...
            'Final selected shape','Active edges / coefficients'};
    elseif strcmp(methodField,'sindy') || strcmp(methodField,'neural_sindy')
        extraPrefixes = {'Dictionary source','Declared neural-ridge bases', ...
            'Evaluated neural-ridge columns','Library rows total / used', ...
            'Selected STLSQ threshold','Active terms / coefficients'};
    else
        extraPrefixes = {'Selected depth','Selected width','Selected activation', ...
            'Selected lambda','Selected grid','Active terms','Active coefficients','Parameter count'};
    end
    prefixes = [commonPrefixes,extraPrefixes];
    for i = 1:numel(lines)
        s = strtrim(lines{i});
        if isempty(s); continue; end
        for j = 1:numel(prefixes)
            if startsWith(s,prefixes{j},'IgnoreCase',true)
                keep(i) = true;
                break;
            end
        end
    end
    selected = lines(keep);
    header = {'[minimal replay report: historical sweep/candidate/term details omitted]', ...
              '[full archived report remains unchanged in recordedConsoleReport]'};
    if isempty(selected)
        compactText = strjoin(header,sprintf('\n'));
    else
        compactText = strjoin([header,selected],sprintf('\n'));
    end
    compactText = [compactText sprintf('\n')];
end
