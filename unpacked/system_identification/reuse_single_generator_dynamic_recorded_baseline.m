function [methodResult, replayInfo] = reuse_single_generator_dynamic_recorded_baseline( ...
    sourceRoot,roundIndex,nTrain,methodField,methodLabel,strictMode,task,currentPhdnOpts,compactMode)
%REUSE_SINGLE_GENERATOR_DYNAMIC_RECORDED_BASELINE Load and replay a saved method.
%
% Independent method records are authoritative. Legacy result_pack.mat and
% aggregate summary files remain supported as fallbacks. The loader continues
% through every fallback even when an existing legacy pack lacks the requested
% method, preventing a later PhDN-only save from hiding an older MLP/EQL/KAN
% result stored elsewhere.

    if nargin < 9 || isempty(compactMode); compactMode = false; end
    if nargin < 8; currentPhdnOpts = []; end
    if nargin < 7; task = []; end
    if nargin < 6 || isempty(strictMode); strictMode = true; end
    if nargin < 5 || isempty(methodLabel); methodLabel = upper(char(methodField)); end
    methodField = lower(strtrim(char(methodField)));
    methodLabel = char(methodLabel);

    replayInfo = struct();
    replayInfo.available = false;
    replayInfo.methodField = methodField;
    replayInfo.methodLabel = methodLabel;
    replayInfo.roundIndex = roundIndex;
    replayInfo.nTrain = nTrain;
    replayInfo.sourcePath = '';
    replayInfo.reportMode = '';
    replayInfo.compactMode = logical(compactMode);
    replayInfo.loadInfo = struct();
    methodResult = [];

    % Fast replay path: load ONLY the requested independent method record.
    % The previous implementation rebuilt the entire persisted result pack for
    % every replayed method, repeatedly loading unrelated large .mat files.
    directLoadTimer = tic;
    [methodResult,directInfo] = load_requested_method_direct_local( ...
        sourceRoot,roundIndex,nTrain,methodField);
    replayInfo.loadInfo = directInfo;
    replayInfo.loadInfo.directLoadSeconds = toc(directLoadTimer);

    % Legacy fallback is used only if the authoritative independent method
    % record is absent/malformed. This preserves compatibility with old archives.
    if isempty(methodResult)
        fallbackTimer = tic;
        [savedPack,loadInfo] = load_single_generator_dynamic_persisted_result_pack( ...
            sourceRoot,roundIndex,nTrain,false);
        loadInfo.directLoadAttempt = directInfo;
        loadInfo.fallbackLoadSeconds = toc(fallbackTimer);
        replayInfo.loadInfo = loadInfo;
        if isstruct(savedPack) && isfield(savedPack,methodField) && ...
                ~isempty(savedPack.(methodField))
            methodResult = savedPack.(methodField);
        end
    else
        savedPack = struct();
    end

    if isempty(methodResult)
        roundKey = sprintf('round_%02d',roundIndex);
        sampleKey = sprintf('N_%05d',nTrain);
        message = sprintf(['Saved %s result was not found for %s/%s under:\n%s\n', ...
            'Run the complete method once or point RecordedBaselineSourceRoot ', ...
            'to a valid archived output tree.'], ...
            methodLabel,roundKey,sampleKey,sourceRoot);
        if strictMode
            error('%s',message);
        else
            warning('%s',message);
            return;
        end
    end

    if isfield(replayInfo.loadInfo,'directSourcePath') && ...
            ~isempty(replayInfo.loadInfo.directSourcePath)
        replayInfo.sourcePath = replayInfo.loadInfo.directSourcePath;
    elseif isfield(replayInfo.loadInfo,'methodSourcePaths') && ...
            isfield(replayInfo.loadInfo.methodSourcePaths,methodField)
        replayInfo.sourcePath = replayInfo.loadInfo.methodSourcePaths.(methodField);
    elseif isfield(replayInfo.loadInfo,'legacyPackPath') && ...
            ~isempty(replayInfo.loadInfo.legacyPackPath)
        replayInfo.sourcePath = replayInfo.loadInfo.legacyPackPath;
    elseif isfield(replayInfo.loadInfo,'aggregatePath') && ...
            ~isempty(replayInfo.loadInfo.aggregatePath)
        replayInfo.sourcePath = replayInfo.loadInfo.aggregatePath;
    else
        replayInfo.sourcePath = sourceRoot;
    end

    if ~isstruct(methodResult) || isempty(methodResult)
        message = sprintf('Saved %s result is empty in %s.',methodLabel,replayInfo.sourcePath);
        if strictMode; error('%s',message); else; warning('%s',message); methodResult=[]; return; end
    end

    validate_sample_count_local(methodResult,nTrain,methodLabel,replayInfo.sourcePath,strictMode);

    if strcmp(methodField,'sindy')
        [priorCompatible,priorMessage] = validate_sindy_guess_prior_local( ...
            methodResult,currentPhdnOpts,task,replayInfo.sourcePath);
        if ~priorCompatible
            if strictMode
                error('%s',priorMessage);
            else
                warning(['%s\nBackup/non-strict replay continues with the archived ', ...
                    'SINDy model and its saved dictionary. No retraining is performed.'], ...
                    priorMessage);
            end
        end
    end

    if isfield(methodResult,'recordedConsoleReport') && ...
            ~isempty(methodResult.recordedConsoleReport)
        reportText = char(methodResult.recordedConsoleReport);
        replayInfo.reportMode = 'exact_saved_report';
    else
        try
            methodResult = record_single_generator_dynamic_method_report( ...
                methodResult,methodField,task);
        catch MEreport
            message = sprintf(['Saved %s result was loaded from %s, but its legacy ', ...
                'report could not be reconstructed: %s'], ...
                methodLabel,replayInfo.sourcePath,MEreport.message);
            if strictMode
                error('%s',message);
            else
                warning('%s',message);
                return;
            end
        end
        reportText = char(methodResult.recordedConsoleReport);
        replayInfo.reportMode = 'legacy_result_reconstructed_report';
    end

    methodResult.reusedRecordedBaseline = true;
    methodResult.reusedRecordedMethod = true;
    methodResult.recordedReplaySourcePath = replayInfo.sourcePath;
    methodResult.recordedReplayReportMode = replayInfo.reportMode;
    methodResult.recordedReplayRoundIndex = roundIndex;
    methodResult.recordedReplayNTrain = nTrain;
    replayInfo.available = true;

    fprintf('\n============================================================\n');
    fprintf('RECORDED METHOD REPLAY: %s (training skipped)\n',methodLabel);
    fprintf('Source: %s\n',replayInfo.sourcePath);
    if isfield(replayInfo.loadInfo,'directLoadSeconds')
        fprintf('Replay load: %.3f s | mode=%s\n', ...
            replayInfo.loadInfo.directLoadSeconds, ...
            char(getfield_default_local(replayInfo.loadInfo,'loadMode','unknown')));
    end
    if isfield(replayInfo.loadInfo,'fallbackLoadSeconds')
        fprintf('Replay fallback load: %.3f s\n',replayInfo.loadInfo.fallbackLoadSeconds);
    end
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

function [result,info] = load_requested_method_direct_local( ...
    sourceRoot,roundIndex,nTrain,methodField)
%LOAD_REQUESTED_METHOD_DIRECT_LOCAL Read only the requested method .mat file.
% This is the authoritative fast path for replay. It intentionally does not
% scan/load unrelated method files, result_pack.mat, or aggregate summaries.

    result = [];
    roundKey = sprintf('round_%02d',roundIndex);
    sampleKey = sprintf('N_%05d',nTrain);
    methodDir = fullfile(sourceRoot,roundKey,sampleKey,'method_results');
    methodPath = fullfile(methodDir,[methodField '.mat']);

    info = struct();
    info.loadMode = 'direct_independent_method';
    info.directSourcePath = '';
    info.directMethodPath = methodPath;
    info.methodSourcePaths = struct();
    info.legacyPackPath = '';
    info.aggregatePath = '';
    info.directRecordFound = false;

    if exist(methodPath,'file') ~= 2
        info.loadMode = 'direct_missing_fallback_required';
        return;
    end

    loaded = load(methodPath,'methodRecord');
    if ~isfield(loaded,'methodRecord') || ~isstruct(loaded.methodRecord) || ...
            ~isfield(loaded.methodRecord,'result') || isempty(loaded.methodRecord.result)
        warning('Ignoring malformed independent method record: %s',methodPath);
        info.loadMode = 'direct_malformed_fallback_required';
        return;
    end

    record = loaded.methodRecord;
    validate_direct_record_local(record,roundIndex,nTrain,methodField,methodPath);
    result = restore_direct_artifact_paths_local(methodField,record.result,record);
    info.directSourcePath = methodPath;
    info.methodSourcePaths.(methodField) = methodPath;
    info.directRecordFound = true;

    % Current PhDN files normally already contain their Stage0-SR ablation.
    % For older independent records that do not, attach only the matching
    % Stage0-SR file (one extra small targeted load), never the whole method pack.
    if startsWith(methodField,'phdn')
        result = attach_matching_stage0_direct_local( ...
            result,sourceRoot,roundIndex,nTrain,methodField);
    end
end

function result = attach_matching_stage0_direct_local( ...
    result,sourceRoot,roundIndex,nTrain,methodField)
    hasStage0 = isstruct(result) && isfield(result,'ablations') && ...
        isstruct(result.ablations) && isfield(result.ablations,'stage0SR') && ...
        ~isempty(result.ablations.stage0SR);
    if hasStage0
        return;
    end

    stage0Field = '';
    switch lower(methodField)
        case 'phdn_g1'; stage0Field = 'stage0sr_g1';
        case 'phdn_g2'; stage0Field = 'stage0sr_g2';
        case 'phdn_g3'; stage0Field = 'stage0sr_g3';
        case 'phdn'; stage0Field = 'stage0sr';
    end
    if isempty(stage0Field)
        return;
    end

    roundKey = sprintf('round_%02d',roundIndex);
    sampleKey = sprintf('N_%05d',nTrain);
    stage0Path = fullfile(sourceRoot,roundKey,sampleKey,'method_results', ...
        [stage0Field '.mat']);
    if exist(stage0Path,'file') ~= 2
        return;
    end

    loaded = load(stage0Path,'methodRecord');
    if ~isfield(loaded,'methodRecord') || ~isstruct(loaded.methodRecord) || ...
            ~isfield(loaded.methodRecord,'result') || isempty(loaded.methodRecord.result)
        return;
    end
    validate_direct_record_local(loaded.methodRecord,roundIndex,nTrain, ...
        stage0Field,stage0Path);
    if ~isfield(result,'ablations') || ~isstruct(result.ablations)
        result.ablations = struct();
    end
    result.ablations.stage0SR = loaded.methodRecord.result;
end

function validate_direct_record_local(record,roundIndex,nTrain,fieldName,path)
    if isfield(record,'roundIndex') && record.roundIndex ~= roundIndex
        error('Method record round mismatch in %s.',path);
    end
    if isfield(record,'nTrain') && record.nTrain ~= nTrain
        error('Method record sample-count mismatch in %s.',path);
    end
    if isfield(record,'methodField') && ...
            ~strcmpi(char(record.methodField),fieldName)
        error('Method record field mismatch in %s.',path);
    end
end

function result = restore_direct_artifact_paths_local(fieldName,result,record)
    if ~isfield(record,'artifacts') || ~isstruct(record.artifacts)
        return;
    end
    if strcmp(fieldName,'kan') && ...
            isfield(record.artifacts,'kanNativeCheckpointPath') && ...
            exist(record.artifacts.kanNativeCheckpointPath,'file') == 2
        result.nativeCheckpointPath = record.artifacts.kanNativeCheckpointPath;
    elseif strcmp(fieldName,'eql') && ...
            isfield(record.artifacts,'eqlSelectedStatePath') && ...
            exist(record.artifacts.eqlSelectedStatePath,'file') == 2
        result.selectedStatePath = record.artifacts.eqlSelectedStatePath;
        if isfield(result,'pyResult') && isstruct(result.pyResult)
            result.pyResult.selected_state_path = record.artifacts.eqlSelectedStatePath;
        end
    end
end

function value = getfield_default_local(s,fieldName,defaultValue)
    value = defaultValue;
    if isstruct(s) && isfield(s,fieldName) && ~isempty(s.(fieldName))
        value = s.(fieldName);
    end
end

function compactText = compact_recorded_report_local(reportText,methodField)
%COMPACT_RECORDED_REPORT_LOCAL Minimal replay report for every method family.
% DISPLAY ONLY: archived recordedConsoleReport/result structures are untouched.
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
    prefixes=[commonPrefixes,extraPrefixes];
    for i=1:numel(lines)
        s=strtrim(lines{i});
        if isempty(s); continue; end
        for j=1:numel(prefixes)
            if startsWith(s,prefixes{j},'IgnoreCase',true)
                keep(i)=true; break;
            end
        end
    end
    selected=lines(keep);
    header={'[minimal replay report: historical sweep/candidate/term details omitted]', ...
            '[full archived report remains unchanged in recordedConsoleReport]'};
    if isempty(selected)
        compactText=strjoin(header,sprintf('\n'));
    else
        compactText=strjoin([header,selected],sprintf('\n'));
    end
    compactText=[compactText sprintf('\n')];
end

function [compatible,message] = validate_sindy_guess_prior_local(result,currentPhdnOpts,task,sourcePath)
    compatible = true;
    message = '';
    if isempty(currentPhdnOpts) || ~isstruct(currentPhdnOpts) || isempty(task) || ...
            ~isstruct(task) || ~isfield(task,'nx')
        return;
    end

    [~,expected] = sync_sindy_stage0_initial_guesses( ...
        sindy_default_options(),currentPhdnOpts,task.nx);
    if ~expected.enabled
        return;
    end

    actual = [];
    if isfield(result,'stage0InitialGuessSync') && isstruct(result.stage0InitialGuessSync)
        actual = result.stage0InitialGuessSync;
    elseif isfield(result,'opts') && isstruct(result.opts) && ...
            isfield(result.opts,'stage0InitialGuessSyncInfo') && ...
            isstruct(result.opts.stage0InitialGuessSyncInfo)
        actual = result.opts.stage0InitialGuessSyncInfo;
    end

    if isempty(actual) || ~isfield(actual,'normalizedTerms')
        compatible = false;
        message = sprintf(['Saved SINDy result in %s predates the Stage0SRInitialGuesses ', ...
            'dictionary-union fairness update. Rerun SINDy once with RunSINDyBaseline=true ', ...
            'before replaying it against the seeded Stage-0 SR result.'],sourcePath);
        return;
    end

    expectedTerms = normalize_terms_for_compare_local(expected.normalizedTerms);
    actualTerms = normalize_terms_for_compare_local(actual.normalizedTerms);
    if ~isequal(expectedTerms,actualTerms)
        compatible = false;
        message = sprintf(['Saved SINDy prior library in %s does not match the current ', ...
            'Stage0SRInitialGuesses. Rerun SINDy once so both methods receive the ', ...
            'same candidate expressions.'],sourcePath);
    end
end

function terms = normalize_terms_for_compare_local(value)
    if isempty(value)
        terms = {};
    elseif ischar(value)
        terms = {strrep(strtrim(value),' ','')};
    elseif isstring(value)
        terms = cellstr(value(:));
    else
        terms = value(:);
    end
    for k = 1:numel(terms)
        terms{k} = lower(strrep(strtrim(char(terms{k})),' ',''));
    end
    terms = sort(unique(terms));
end

function validate_sample_count_local(result,nTrain,methodLabel,sourcePath,strictMode)
    recordedN = NaN;
    if isfield(result,'data') && isstruct(result.data) && ...
            isfield(result.data,'nTrain') && isnumeric(result.data.nTrain) && ...
            isscalar(result.data.nTrain)
        recordedN = double(result.data.nTrain);
    elseif isfield(result,'modelTrainingSampleCount') && ...
            isnumeric(result.modelTrainingSampleCount) && ...
            isscalar(result.modelTrainingSampleCount)
        recordedN = double(result.modelTrainingSampleCount);
    end
    if isfinite(recordedN) && recordedN ~= nTrain
        message = sprintf(['Saved %s result sample-count mismatch: requested N=%d, ', ...
            'recorded N=%d in %s.'],methodLabel,nTrain,round(recordedN),sourcePath);
        if strictMode; error('%s',message); else; warning('%s',message); end
    end
end
