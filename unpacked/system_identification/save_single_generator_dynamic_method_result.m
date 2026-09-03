function [methodResult, outputInfo] = save_single_generator_dynamic_method_result( ...
    sampleOutputDir,methodField,methodLabel,methodResult,experimentContext, ...
    savePhase,rollout,systemIdentificationRow,standardSummaryRow)
%SAVE_SINGLE_GENERATOR_DYNAMIC_METHOD_RESULT Persist one method independently.
%
% Authoritative layout for one round/sample:
%   method_results/<method>.mat
%       variable methodRecord containing the complete trained result, task,
%       sampling plan, PhDN/baseline options, seeds, rollout, and summary rows.
%   method_artifacts/<method>/...
%       native checkpoints required by EQL or KAN replay.
%   method_reports/recorded_report_<method>.txt
%       exact console report, when available.
%
% The function also updates the legacy result_pack.mat by FIELD MERGE rather
% than replacement. Therefore saving PhDN can never remove an older MLP, EQL,
% KAN, or SINDy field, and vice versa.

    if nargin < 6 || isempty(savePhase); savePhase = 'post_training'; end
    if nargin < 7; rollout = []; end
    if nargin < 8; systemIdentificationRow = []; end
    if nargin < 9; standardSummaryRow = []; end

    methodField = lower(strtrim(char(methodField)));
    methodLabel = char(methodLabel);
    savePhase = lower(strtrim(char(savePhase)));
    assert(isstruct(methodResult) && ~isempty(methodResult), ...
        'Cannot persist an empty %s result.',methodLabel);

    ensure_dir_local(sampleOutputDir);
    methodResultDir = fullfile(sampleOutputDir,'method_results');
    artifactDir = fullfile(sampleOutputDir,'method_artifacts',methodField);
    reportDir = fullfile(sampleOutputDir,'method_reports');
    ensure_dir_local(methodResultDir);
    ensure_dir_local(artifactDir);
    ensure_dir_local(reportDir);

    methodPath = fullfile(methodResultDir,[methodField '.mat']);
    reportPath = fullfile(reportDir,sprintf('recorded_report_%s.txt',methodField));
    legacyPackPath = fullfile(sampleOutputDir,'result_pack.mat');

    existingRecord = struct();
    if exist(methodPath,'file') == 2
        loaded = load(methodPath,'methodRecord');
        if isfield(loaded,'methodRecord') && isstruct(loaded.methodRecord)
            existingRecord = loaded.methodRecord;
        end
    end

    [methodResult,artifactInfo] = persist_native_artifacts_local( ...
        methodField,methodResult,artifactDir);
    if ~strcmp(savePhase,'post_training') && ...
            isfield(existingRecord,'artifacts') && isstruct(existingRecord.artifacts)
        artifactInfo = merge_artifact_info_local(existingRecord.artifacts,artifactInfo);
    end

    nowText = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    methodRecord = struct();
    methodRecord.schemaVersion = 'single_generator_dynamic_method_record_v1';
    methodRecord.methodField = methodField;
    methodRecord.methodLabel = methodLabel;
    methodRecord.roundIndex = experimentContext.roundIndex;
    methodRecord.nTrain = experimentContext.samplingPlan.nTrain;
    methodRecord.caseName = text_field_local(experimentContext.task,'name','');
    methodRecord.modelVariant = ...
        text_field_local(experimentContext.task,'modelVariant','');
    methodRecord.savedAt = nowText;
    methodRecord.lastSavePhase = savePhase;
    methodRecord.sourceType = infer_source_type_local(methodResult);
    methodRecord.result = methodResult;
    methodRecord.task = experimentContext.task;
    methodRecord.samplingPlan = experimentContext.samplingPlan;
    methodRecord.experimentContext = experimentContext;
    methodRecord.artifacts = artifactInfo;

    if strcmp(savePhase,'post_training') || ~isfield(existingRecord,'trainingCompletedAt')
        methodRecord.trainingCompletedAt = nowText;
    else
        methodRecord.trainingCompletedAt = existingRecord.trainingCompletedAt;
    end

    if ~isempty(rollout)
        methodRecord.rollout = rollout;
        methodRecord.rolloutCompletedAt = nowText;
    elseif ~strcmp(savePhase,'post_training') && isfield(existingRecord,'rollout')
        methodRecord.rollout = existingRecord.rollout;
        if isfield(existingRecord,'rolloutCompletedAt')
            methodRecord.rolloutCompletedAt = existingRecord.rolloutCompletedAt;
        end
    else
        methodRecord.rollout = [];
    end

    if ~isempty(systemIdentificationRow)
        methodRecord.systemIdentificationRow = systemIdentificationRow;
    elseif ~strcmp(savePhase,'post_training') && ...
            isfield(existingRecord,'systemIdentificationRow')
        methodRecord.systemIdentificationRow = existingRecord.systemIdentificationRow;
    else
        methodRecord.systemIdentificationRow = struct([]);
    end

    if ~isempty(standardSummaryRow)
        methodRecord.standardSummaryRow = standardSummaryRow;
    elseif ~strcmp(savePhase,'post_training') && ...
            isfield(existingRecord,'standardSummaryRow')
        methodRecord.standardSummaryRow = existingRecord.standardSummaryRow;
    else
        methodRecord.standardSummaryRow = struct([]);
    end

    atomic_save_local(methodPath,'methodRecord',methodRecord);
    write_report_local(reportPath,methodResult);
    merge_legacy_result_pack_local(legacyPackPath,methodField,methodResult);
    methodIndexPath = update_method_index_local( ...
        sampleOutputDir,methodPath,methodRecord,artifactInfo);
    methodArchivePath = update_method_archive_local( ...
        sampleOutputDir,methodPath,methodRecord,artifactInfo);

    outputInfo = struct();
    outputInfo.methodPath = methodPath;
    outputInfo.reportPath = reportPath;
    outputInfo.artifactDir = artifactDir;
    outputInfo.methodIndexPath = methodIndexPath;
    outputInfo.methodArchivePath = methodArchivePath;
    outputInfo.legacyPackPath = legacyPackPath;
    outputInfo.savePhase = savePhase;

    fprintf('Persisted %s independently [%s]:\n%s\n', ...
        methodLabel,savePhase,methodPath);
end


function merged = merge_artifact_info_local(existing,current)
    merged = existing;
    names = fieldnames(current);
    for k = 1:numel(names)
        name = names{k};
        value = current.(name);
        if ischar(value) || (isstring(value) && isscalar(value))
            if ~isempty(value); merged.(name) = value; end
        elseif iscell(value)
            if ~isempty(value); merged.(name) = value; end
        else
            merged.(name) = value;
        end
    end
end

function [result,artifactInfo] = persist_native_artifacts_local(methodField,result,artifactDir)
    artifactInfo = struct();
    artifactInfo.directory = artifactDir;
    artifactInfo.kanNativeCheckpointPath = '';
    artifactInfo.eqlSelectedStatePath = '';
    artifactInfo.copyWarnings = {};

    if strcmp(methodField,'kan') && isfield(result,'nativeCheckpointPath') && ...
            ~isempty(result.nativeCheckpointPath) && ...
            exist(result.nativeCheckpointPath,'file') == 2
        destination = fullfile(artifactDir,'selected_native_checkpoint.pt');
        [ok,message] = copy_if_needed_local(result.nativeCheckpointPath,destination);
        if ok
            result.nativeCheckpointPath = destination;
            artifactInfo.kanNativeCheckpointPath = destination;
        else
            artifactInfo.copyWarnings{end+1} = message;
        end
    end

    if strcmp(methodField,'eql') && isfield(result,'selectedStatePath') && ...
            ~isempty(result.selectedStatePath) && ...
            exist(result.selectedStatePath,'file') == 2
        destination = fullfile(artifactDir,'selected_state.pkl');
        [ok,message] = copy_if_needed_local(result.selectedStatePath,destination);
        if ok
            result.selectedStatePath = destination;
            artifactInfo.eqlSelectedStatePath = destination;
            if isfield(result,'pyResult') && isstruct(result.pyResult)
                result.pyResult.selected_state_path = destination;
            end
        else
            artifactInfo.copyWarnings{end+1} = message;
        end
    end
end

function [ok,message] = copy_if_needed_local(source,destination)
    ok = true;
    message = '';
    try
        if ~strcmpi(char(source),char(destination))
            copyfile(source,destination,'f');
        end
    catch ME
        ok = false;
        message = sprintf('Could not persist native artifact %s -> %s: %s', ...
            source,destination,ME.message);
        warning('%s',message);
    end
end

function merge_legacy_result_pack_local(packPath,methodField,methodResult)
    resultPack = struct();
    if exist(packPath,'file') == 2
        loaded = load(packPath,'resultPack');
        if isfield(loaded,'resultPack') && isstruct(loaded.resultPack)
            resultPack = loaded.resultPack;
        end
    end
    resultPack.(methodField) = methodResult;
    atomic_save_local(packPath,'resultPack',resultPack);
end

function indexPath = update_method_index_local(sampleOutputDir,methodPath,record,artifactInfo)
    caseRoot = fileparts(fileparts(sampleOutputDir));
    indexDir = fullfile(caseRoot,'summary','method_archives');
    ensure_dir_local(indexDir);
    indexPath = fullfile(indexDir,[record.methodField '_index.mat']);

    methodIndex = struct();
    methodIndex.schemaVersion = 'single_generator_dynamic_method_index_v1';
    methodIndex.methodField = record.methodField;
    methodIndex.methodLabel = record.methodLabel;
    methodIndex.updatedAt = record.savedAt;
    methodIndex.entries = struct([]);
    if exist(indexPath,'file') == 2
        loaded = load(indexPath,'methodIndex');
        if isfield(loaded,'methodIndex') && isstruct(loaded.methodIndex) && ...
                isfield(loaded.methodIndex,'entries')
            methodIndex = loaded.methodIndex;
            methodIndex.updatedAt = record.savedAt;
        end
    end

    entry = make_index_entry_local(methodPath,record,artifactInfo);
    entries = methodIndex.entries;
    replaceIndex = [];
    for k = 1:numel(entries)
        if entries(k).roundIndex == entry.roundIndex && ...
                entries(k).nTrain == entry.nTrain
            replaceIndex = k;
            break;
        end
    end
    if isempty(replaceIndex)
        if isempty(entries); entries = entry; else; entries(end+1) = entry; end
    else
        entries(replaceIndex) = entry;
    end
    methodIndex.entries = entries;
    atomic_save_local(indexPath,'methodIndex',methodIndex);
end


function archivePath = update_method_archive_local(sampleOutputDir,methodPath,record,artifactInfo)
%UPDATE_METHOD_ARCHIVE_LOCAL Maintain one complete case-level archive per method.
%
% The per-round/per-N method file remains authoritative. The archive is an
% additional standalone collection containing every completed experiment for
% this method across all rounds and sample sizes. Full methodRecord objects are
% stored in a cell array because post-training and post-rollout records may
% contain different optional fields.

    caseRoot = fileparts(fileparts(sampleOutputDir));
    archiveDir = fullfile(caseRoot,'summary','method_archives');
    ensure_dir_local(archiveDir);
    archivePath = fullfile(archiveDir,[record.methodField '_archive.mat']);

    methodArchive = struct();
    methodArchive.schemaVersion = 'single_generator_dynamic_method_archive_v1';
    methodArchive.methodField = record.methodField;
    methodArchive.methodLabel = record.methodLabel;
    methodArchive.caseRoot = caseRoot;
    methodArchive.updatedAt = record.savedAt;
    methodArchive.records = {};
    methodArchive.entries = struct([]);

    if exist(archivePath,'file') == 2
        loaded = load(archivePath,'methodArchive');
        if isfield(loaded,'methodArchive') && isstruct(loaded.methodArchive)
            methodArchive = loaded.methodArchive;
            if ~isfield(methodArchive,'records') || ~iscell(methodArchive.records)
                methodArchive.records = {};
            end
            if ~isfield(methodArchive,'entries') || ~isstruct(methodArchive.entries)
                methodArchive.entries = struct([]);
            end
            methodArchive.updatedAt = record.savedAt;
            methodArchive.caseRoot = caseRoot;
        end
    end

    replaceIndex = [];
    for k = 1:numel(methodArchive.records)
        oldRecord = methodArchive.records{k};
        if isstruct(oldRecord) && isfield(oldRecord,'roundIndex') && ...
                isfield(oldRecord,'nTrain') && ...
                oldRecord.roundIndex == record.roundIndex && ...
                oldRecord.nTrain == record.nTrain
            replaceIndex = k;
            break;
        end
    end

    if isempty(replaceIndex)
        methodArchive.records{end+1} = record;
    else
        methodArchive.records{replaceIndex} = record;
    end

    entry = make_index_entry_local(methodPath,record,artifactInfo);
    entryIndex = [];
    for k = 1:numel(methodArchive.entries)
        if methodArchive.entries(k).roundIndex == entry.roundIndex && ...
                methodArchive.entries(k).nTrain == entry.nTrain
            entryIndex = k;
            break;
        end
    end
    if isempty(entryIndex)
        if isempty(methodArchive.entries)
            methodArchive.entries = entry;
        else
            methodArchive.entries(end+1) = entry;
        end
    else
        methodArchive.entries(entryIndex) = entry;
    end

    methodArchive.completedExperimentCount = numel(methodArchive.records);
    atomic_save_local(archivePath,'methodArchive',methodArchive);
end

function entry = make_index_entry_local(methodPath,record,artifactInfo)
    entry = struct();
    entry.roundIndex = record.roundIndex;
    entry.nTrain = record.nTrain;
    entry.caseName = record.caseName;
    entry.modelVariant = record.modelVariant;
    entry.methodRecordPath = methodPath;
    entry.savedAt = record.savedAt;
    entry.lastSavePhase = record.lastSavePhase;
    entry.sourceType = record.sourceType;
    entry.validationMSE = nested_numeric_local(record.result,{'valMetrics','mse'},NaN);
    if startsWith(record.methodField,'phdn')
        entry.validationMSE = first_finite_local([ ...
            nested_numeric_local(record.result,{'bestValidationMSE'},NaN), ...
            entry.validationMSE]);
    end
    entry.idTestRMSE = nested_numeric_local(record.result,{'testMetrics','rmse'},NaN);
    entry.oodRMSE = nested_numeric_local(record.result,{'oodMetrics','rmse'},NaN);
    entry.activeCoefficients = first_finite_local([ ...
        nested_numeric_local(record.result,{'nActiveFinal'},NaN), ...
        nested_numeric_local(record.result,{'nActiveCoefficients'},NaN), ...
        nested_numeric_local(record.result,{'parameterCount'},NaN)]);
    entry.hasRollout = isstruct(record.rollout) && ~isempty(record.rollout);
    entry.rolloutRMSE = NaN;
    entry.rolloutNRMSE = NaN;
    if entry.hasRollout
        entry.rolloutRMSE = nested_numeric_local(record.rollout,{'rawRMSE'},NaN);
        entry.rolloutNRMSE = nested_numeric_local(record.rollout,{'normalizedRMSE'},NaN);
    end
    entry.kanNativeCheckpointPath = artifactInfo.kanNativeCheckpointPath;
    entry.eqlSelectedStatePath = artifactInfo.eqlSelectedStatePath;
end

function sourceType = infer_source_type_local(result)
    sourceType = 'trained';
    if isfield(result,'reusedRecordedMethod') && logical_scalar_local(result.reusedRecordedMethod)
        sourceType = 'replayed';
    elseif isfield(result,'reusedRecordedBaseline') && logical_scalar_local(result.reusedRecordedBaseline)
        sourceType = 'replayed';
    end
end

function tf = logical_scalar_local(value)
    tf = false;
    if islogical(value) && ~isempty(value)
        tf = value(1);
    elseif isnumeric(value) && ~isempty(value) && isfinite(value(1))
        tf = logical(value(1));
    end
end

function value = nested_numeric_local(s,path,defaultValue)
    value = defaultValue;
    current = s;
    for k = 1:numel(path)
        if ~isstruct(current) || ~isfield(current,path{k}) || isempty(current.(path{k}))
            return;
        end
        current = current.(path{k});
    end
    if (isnumeric(current) || islogical(current)) && isscalar(current)
        value = double(current);
    end
end

function value = first_finite_local(values)
    value = NaN;
    idx = find(isfinite(values),1,'first');
    if ~isempty(idx); value = values(idx); end
end

function value = text_field_local(s,name,defaultValue)
    value = defaultValue;
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = char(string(s.(name)));
    end
end

function write_report_local(path,result)
    if ~isfield(result,'recordedConsoleReport') || isempty(result.recordedConsoleReport)
        return;
    end
    fid = fopen(path,'w');
    if fid < 0
        warning('Could not write recorded method report: %s',path);
        return;
    end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'%s',char(result.recordedConsoleReport));
end

function atomic_save_local(path,variableName,value)
    parent = fileparts(path);
    ensure_dir_local(parent);
    temporaryPath = [tempname(parent) '.mat'];
    payload = struct();
    payload.(variableName) = value;
    save(temporaryPath,'-struct','payload','-v7.3');
    [ok,message] = movefile(temporaryPath,path,'f');
    if ~ok
        if exist(temporaryPath,'file') == 2; delete(temporaryPath); end
        error('Atomic save failed for %s: %s',path,message);
    end
end

function ensure_dir_local(path)
    if exist(path,'dir') ~= 7; mkdir(path); end
end
