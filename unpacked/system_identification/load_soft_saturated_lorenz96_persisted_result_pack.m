function [resultPack, loadInfo] = load_soft_saturated_lorenz96_persisted_result_pack( ...
    sourceRoot,roundIndex,nTrain,strictMode)
%LOAD_SOFT_SATURATED_LORENZ96_PERSISTED_RESULT_PACK Merge saved method records.
%
% Search order for each method:
%   1) independent method_results/<method>.mat (authoritative);
%   2) case-level summary/method_archives/<method>_archive.mat;
%   3) legacy per-sample result_pack.mat;
%   4) aggregate summary/soft_saturated_lorenz96_results.mat.
%
% A legacy result_pack.mat that exists but lacks one method no longer blocks
% fallback to the aggregate archive or another independent method file.

    if nargin < 4 || isempty(strictMode); strictMode = true; end
    roundKey = sprintf('round_%02d',roundIndex);
    sampleKey = sprintf('N_%05d',nTrain);
    sampleDir = fullfile(sourceRoot,roundKey,sampleKey);
    methodDir = fullfile(sampleDir,'method_results');
    legacyPackPath = fullfile(sampleDir,'result_pack.mat');
    aggregatePath = fullfile(sourceRoot,'summary','soft_saturated_lorenz96_results.mat');

    resultPack = struct();
    loadInfo = struct();
    loadInfo.sourceRoot = sourceRoot;
    loadInfo.roundKey = roundKey;
    loadInfo.sampleKey = sampleKey;
    loadInfo.sampleDir = sampleDir;
    loadInfo.methodSourcePaths = struct();
    loadInfo.methodRecords = struct();
    loadInfo.methodArchivePaths = struct();
    loadInfo.legacyPackPath = '';
    loadInfo.aggregatePath = '';

    % Current Lorenz--96 paper workflow persists G1/G2/G3 independently.
    % Keep legacy phdn/stage0sr fields at the end only as compatibility fallbacks.
    methodFields = {'phdn_g1','stage0sr_g1','phdn_g2','stage0sr_g2', ...
        'phdn_g3','stage0sr_g3','mlp','eql','kan','sindy','neural_sindy', ...
        'phdn','stage0sr'};
    for k = 1:numel(methodFields)
        fieldName = methodFields{k};
        methodPath = fullfile(methodDir,[fieldName '.mat']);
        record = struct();
        sourcePath = '';

        if exist(methodPath,'file') == 2
            loaded = load(methodPath,'methodRecord');
            if isfield(loaded,'methodRecord') && isstruct(loaded.methodRecord) && ...
                    isfield(loaded.methodRecord,'result') && ...
                    ~isempty(loaded.methodRecord.result)
                record = loaded.methodRecord;
                sourcePath = methodPath;
            else
                warning('Ignoring malformed independent method record: %s',methodPath);
            end
        end

        if isempty(fieldnames(record))
            archivePath = fullfile(sourceRoot,'summary','method_archives', ...
                [fieldName '_archive.mat']);
            [record,foundInArchive] = find_record_in_archive_local( ...
                archivePath,roundIndex,nTrain,fieldName);
            if foundInArchive
                sourcePath = archivePath;
                loadInfo.methodArchivePaths.(fieldName) = archivePath;
            end
        end

        if isempty(fieldnames(record))
            continue;
        end
        validate_record_local(record,roundIndex,nTrain,fieldName,sourcePath);
        resultPack.(fieldName) = record.result;
        resultPack.(fieldName) = restore_artifact_paths_local( ...
            fieldName,resultPack.(fieldName),record);
        loadInfo.methodSourcePaths.(fieldName) = sourcePath;
        loadInfo.methodRecords.(fieldName) = record;
        if isfield(record,'task') && ~isfield(resultPack,'task')
            resultPack.task = record.task;
        end
        if isfield(record,'samplingPlan') && ~isfield(resultPack,'samplingPlan')
            resultPack.samplingPlan = record.samplingPlan;
        end
    end

    if exist(legacyPackPath,'file') == 2
        loaded = load(legacyPackPath,'resultPack');
        if isfield(loaded,'resultPack') && isstruct(loaded.resultPack)
            resultPack = merge_missing_fields_local(resultPack,loaded.resultPack);
            loadInfo.legacyPackPath = legacyPackPath;
        end
    end

    if exist(aggregatePath,'file') == 2
        loaded = load(aggregatePath,'allResults');
        if isfield(loaded,'allResults') && isstruct(loaded.allResults) && ...
                isfield(loaded.allResults,roundKey) && ...
                isfield(loaded.allResults.(roundKey),sampleKey)
            aggregatePack = loaded.allResults.(roundKey).(sampleKey);
            if isstruct(aggregatePack)
                resultPack = merge_missing_fields_local(resultPack,aggregatePack);
                loadInfo.aggregatePath = aggregatePath;
            end
        end
    end

    phdnStage0Pairs = { ...
        'phdn_g1','stage0sr_g1'; ...
        'phdn_g2','stage0sr_g2'; ...
        'phdn_g3','stage0sr_g3'; ...
        'phdn','stage0sr'};
    for iPair = 1:size(phdnStage0Pairs,1)
        phdnField = phdnStage0Pairs{iPair,1};
        stage0Field = phdnStage0Pairs{iPair,2};
        if ~isfield(resultPack,phdnField) || ~isfield(resultPack,stage0Field)
            continue;
        end
        if ~isfield(resultPack.(phdnField),'ablations') || ...
                ~isstruct(resultPack.(phdnField).ablations)
            resultPack.(phdnField).ablations = struct();
        end
        if ~isfield(resultPack.(phdnField).ablations,'stage0SR') || ...
                isempty(resultPack.(phdnField).ablations.stage0SR)
            resultPack.(phdnField).ablations.stage0SR = ...
                resultPack.(stage0Field);
        end
    end

    loadInfo.availableMethods = intersect(methodFields,fieldnames(resultPack),'stable');
    if isempty(loadInfo.availableMethods) && strictMode
        error(['No independently persisted or legacy method result was found for ', ...
            '%s/%s under:\n%s'],roundKey,sampleKey,sourceRoot);
    end
end


function [record,found] = find_record_in_archive_local( ...
    archivePath,roundIndex,nTrain,fieldName)
    record = struct();
    found = false;
    if exist(archivePath,'file') ~= 2
        return;
    end
    loaded = load(archivePath,'methodArchive');
    if ~isfield(loaded,'methodArchive') || ~isstruct(loaded.methodArchive) || ...
            ~isfield(loaded.methodArchive,'records') || ...
            ~iscell(loaded.methodArchive.records)
        warning('Ignoring malformed method archive: %s',archivePath);
        return;
    end
    for k = 1:numel(loaded.methodArchive.records)
        candidate = loaded.methodArchive.records{k};
        if ~isstruct(candidate) || ~isfield(candidate,'roundIndex') || ...
                ~isfield(candidate,'nTrain') || ~isfield(candidate,'methodField') || ...
                ~isfield(candidate,'result') || isempty(candidate.result)
            continue;
        end
        if candidate.roundIndex == roundIndex && candidate.nTrain == nTrain && ...
                strcmpi(char(candidate.methodField),fieldName)
            record = candidate;
            found = true;
            return;
        end
    end
end

function result = restore_artifact_paths_local(fieldName,result,record)
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

function merged = merge_missing_fields_local(primary,fallback)
    merged = primary;
    names = fieldnames(fallback);
    for k = 1:numel(names)
        name = names{k};
        if ~isfield(merged,name)
            merged.(name) = fallback.(name);
        end
    end
end

function validate_record_local(record,roundIndex,nTrain,fieldName,path)
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
