%% Audit saved system-identification scalar rows (NO TRAINING / NO ROLLOUT)
% Checks the two manuscript SI cases for inconsistent or duplicated saved
% scalar rows across public_summary.mat, per-sample row files, and independent
% method_results/*.mat post-rollout records.
%
% This script is read-only. It does not modify any saved result.

clear; clc;

projectRoot = pwd;
if ~exist(fullfile(projectRoot,'core'),'dir') || ...
        ~exist(fullfile(projectRoot,'system_identification'),'dir')
    try
        activeFile = matlab.desktop.editor.getActiveFilename;
        projectRoot = fileparts(activeFile);
    catch
        error('Cannot locate project root. Set the current folder to the framework root.');
    end
end

caseNames = { ...
    'SingleGeneratorDynamic_SMIB_AVR', ...
    'SoftSaturatedLorenz96_K10_F8_kappa1'};
metricNames = {'derivativeRMSE','trajectoryNRMSE'};
relativeTolerance = 1e-10;
absoluteTolerance = 1e-14;

fprintf('\n============================================================\n');
fprintf('System-identification saved-row audit (read only)\n');
fprintf('============================================================\n');

for iCase = 1:numel(caseNames)
    caseName = caseNames{iCase};
    caseRoot = fullfile(projectRoot,'outputs',caseName);
    fprintf('\nCASE: %s\n',caseName);
    fprintf('Root: %s\n',caseRoot);
    if exist(caseRoot,'dir') ~= 7
        fprintf('  [SKIP] output root does not exist.\n');
        continue;
    end

    summaryRows = struct([]);
    summaryPath = fullfile(caseRoot,'summary','public_summary.mat');
    if exist(summaryPath,'file') == 2
        loadedSummary = load(summaryPath,'systemIdentificationRows');
        if isfield(loadedSummary,'systemIdentificationRows') && ...
                isstruct(loadedSummary.systemIdentificationRows)
            summaryRows = loadedSummary.systemIdentificationRows;
        end
    end

    sampleRowsAll = struct([]);
    methodRowsAll = struct([]);
    nMethodFiles = 0;
    nMissingStage0Records = 0;

    roundDirs = dir(fullfile(caseRoot,'round_*'));
    roundDirs = roundDirs([roundDirs.isdir]);
    for iRoundDir = 1:numel(roundDirs)
        roundPath = fullfile(roundDirs(iRoundDir).folder,roundDirs(iRoundDir).name);
        sampleDirs = dir(fullfile(roundPath,'N_*'));
        sampleDirs = sampleDirs([sampleDirs.isdir]);
        for iSampleDir = 1:numel(sampleDirs)
            samplePath = fullfile(sampleDirs(iSampleDir).folder,sampleDirs(iSampleDir).name);
            rowsPath = fullfile(samplePath,'system_identification_rows.mat');
            if exist(rowsPath,'file') == 2
                loadedRows = load(rowsPath,'sampleRows');
                if isfield(loadedRows,'sampleRows') && isstruct(loadedRows.sampleRows)
                    sampleRowsAll = append_rows_local(sampleRowsAll,loadedRows.sampleRows);
                end
            end

            methodDir = fullfile(samplePath,'method_results');
            methodFiles = dir(fullfile(methodDir,'*.mat'));
            nMethodFiles = nMethodFiles + numel(methodFiles);
            for iFile = 1:numel(methodFiles)
                path = fullfile(methodFiles(iFile).folder,methodFiles(iFile).name);
                try
                    loadedRecord = load(path,'methodRecord');
                catch
                    continue;
                end
                if isfield(loadedRecord,'methodRecord') && ...
                        isstruct(loadedRecord.methodRecord) && ...
                        isfield(loadedRecord.methodRecord,'systemIdentificationRow') && ...
                        isstruct(loadedRecord.methodRecord.systemIdentificationRow) && ...
                        numel(loadedRecord.methodRecord.systemIdentificationRow)==1
                    methodRowsAll = append_rows_local(methodRowsAll, ...
                        loadedRecord.methodRecord.systemIdentificationRow);
                end
            end

            % Current v75 should persist one independent Stage0-SR record for
            % every G level. Missing records are allowed in historical archives
            % but are reported because sample rows then become the fallback.
            for g = 1:3
                stage0Path = fullfile(methodDir,sprintf('stage0sr_g%d.mat',g));
                if exist(stage0Path,'file') ~= 2
                    nMissingStage0Records = nMissingStage0Records + 1;
                end
            end
        end
    end

    [summaryRows,nDupSummary] = dedupe_rows_local(summaryRows);
    [sampleRowsAll,nDupSample] = dedupe_rows_local(sampleRowsAll);
    [methodRowsAll,nDupMethod] = dedupe_rows_local(methodRowsAll);

    fprintf('  unique summary rows : %d (duplicates=%d)\n',numel(summaryRows),nDupSummary);
    fprintf('  unique sample rows  : %d (duplicates=%d)\n',numel(sampleRowsAll),nDupSample);
    fprintf('  method rows/files   : %d / %d (duplicates=%d)\n', ...
        numel(methodRowsAll),nMethodFiles,nDupMethod);
    fprintf('  missing Stage0-SR independent records: %d\n',nMissingStage0Records);

    nMismatchSampleMethod = compare_sources_local( ...
        sampleRowsAll,methodRowsAll,'sample','method',metricNames, ...
        relativeTolerance,absoluteTolerance);
    nMismatchSummaryMethod = compare_sources_local( ...
        summaryRows,methodRowsAll,'summary','method',metricNames, ...
        relativeTolerance,absoluteTolerance);

    if nMismatchSampleMethod==0 && nMismatchSummaryMethod==0 && ...
            nDupSummary==0 && nDupSample==0 && nDupMethod==0
        fprintf('  RESULT: no inconsistency found among overlapping saved rows.\n');
    else
        fprintf(['  RESULT: review warnings above. Current permanent replot uses ', ...
            'independent method rows as highest-priority source when present.\n']);
    end
end

fprintf('\nAudit finished. No files were modified.\n');

function nMismatch = compare_sources_local(leftRows,rightRows,leftName,rightName, ...
        metricNames,relTol,absTol)
    nMismatch = 0;
    if isempty(leftRows) || isempty(rightRows); return; end
    for iRight = 1:numel(rightRows)
        idx = find_key_local(leftRows,rightRows(iRight));
        if isempty(idx); continue; end
        left = leftRows(idx(1));
        right = rightRows(iRight);
        for iMetric = 1:numel(metricNames)
            metric = metricNames{iMetric};
            a = scalar_metric_local(left,metric);
            b = scalar_metric_local(right,metric);
            if values_differ_local(a,b,relTol,absTol)
                fprintf(['  [MISMATCH] %s | %s: %s=%.12e, %s=%.12e\n'], ...
                    row_key_local(right),metric,leftName,a,rightName,b);
                nMismatch = nMismatch + 1;
            end
        end
    end
end

function tf = values_differ_local(a,b,relTol,absTol)
    if isnan(a) && isnan(b); tf=false; return; end
    if isinf(a) || isinf(b); tf=~isequal(a,b); return; end
    if ~isfinite(a) || ~isfinite(b); tf=~isequaln(a,b); return; end
    tf = abs(a-b) > max(absTol,relTol*max([1,abs(a),abs(b)]));
end

function value = scalar_metric_local(row,name)
    value = NaN;
    if isstruct(row) && isfield(row,name) && isnumeric(row.(name)) && ...
            isscalar(row.(name))
        value = double(row.(name));
    end
end

function idx = find_key_local(rows,target)
    idx = [];
    key = row_key_local(target);
    for k = 1:numel(rows)
        if strcmp(row_key_local(rows(k)),key)
            idx(end+1) = k; %#ok<AGROW>
        end
    end
end

function key = row_key_local(row)
    roundIndex = scalar_field_local(row,'roundIndex',NaN);
    nTrain = scalar_field_local(row,'nTrain',NaN);
    method = '';
    if isfield(row,'method') && ~isempty(row.method)
        method = lower(strtrim(char(string(row.method))));
    end
    key = sprintf('R%.15g|N%.15g|M%s',roundIndex,nTrain,method);
end

function value = scalar_field_local(row,name,defaultValue)
    value = defaultValue;
    if isfield(row,name) && isnumeric(row.(name)) && isscalar(row.(name))
        value = double(row.(name));
    end
end

function rows = append_rows_local(rows,newRows)
    if isempty(newRows); return; end
    if isempty(rows); rows=newRows; return; end
    [rows,newRows] = harmonize_local(rows,newRows);
    rows = [rows(:);newRows(:)].';
end

function [rows,nRemoved] = dedupe_rows_local(rows)
    nRemoved = 0;
    if isempty(rows); return; end
    keep = true(1,numel(rows));
    keys = cell(1,numel(rows));
    for k=1:numel(rows); keys{k}=row_key_local(rows(k)); end
    % Keep the last occurrence because later saves are normally newer.
    seen = containers.Map('KeyType','char','ValueType','logical');
    for k=numel(rows):-1:1
        if isKey(seen,keys{k})
            keep(k)=false; nRemoved=nRemoved+1;
        else
            seen(keys{k})=true;
        end
    end
    rows=rows(keep);
end

function [left,right] = harmonize_local(left,right)
    allNames=unique([fieldnames(left);fieldnames(right)],'stable');
    for i=1:numel(allNames)
        name=allNames{i};
        proto=[];
        if isfield(left,name) && ~isempty(left); proto=left(1).(name); end
        if isempty(proto) && isfield(right,name) && ~isempty(right); proto=right(1).(name); end
        default=missing_local(proto);
        if ~isfield(left,name)
            for k=1:numel(left); left(k).(name)=default; end
        end
        if ~isfield(right,name)
            for k=1:numel(right); right(k).(name)=default; end
        end
    end
    left=orderfields(left,allNames); right=orderfields(right,allNames);
end

function value = missing_local(proto)
    if isnumeric(proto); value=nan(size(proto));
    elseif islogical(proto); value=false(size(proto));
    elseif ischar(proto); value='';
    elseif isstring(proto); value=strings(size(proto));
    elseif isstruct(proto); value=struct();
    elseif iscell(proto); value=cell(size(proto));
    else; value=[];
    end
end
