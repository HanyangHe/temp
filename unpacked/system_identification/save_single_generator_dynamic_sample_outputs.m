function outputInfo = save_single_generator_dynamic_sample_outputs( ...
    outputDir,resultPack,task,samplingPlan,roundIndex,sampleRows, ...
    trajectoryFigure,trajectoryFigureData)
%SAVE_SINGLE_GENERATOR_DYNAMIC_SAMPLE_OUTPUTS Persist one N/round result.
%
% Independent files under method_results/ are authoritative. This function
% writes a merged compatibility result_pack.mat and never removes a method
% field that was saved by an earlier run. Scalar rows are merged by
% (roundIndex,nTrain,method) for the same reason.

    if nargin < 8 || isempty(trajectoryFigureData)
        trajectoryFigureData = struct();
    end
    if exist(outputDir,'dir') ~= 7; mkdir(outputDir); end

    nTrain = samplingPlan.nTrain;
    baseName = sprintf('trajectory_N_%05d_R%02d',nTrain,roundIndex);

    outputInfo = struct();
    outputInfo.outputDir = outputDir;
    outputInfo.resultPackPath = fullfile(outputDir,'result_pack.mat');
    outputInfo.contextPath = fullfile(outputDir,'sample_context.mat');
    outputInfo.rowsPath = fullfile(outputDir,'system_identification_rows.mat');
    outputInfo.metricsCsvPath = fullfile(outputDir,'metrics.csv');
    outputInfo.trajectoryDataPath = fullfile(outputDir,'trajectory_figure_data.mat');
    outputInfo.trajectoryPdfPath = fullfile(outputDir,[baseName '.pdf']);
    outputInfo.trajectoryFigPath = fullfile(outputDir,[baseName '.fig']);
    outputInfo.methodResultDir = fullfile(outputDir,'method_results');
    outputInfo.methodArtifactDir = fullfile(outputDir,'method_artifacts');
    outputInfo.recordedReportPaths = struct();
    outputInfo.kanNativeCheckpointPath = '';
    outputInfo.eqlSelectedStatePath = '';

    caseRoot = fileparts(fileparts(outputDir));
    [persistedPack,~] = load_single_generator_dynamic_persisted_result_pack( ...
        caseRoot,roundIndex,nTrain,false);
    mergedResultPack = merge_struct_fields_local(persistedPack,resultPack,true);
    mergedResultPack.task = task;
    mergedResultPack.samplingPlan = samplingPlan;

    existingRows = struct([]);
    if exist(outputInfo.rowsPath,'file') == 2
        loadedRows = load(outputInfo.rowsPath,'sampleRows');
        if isfield(loadedRows,'sampleRows') && isstruct(loadedRows.sampleRows)
            existingRows = loadedRows.sampleRows;
        end
    end
    sampleRows = merge_rows_local(existingRows,sampleRows);

    atomic_save_local(outputInfo.resultPackPath,'resultPack',mergedResultPack);
    save(outputInfo.contextPath,'task','samplingPlan','roundIndex','-v7.3');
    atomic_save_local(outputInfo.rowsPath,'sampleRows',sampleRows);
    atomic_save_local(outputInfo.trajectoryDataPath, ...
        'trajectoryFigureData',trajectoryFigureData);
    outputInfo.recordedReportPaths = write_recorded_reports_local(outputDir,mergedResultPack);

    metricsTable = rows_to_flat_table_local(sampleRows);
    writetable(metricsTable,outputInfo.metricsCsvPath);

    kanArtifact = fullfile(outputInfo.methodArtifactDir,'kan','selected_native_checkpoint.pt');
    eqlArtifact = fullfile(outputInfo.methodArtifactDir,'eql','selected_state.pkl');
    if exist(kanArtifact,'file') == 2; outputInfo.kanNativeCheckpointPath = kanArtifact; end
    if exist(eqlArtifact,'file') == 2; outputInfo.eqlSelectedStatePath = eqlArtifact; end

    if isgraphics(trajectoryFigure)
        savefig(trajectoryFigure,outputInfo.trajectoryFigPath);
        export_pdf_local(trajectoryFigure,outputInfo.trajectoryPdfPath);
    else
        outputInfo.trajectoryPdfPath = '';
        outputInfo.trajectoryFigPath = '';
    end

    fprintf('Saved merged N=%d, round=%d outputs to:\n%s\n', ...
        nTrain,roundIndex,outputDir);
end

function merged = merge_struct_fields_local(base,update,overwriteExisting)
    if nargin < 3; overwriteExisting = true; end
    if ~isstruct(base); base = struct(); end
    merged = base;
    if ~isstruct(update); return; end
    names = fieldnames(update);
    for k = 1:numel(names)
        name = names{k};
        if overwriteExisting || ~isfield(merged,name)
            merged.(name) = update.(name);
        end
    end
end

function rows = merge_rows_local(existingRows,newRows)
    rows = existingRows;
    if isempty(newRows); return; end
    if isempty(rows); rows = newRows; return; end

    % Saved row files may have been produced by an earlier v75 schema.
    % MATLAB does not permit assignment between struct arrays with different
    % field sets, so first upgrade both arrays to the union schema. Missing
    % scalar metrics are represented by NaN; text by ''; nested diagnostics
    % by an empty value of the corresponding type.
    [rows,newRows] = harmonize_row_schemas_local(rows,newRows);

    % Enforce exactly one scalar row per experimental key. Historical
    % incremental/replay saves can contain duplicate (round,N,method) rows;
    % replacing only the first match leaves stale rows that are later averaged
    % by paper plotting. Remove ALL matching rows before appending the new row.
    for k = 1:numel(newRows)
        sameKey = false(1,numel(rows));
        for j = 1:numel(rows)
            sameKey(j) = rows(j).roundIndex == newRows(k).roundIndex && ...
                rows(j).nTrain == newRows(k).nTrain && ...
                strcmpi(char(rows(j).method),char(newRows(k).method));
        end
        rows(sameKey) = [];
        rows(end+1) = newRows(k);
    end
end

function [leftRows,rightRows] = harmonize_row_schemas_local(leftRows,rightRows)
    leftNames = fieldnames(leftRows);
    rightNames = fieldnames(rightRows);
    allNames = unique([leftNames;rightNames],'stable');

    for iField = 1:numel(allNames)
        fieldName = allNames{iField};
        prototype = [];
        if isfield(leftRows,fieldName) && ~isempty(leftRows)
            prototype = leftRows(1).(fieldName);
        elseif isfield(rightRows,fieldName) && ~isempty(rightRows)
            prototype = rightRows(1).(fieldName);
        end
        defaultValue = missing_row_field_value_local(prototype);

        if ~isfield(leftRows,fieldName)
            for iRow = 1:numel(leftRows)
                leftRows(iRow).(fieldName) = defaultValue;
            end
        end
        if ~isfield(rightRows,fieldName)
            for iRow = 1:numel(rightRows)
                rightRows(iRow).(fieldName) = defaultValue;
            end
        end
    end

    % Field order is part of MATLAB's struct-assignment compatibility.
    leftRows = orderfields(leftRows,allNames);
    rightRows = orderfields(rightRows,allNames);
end

function value = missing_row_field_value_local(prototype)
    if isnumeric(prototype)
        value = nan(size(prototype));
    elseif islogical(prototype)
        value = false(size(prototype));
    elseif ischar(prototype)
        value = '';
    elseif isstring(prototype)
        value = strings(size(prototype));
    elseif isstruct(prototype)
        value = struct();
    elseif iscell(prototype)
        value = cell(size(prototype));
    else
        value = [];
    end
end

function T = rows_to_flat_table_local(rows)
    if isempty(rows); T = table(); return; end

    n = numel(rows);
    nTrain = reshape([rows.nTrain],[],1);
    roundIndex = reshape([rows.roundIndex],[],1);
    method = string({rows.method}).';
    derivativeRMSE = reshape([rows.derivativeRMSE],[],1);
    oodDerivativeRMSE = reshape([rows.oodDerivativeRMSE],[],1);
    validationMSE = reshape([rows.validationMSE],[],1);
    activeCoefficients = reshape([rows.activeCoefficients],[],1);
    trainTime = reshape([rows.trainTime],[],1);
    modelTrainingSampleCount = reshape([rows.modelTrainingSampleCount],[],1);
    sampleEfficiencyProtocol = string({rows.sampleEfficiencyProtocol}).';
    previousModelRole = string({rows.previousModelRole}).';
    selectedCheckpoint = string({rows.selectedCheckpoint}).';
    selectedCheckpointEpoch = reshape([rows.selectedCheckpointEpoch],[],1);
    selectedCheckpointPhase = string({rows.selectedCheckpointPhase}).';
    selectedCandidateSource = string({rows.selectedCandidateSource}).';
    selectedRestartIndex = reshape([rows.selectedRestartIndex],[],1);
    strictCurrentNImprovementAchieved = reshape([rows.strictCurrentNImprovementAchieved],[],1);
    strictCurrentNValidationTargetMSE = reshape([rows.strictCurrentNValidationTargetMSE],[],1);
    monotoneEnvelopeValidationMSE = reshape([rows.monotoneEnvelopeValidationMSE],[],1);
    monotoneEnvelopeModelTrainingSampleCount = reshape( ...
        [rows.monotoneEnvelopeModelTrainingSampleCount],[],1);
    trajectoryRMSE = reshape([rows.trajectoryRMSE],[],1);
    trajectoryNRMSE = reshape([rows.trajectoryNRMSE],[],1);
    rolloutSuccessRate = reshape([rows.rolloutSuccessRate],[],1);
    rolloutAvailable = logical(reshape([rows.rolloutAvailable],[],1));
    rolloutReason = strings(n,1);
    for k = 1:n; rolloutReason(k) = string(rows(k).rolloutReason); end

    T = table(nTrain,roundIndex,method,derivativeRMSE,oodDerivativeRMSE, ...
        validationMSE,activeCoefficients,trainTime,modelTrainingSampleCount, ...
        sampleEfficiencyProtocol,previousModelRole,selectedCheckpoint, ...
        selectedCheckpointEpoch,selectedCheckpointPhase,selectedCandidateSource, ...
        selectedRestartIndex,strictCurrentNImprovementAchieved, ...
        strictCurrentNValidationTargetMSE,monotoneEnvelopeValidationMSE, ...
        monotoneEnvelopeModelTrainingSampleCount,trajectoryRMSE, ...
        trajectoryNRMSE,rolloutSuccessRate,rolloutAvailable,rolloutReason);
end

function export_pdf_local(fig,pdfPath)
    try
        exportgraphics(fig,pdfPath,'ContentType','vector');
    catch
        set(fig,'PaperPositionMode','auto');
        print(fig,pdfPath,'-dpdf','-painters');
    end
end

function reportPaths = write_recorded_reports_local(outputDir,resultPack)
    reportPaths = struct();
    reportDir = fullfile(outputDir,'method_reports');
    if exist(reportDir,'dir') ~= 7; mkdir(reportDir); end
    methodMap = { ...
        'phdn_g1','PhDN_G1'; 'stage0sr_g1','Stage0SR_G1'; ...
        'phdn_g2','PhDN_G2'; 'stage0sr_g2','Stage0SR_G2'; ...
        'phdn_g3','PhDN_G3'; 'stage0sr_g3','Stage0SR_G3'; ...
        'phdn','PhDN'; 'stage0sr','Stage0SR'; 'mlp','MLP'; ...
        'eql','EQL'; 'kan','KAN'; 'sindy','SINDy'};
    for k = 1:size(methodMap,1)
        fieldName = methodMap{k,1}; fileTag = methodMap{k,2};
        if ~isfield(resultPack,fieldName) || ~isstruct(resultPack.(fieldName)) || ...
                ~isfield(resultPack.(fieldName),'recordedConsoleReport') || ...
                isempty(resultPack.(fieldName).recordedConsoleReport)
            continue;
        end
        reportPath = fullfile(reportDir,sprintf('recorded_report_%s.txt',fileTag));
        fid = fopen(reportPath,'w');
        if fid < 0; warning('Could not write recorded method report: %s',reportPath); continue; end
        cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
        fprintf(fid,'%s',char(resultPack.(fieldName).recordedConsoleReport));
        clear cleanup;
        reportPaths.(fieldName) = reportPath;
    end
end

function atomic_save_local(path,variableName,value)
    parent = fileparts(path);
    if exist(parent,'dir') ~= 7; mkdir(parent); end
    temporaryPath = [tempname(parent) '.mat'];
    payload = struct(); payload.(variableName) = value;
    save(temporaryPath,'-struct','payload','-v7.3');
    [ok,message] = movefile(temporaryPath,path,'f');
    if ~ok
        if exist(temporaryPath,'file') == 2; delete(temporaryPath); end
        error('Atomic save failed for %s: %s',path,message);
    end
end
