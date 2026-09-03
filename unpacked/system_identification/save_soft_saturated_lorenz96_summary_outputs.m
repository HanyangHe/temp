function outputInfo = save_soft_saturated_lorenz96_summary_outputs( ...
    summaryDir,allResults,systemIdentificationRows,standardSummaryRows, ...
    sampleEfficiencyTable,sampleEfficiencyFigure,sampleEfficiencyFigureData, ...
    runMetadata)
%SAVE_SOFT_SATURATED_LORENZ96_SUMMARY_OUTPUTS Persist a slim aggregate summary.
%
% Independent method_results/*.mat files are the authoritative trained-model
% checkpoints.  The public summary intentionally does NOT duplicate allResults.
% The allResults input is retained only for backward-compatible call syntax.

    %#ok<INUSD> allResults is intentionally not persisted.
    if exist(summaryDir,'dir') ~= 7; mkdir(summaryDir); end

    outputInfo = struct();
    outputInfo.resultsMatPath = fullfile(summaryDir,'public_summary.mat');
    outputInfo.publicSummaryMatPath = outputInfo.resultsMatPath;
    outputInfo.legacyResultsMatPath = fullfile(summaryDir,'soft_saturated_lorenz96_results.mat');
    outputInfo.methodSummaryCsvPath = fullfile(summaryDir,'method_comparison.csv');
    outputInfo.sampleEfficiencyCsvPath = fullfile(summaryDir,'sample_efficiency.csv');
    outputInfo.figureDataPath = fullfile(summaryDir,'sample_efficiency_figure_data.mat');
    outputInfo.figurePdfPath = fullfile(summaryDir,'sample_efficiency.pdf');
    outputInfo.figureFigPath = fullfile(summaryDir,'sample_efficiency.fig');
    outputInfo.runMetadataPath = fullfile(summaryDir,'run_metadata.mat');
    outputInfo.runMetadataTextPath = fullfile(summaryDir,'run_metadata.txt');

    % One-time migration of an existing giant aggregate.  Only the small
    % plotting/table fields are extracted.  Deletion occurs only after the
    % new public_summary.mat has been written and verified.
    if exist(outputInfo.resultsMatPath,'file') ~= 2 && ...
            exist(outputInfo.legacyResultsMatPath,'file') == 2
        migrate_system_identification_summary_to_public( ...
            outputInfo.legacyResultsMatPath,outputInfo.resultsMatPath, ...
            'DeleteLegacy',true);
    end

    oldRows = struct([]);
    oldSummaryRows = struct([]);
    oldSampleEfficiencyTable = table();
    if exist(outputInfo.resultsMatPath,'file') == 2
        previous = load(outputInfo.resultsMatPath, ...
            'systemIdentificationRows','standardSummaryRows', ...
            'sampleEfficiencyTable');
        if isfield(previous,'systemIdentificationRows') && ...
                isstruct(previous.systemIdentificationRows)
            oldRows = previous.systemIdentificationRows;
        end
        if isfield(previous,'standardSummaryRows') && ...
                isstruct(previous.standardSummaryRows)
            oldSummaryRows = previous.standardSummaryRows;
        end
        if isfield(previous,'sampleEfficiencyTable') && ...
                istable(previous.sampleEfficiencyTable)
            oldSampleEfficiencyTable = previous.sampleEfficiencyTable;
        end
    end

    systemIdentificationRows = merge_si_rows_local(oldRows,systemIdentificationRows);
    standardSummaryRows = merge_summary_rows_local(oldSummaryRows,standardSummaryRows);
    sampleEfficiencyTable = merge_tables_local(oldSampleEfficiencyTable,sampleEfficiencyTable);

    atomic_save_many_local(outputInfo.resultsMatPath,struct( ...
        'systemIdentificationRows',systemIdentificationRows, ...
        'standardSummaryRows',standardSummaryRows, ...
        'sampleEfficiencyTable',sampleEfficiencyTable, ...
        'runMetadata',runMetadata));
    atomic_save_local(outputInfo.figureDataPath, ...
        'sampleEfficiencyFigureData',sampleEfficiencyFigureData);
    atomic_save_local(outputInfo.runMetadataPath,'runMetadata',runMetadata);

    if istable(sampleEfficiencyTable)
        writetable(sampleEfficiencyTable,outputInfo.sampleEfficiencyCsvPath);
    end
    if ~isempty(standardSummaryRows)
        methodSummaryTable = struct_rows_to_table_local(standardSummaryRows);
        writetable(methodSummaryTable,outputInfo.methodSummaryCsvPath);
    end

    if isgraphics(sampleEfficiencyFigure)
        savefig(sampleEfficiencyFigure,outputInfo.figureFigPath);
        export_pdf_local(sampleEfficiencyFigure,outputInfo.figurePdfPath);
    else
        outputInfo.figurePdfPath = '';
        outputInfo.figureFigPath = '';
    end

    write_metadata_text_local(runMetadata,outputInfo.runMetadataTextPath);
    fprintf('Saved slim aggregate SoftSaturatedLorenz96 summary to:\n%s\n', ...
        outputInfo.resultsMatPath);
end

function merged = merge_fields_local(oldStruct,newStruct)
    merged = oldStruct;
    if ~isstruct(newStruct); merged = newStruct; return; end
    names = fieldnames(newStruct);
    for k = 1:numel(names); merged.(names{k}) = newStruct.(names{k}); end
end

function rows = merge_si_rows_local(oldRows,newRows)
    rows = oldRows;
    if isempty(newRows); return; end
    if isempty(rows); rows = newRows; return; end

    % Aggregate files may have been written by an earlier v75 row schema.
    % MATLAB struct-array assignment requires identical field names and field
    % order, so upgrade both sides to the union schema before replacement.
    [rows,newRows] = harmonize_struct_array_schemas_local(rows,newRows);

    % Keep one and only one row for each (round,N,method) key.
    % Removing all historical matches prevents stale replay rows from being
    % interpreted as additional statistical rounds by a later replot.
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

function rows = merge_summary_rows_local(oldRows,newRows)
    rows = oldRows;
    if isempty(newRows); return; end
    if isempty(rows); rows = newRows; return; end

    % standardSummaryRows evolved independently from the sample-efficiency
    % rows and needs the same schema upgrade before indexed assignment.
    [rows,newRows] = harmonize_struct_array_schemas_local(rows,newRows);

    % standardSummaryRows use (round,baseCaseName,method) as the key.
    % Remove every old match, not only the first, before appending the latest
    % record so summary archives cannot accumulate stale duplicates.
    for k = 1:numel(newRows)
        sameKey = false(1,numel(rows));
        for j = 1:numel(rows)
            sameRound = isequaln(rows(j).roundIndex,newRows(k).roundIndex);
            sameMethod = strcmpi(char(rows(j).method),char(newRows(k).method));
            sameCase = strcmp(char(rows(j).baseCaseName),char(newRows(k).baseCaseName));
            sameKey(j) = sameRound && sameMethod && sameCase;
        end
        rows(sameKey) = [];
        rows(end+1) = newRows(k);
    end
end

function merged = merge_tables_local(oldTable,newTable)
    if ~istable(oldTable) || isempty(oldTable); merged = newTable; return; end
    if ~istable(newTable) || isempty(newTable); merged = oldTable; return; end

    % Do not silently discard the historical table when a new metric column
    % (for example dX_OOD_RMSE) is introduced. Convert to row structs, align
    % the schemas, then merge by the same experimental key.
    oldRows = table2struct(oldTable);
    newRows = table2struct(newTable);
    [oldRows,newRows] = harmonize_struct_array_schemas_local(oldRows,newRows);

    requiredKeys = {'nTrain','roundIndex','method'};
    if all(isfield(oldRows,requiredKeys)) && all(isfield(newRows,requiredKeys))
        mergedRows = merge_si_rows_local(oldRows,newRows);
    else
        mergedRows = [oldRows(:);newRows(:)];
    end
    merged = struct_rows_to_table_local(mergedRows);
end

function [leftRows,rightRows] = harmonize_struct_array_schemas_local(leftRows,rightRows)
    leftNames = fieldnames(leftRows);
    rightNames = fieldnames(rightRows);
    allNames = unique([leftNames;rightNames],'stable');

    for iField = 1:numel(allNames)
        fieldName = allNames{iField};
        prototype = first_available_field_value_local(leftRows,rightRows,fieldName);
        defaultValue = missing_field_value_local(prototype);

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

    % Field order is also part of MATLAB struct-assignment compatibility.
    leftRows = orderfields(leftRows,allNames);
    rightRows = orderfields(rightRows,allNames);
end

function value = first_available_field_value_local(leftRows,rightRows,fieldName)
    value = [];
    groups = {leftRows,rightRows};
    for iGroup = 1:numel(groups)
        rows = groups{iGroup};
        if ~isfield(rows,fieldName); continue; end
        for iRow = 1:numel(rows)
            candidate = rows(iRow).(fieldName);
            if ~isempty(candidate)
                value = candidate;
                return;
            end
        end
        if ~isempty(rows)
            value = rows(1).(fieldName);
        end
    end
end

function value = missing_field_value_local(prototype)
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
    elseif istable(prototype)
        value = table();
    else
        value = [];
    end
end

function T = struct_rows_to_table_local(rows)
%STRUCT_ROWS_TO_TABLE_LOCAL Convert one-method and multi-method summaries.
% Empty character fields make default struct2table fail for a scalar struct;
% AsArray=true explicitly treats it as one experimental record.
    if isempty(rows)
        T = table();
    elseif isscalar(rows)
        T = struct2table(rows,'AsArray',true);
    else
        T = struct2table(rows);
    end
end

function write_metadata_text_local(meta,path)
    fid = fopen(path,'w');
    if fid < 0; warning('Could not open run metadata text file: %s',path); return; end
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
    fprintf(fid,'SoftSaturatedLorenz96 soft-saturated Lorenz--96 run metadata\n');
    fprintf(fid,'Generated at: %s\n',string_field_local(meta,'generatedAt',''));
    fprintf(fid,'Complete case wall time [s]: %.6f\n',numeric_field_local(meta,'completeCaseWallTime',NaN));
    fprintf(fid,'Complete case wall time [h]: %.6f\n',numeric_field_local(meta,'completeCaseWallTime',NaN)/3600);
    if isfield(meta,'trainingSampleList')
        fprintf(fid,'Training sample list: [%s]\n',num2str(meta.trainingSampleList));
    end
    fprintf(fid,'Validation samples: %g\n',numeric_field_local(meta,'nValidationSamples',NaN));
    fprintf(fid,'ID test samples: %g\n',numeric_field_local(meta,'nIDTestSamples',NaN));
    fprintf(fid,'Rounds: %g\n',numeric_field_local(meta,'numRounds',NaN));
    fprintf(fid,'Output root: %s\n',string_field_local(meta,'outputRoot',''));
end

function value = numeric_field_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && isnumeric(s.(name)) && isscalar(s.(name))
        value = double(s.(name));
    else
        value = defaultValue;
    end
end

function value = string_field_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = char(string(s.(name)));
    else
        value = defaultValue;
    end
end

function export_pdf_local(fig,pdfPath)
    try
        exportgraphics(fig,pdfPath,'ContentType','vector');
    catch
        set(fig,'PaperPositionMode','auto');
        print(fig,pdfPath,'-dpdf','-painters');
    end
end

function atomic_save_local(path,variableName,value)
    payload = struct(); payload.(variableName) = value;
    atomic_save_many_local(path,payload);
end

function atomic_save_many_local(path,payload)
    parent = fileparts(path);
    if exist(parent,'dir') ~= 7; mkdir(parent); end
    temporaryPath = [tempname(parent) '.mat'];
    save(temporaryPath,'-struct','payload','-v7.3');
    [ok,message] = movefile(temporaryPath,path,'f');
    if ~ok
        if exist(temporaryPath,'file') == 2; delete(temporaryPath); end
        error('Atomic save failed for %s: %s',path,message);
    end
end
