function paperFigureInfo = regenerate_single_generator_dynamic_paper_figures(resultsFile,exportDir,varargin)
%REGENERATE_SINGLE_GENERATOR_DYNAMIC_PAPER_FIGURES
% Redraw manuscript figures directly from saved SingleGeneratorDynamic
% public summary results, without retraining or rerunning rollouts.
%
% Required inputs:
%   resultsFile : summary MAT-file, typically
%                 outputs/SingleGeneratorDynamic_SMIB_AVR/summary/
%                 public_summary.mat
%   exportDir   : output directory for the regenerated figure PDFs
%
% Name-value options:
%   'Visible'   : true/false, default true
%   'ExportPDF' : true/false, default true
%   'DeepIntegrityCheck' : false by default. When true, also scans the
%                          potentially large method_results/*.mat files.

    parser = inputParser;
    parser.addRequired('resultsFile',@(x)ischar(x)||isstring(x));
    parser.addRequired('exportDir',@(x)ischar(x)||isstring(x));
    parser.addParameter('Visible',true,@(x)islogical(x)||isnumeric(x));
    parser.addParameter('ExportPDF',true,@(x)islogical(x)||isnumeric(x));
    parser.addParameter('DeepIntegrityCheck',false,@(x)islogical(x)||isnumeric(x));
    parser.parse(resultsFile,exportDir,varargin{:});

    resultsFile = char(parser.Results.resultsFile);
    exportDir = char(parser.Results.exportDir);
    visibleFlag = logical(parser.Results.Visible);
    exportPDF = logical(parser.Results.ExportPDF);
    deepIntegrityCheck = logical(parser.Results.DeepIntegrityCheck);

    if exist(resultsFile,'file') ~= 2
        error(['Saved summary MAT-file was not found:\n%s\n' ...
            'Run the full demo once, or point resultsFile to an existing summary MAT-file.'], ...
            resultsFile);
    end
    if exist(exportDir,'dir') ~= 7
        mkdir(exportDir);
    end

    loaded = load(resultsFile);
    rows = resolve_system_identification_rows_local(loaded);
    if isempty(rows)
        error(['No valid systemIdentificationRows were found in:\n%s\n' ...
            'The MAT-file must contain the saved sample-efficiency rows.'],resultsFile);
    end

    % Harden plot-only regeneration against stale historical scalar rows.
    % Current independent method_results/*.mat post-rollout records are
    % authoritative; per-sample rows and the public summary are fallbacks.
    caseRoot = fileparts(fileparts(resultsFile));
    [rows,rowRepairInfo] = overlay_authoritative_saved_rows_local( ...
        rows,caseRoot,deepIntegrityCheck);
    if deepIntegrityCheck
        fprintf(['Generator replot row check [deep]: sample rows=%d, method rows=%d, ', ...
            'summary duplicates removed=%d.\n'], ...
            rowRepairInfo.nAuthoritativeSampleRows, ...
            rowRepairInfo.nAuthoritativeMethodRows, ...
            rowRepairInfo.nDuplicateRowsRemoved);
    elseif rowRepairInfo.nAuthoritativeSampleRows > 0 || ...
            rowRepairInfo.nDuplicateRowsRemoved > 0
        fprintf(['Generator replot row check [fast]: sample rows=%d, ', ...
            'method-record scan skipped, summary duplicates removed=%d.\n'], ...
            rowRepairInfo.nAuthoritativeSampleRows, ...
            rowRepairInfo.nDuplicateRowsRemoved);
    end
    print_replot_curve_check_local(rows,{'PhDN-G1','Stage0-SR-G1'});

    % Figure 1: all archived/main methods on one figure. Keep exact method names
    % so the legend retains the G3 suffix and other method labels.
    figureSpecs = struct([]);
    figureSpecs(1).tag = 'all_archived_methods_main_comparison';
    figureSpecs(1).titlePrefix = 'SingleGeneratorDynamic paper figure';
    figureSpecs(1).methodOrder = {'SINDy','EQL-Div','MLP','KAN'};
    figureSpecs(1).displayNames = {'SINDy','EQL-Div','MLP','KAN'};
    figureSpecs(1).pdfName = 'single_generator_dynamic_paper_fig_all_archived_methods.pdf';
    figureSpecs(1).addDerivativeZoomInset = false;

    % Figure 2: PhDN prior-level ablation only.
    figureSpecs(2).tag = 'phdn_g1_g2_g3';
    figureSpecs(2).titlePrefix = 'SingleGeneratorDynamic paper figure';
    figureSpecs(2).methodOrder = {'SINDy','PhDN-G1','PhDN-G2','PhDN-G3','Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3'};
    figureSpecs(2).displayNames = {'SINDy','PhDN-G1','PhDN-G2','PhDN-G3','SR-G1','SR-G2','SR-G3'};
    figureSpecs(2).pdfName = 'single_generator_dynamic_paper_fig_phdn_g1_g2_g3.pdf';
    % Keep the full-range main log axis so PhDN-G3 remains visible, and
    % add an in-panel zoom of the upper-error cluster for readability.
    figureSpecs(2).addDerivativeZoomInset = true;

    paperFigureInfo = struct();
    paperFigureInfo.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    paperFigureInfo.sourceResultsFile = resultsFile;
    paperFigureInfo.exportDir = exportDir;
    paperFigureInfo.figures = repmat(struct( ...
        'tag','','figureHandle',[],'pdfPath','','includedMethods',{{}}, ...
        'plotData',struct()),1,numel(figureSpecs));

    for iFigure = 1:numel(figureSpecs)
        spec = figureSpecs(iFigure);
        [selectedRows,missingMethods,includedMethods,includedDisplayNames] = ...
            select_rows_local(rows,spec.methodOrder,spec.displayNames);
        if ~isempty(missingMethods)
            warning('The following requested methods were not found in the saved rows for figure %s: %s', ...
                spec.tag,strjoin(missingMethods,', '));
        end
        [fig,plotData] = plot_selected_sample_efficiency_local( ...
            selectedRows,includedMethods,includedDisplayNames,visibleFlag, ...
            spec.titlePrefix,spec.addDerivativeZoomInset);
        pdfPath = fullfile(exportDir,spec.pdfName);
        if exportPDF
            export_figure_to_pdf_local(fig,pdfPath);
        end

        paperFigureInfo.figures(iFigure).tag = spec.tag;
        paperFigureInfo.figures(iFigure).figureHandle = fig;
        paperFigureInfo.figures(iFigure).pdfPath = pdfPath;
        paperFigureInfo.figures(iFigure).includedMethods = includedMethods;
        paperFigureInfo.figures(iFigure).plotData = plotData;
    end
end

function rows = resolve_system_identification_rows_local(loaded)
    rows = struct([]);
    candidateFields = {'systemIdentificationRows','rows'};
    for iField = 1:numel(candidateFields)
        fieldName = candidateFields{iField};
        if isfield(loaded,fieldName)
            candidate = loaded.(fieldName);
            if isstruct(candidate) && ~isempty(candidate) && ...
                    isfield(candidate,'nTrain') && isfield(candidate,'method')
                rows = candidate;
                return;
            end
        end
    end
    if isfield(loaded,'summaryOutputInfo') && isstruct(loaded.summaryOutputInfo)
        info = loaded.summaryOutputInfo;
        nestedCandidates = {'systemIdentificationRows','rows'};
        for iField = 1:numel(nestedCandidates)
            fieldName = nestedCandidates{iField};
            if isfield(info,fieldName)
                candidate = info.(fieldName);
                if isstruct(candidate) && ~isempty(candidate) && ...
                        isfield(candidate,'nTrain') && isfield(candidate,'method')
                    rows = candidate;
                    return;
                end
            end
        end
    end
end

function [rows,info] = overlay_authoritative_saved_rows_local(rows,caseRoot,deepIntegrityCheck)
    % Fast default replot source precedence:
    %   1) public summary rows,
    %   2) small per-sample system_identification_rows.mat rows.
    %
    % The large method_results/*.mat records are NOT opened during an ordinary
    % replot. They are scanned only when DeepIntegrityCheck=true. This keeps
    % plot-only regeneration fast while preserving a deep recovery/verification
    % path for historical archives. The permanent saver fixes keep the summary
    % and per-sample scalar rows synchronized in normal future runs.

    info = struct('nAuthoritativeSampleRows',0, ...
        'nAuthoritativeMethodRows',0,'nDuplicateRowsRemoved',0);

    [rows,nRemoved] = deduplicate_rows_by_key_local(rows);
    info.nDuplicateRowsRemoved = nRemoved;
    if exist(caseRoot,'dir') ~= 7
        return;
    end

    methodRows = struct([]);
    sampleRows = struct([]);
    roundDirs = dir(fullfile(caseRoot,'round_*'));
    roundDirs = roundDirs([roundDirs.isdir]);
    for iRoundDir = 1:numel(roundDirs)
        roundPath = fullfile(roundDirs(iRoundDir).folder,roundDirs(iRoundDir).name);
        sampleDirs = dir(fullfile(roundPath,'N_*'));
        sampleDirs = sampleDirs([sampleDirs.isdir]);
        for iSampleDir = 1:numel(sampleDirs)
            samplePath = fullfile(sampleDirs(iSampleDir).folder,sampleDirs(iSampleDir).name);

            % Exact rows used by the original run-time trajectory/sample plot.
            sampleRowsPath = fullfile(samplePath,'system_identification_rows.mat');
            if exist(sampleRowsPath,'file') == 2
                try
                    loadedRows = load(sampleRowsPath,'sampleRows');
                    if isfield(loadedRows,'sampleRows') && ...
                            isstruct(loadedRows.sampleRows) && ...
                            ~isempty(loadedRows.sampleRows)
                        for iRow = 1:numel(loadedRows.sampleRows)
                            candidate = loadedRows.sampleRows(iRow);
                            if is_valid_si_row_local(candidate)
                                sampleRows = append_struct_row_schema_safe_local( ...
                                    sampleRows,candidate);
                            end
                        end
                    end
                catch warningInfo
                    warning('Could not read saved sample rows from %s: %s', ...
                        sampleRowsPath,warningInfo.message);
                end
            end

            % Deep mode only: open the potentially large trained-model MAT
            % files and overlay their post-rollout scalar rows as the highest-
            % priority source. Normal replot intentionally skips this disk I/O.
            if deepIntegrityCheck
                methodFiles = dir(fullfile(samplePath,'method_results','*.mat'));
                for iFile = 1:numel(methodFiles)
                    methodPath = fullfile(methodFiles(iFile).folder,methodFiles(iFile).name);
                    try
                        loadedRecord = load(methodPath,'methodRecord');
                    catch
                        continue;
                    end
                    if ~isfield(loadedRecord,'methodRecord') || ...
                            ~isstruct(loadedRecord.methodRecord)
                        continue;
                    end
                    record = loadedRecord.methodRecord;
                    if ~isfield(record,'systemIdentificationRow') || ...
                            ~isstruct(record.systemIdentificationRow) || ...
                            isempty(record.systemIdentificationRow)
                        continue;
                    end
                    candidate = record.systemIdentificationRow;
                    if numel(candidate) ~= 1 || ~is_valid_si_row_local(candidate)
                        continue;
                    end
                    methodRows = append_struct_row_schema_safe_local(methodRows,candidate);
                end
            end
        end
    end

    [methodRows,~] = deduplicate_rows_by_key_local(methodRows);
    [sampleRows,~] = deduplicate_rows_by_key_local(sampleRows);
    info.nAuthoritativeMethodRows = numel(methodRows);
    info.nAuthoritativeSampleRows = numel(sampleRows);

    % Small sample-level rows are the normal authoritative plotting source.
    rows = overlay_rows_by_key_local(rows,sampleRows);
    % Deep mode can additionally verify/recover from independent method records.
    % They are intentionally last so a repaired per-method row wins when used.
    if deepIntegrityCheck
        rows = overlay_rows_by_key_local(rows,methodRows);
    end
end

function tf = is_valid_si_row_local(row)
    tf = isstruct(row) && numel(row)==1 && ...
        isfield(row,'method') && ~isempty(row.method) && ...
        isfield(row,'nTrain') && isnumeric(row.nTrain) && isscalar(row.nTrain) && ...
        isfield(row,'roundIndex') && isnumeric(row.roundIndex) && isscalar(row.roundIndex);
end

function rows = overlay_rows_by_key_local(rows,newRows)
    if isempty(newRows)
        return;
    end
    if isempty(rows)
        rows = newRows;
        return;
    end
    [rows,newRows] = harmonize_struct_array_schemas_local(rows,newRows);
    for k = 1:numel(newRows)
        same = row_key_match_local(rows,newRows(k));
        rows(same) = [];
        rows(end+1) = newRows(k); %#ok<AGROW>
    end
end

function print_replot_curve_check_local(rows,methodNames)
    fprintf('Generator replot trajectory row check:\n');
    for iMethod = 1:numel(methodNames)
        methodName = char(methodNames{iMethod});
        [nValues,yValues] = aggregate_metric_local(rows,methodName,'trajectoryNRMSE');
        if isempty(nValues)
            fprintf('  %-12s : unavailable\n',methodName);
            continue;
        end
        pairs = cell(1,numel(nValues));
        for k = 1:numel(nValues)
            pairs{k} = sprintf('N=%g:%.6e',nValues(k),yValues(k));
        end
        fprintf('  %-12s : %s\n',methodName,strjoin(pairs,' | '));
    end
end

function rows = append_struct_row_schema_safe_local(rows,newRow)
    if isempty(rows)
        rows = newRow;
        return;
    end
    [rows,newRow] = harmonize_struct_array_schemas_local(rows,newRow);
    rows(end+1) = newRow;
end

function [rows,nRemoved] = deduplicate_rows_by_key_local(rows)
    nRemoved = 0;
    if isempty(rows) || ~isstruct(rows)
        return;
    end
    keep = true(1,numel(rows));
    keys = cell(1,numel(rows));
    for k = 1:numel(rows)
        keys{k} = row_key_local(rows(k));
    end
    seen = containers.Map('KeyType','char','ValueType','logical');
    for k = 1:numel(rows)
        key = keys{k};
        if isKey(seen,key)
            keep(k) = false;
            nRemoved = nRemoved + 1;
        else
            seen(key) = true;
        end
    end
    rows = rows(keep);
end

function same = row_key_match_local(rows,row)
    same = false(1,numel(rows));
    target = row_key_local(row);
    for k = 1:numel(rows)
        same(k) = strcmp(row_key_local(rows(k)),target);
    end
end

function key = row_key_local(row)
    roundIndex = numeric_field_key_local(row,'roundIndex',NaN);
    nTrain = numeric_field_key_local(row,'nTrain',NaN);
    method = '';
    if isfield(row,'method') && ~isempty(row.method)
        method = lower(strtrim(char(string(row.method))));
    end
    key = sprintf('R%.15g|N%.15g|M%s',roundIndex,nTrain,method);
end

function value = numeric_field_key_local(s,fieldName,defaultValue)
    value = defaultValue;
    if isfield(s,fieldName) && isnumeric(s.(fieldName)) && isscalar(s.(fieldName))
        value = s.(fieldName);
    end
end

function [leftRows,rightRows] = harmonize_struct_array_schemas_local(leftRows,rightRows)
    if isempty(leftRows) || isempty(rightRows)
        return;
    end
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
    leftRows = orderfields(leftRows,allNames);
    rightRows = orderfields(rightRows,allNames);
end

function value = first_available_field_value_local(leftRows,rightRows,fieldName)
    value = [];
    groups = {leftRows,rightRows};
    for iGroup = 1:numel(groups)
        group = groups{iGroup};
        if ~isfield(group,fieldName); continue; end
        for iRow = 1:numel(group)
            candidate = group(iRow).(fieldName);
            if ~isempty(candidate)
                value = candidate;
                return;
            end
        end
        if ~isempty(group)
            value = group(1).(fieldName);
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

function [selectedRows,missingMethods,includedMethods,includedDisplayNames] = ...
        select_rows_local(rows,methodOrder,displayNames)
    selectedRows = struct([]);
    missingMethods = {};
    includedMethods = {};
    includedDisplayNames = {};
    for iMethod = 1:numel(methodOrder)
        methodName = char(methodOrder{iMethod});
        idx = strcmp({rows.method},methodName);
        if any(idx)
            if isempty(selectedRows)
                selectedRows = rows(idx);
            else
                selectedRows = [selectedRows, rows(idx)]; %#ok<AGROW>
            end
            includedMethods{end+1} = methodName; %#ok<AGROW>
            includedDisplayNames{end+1} = char(displayNames{iMethod}); %#ok<AGROW>
        else
            missingMethods{end+1} = methodName; %#ok<AGROW>
        end
    end
end

function [fig, plotData] = plot_selected_sample_efficiency_local(rows,methodOrder,displayNames,visibleFlag,titlePrefix,addDerivativeZoomInset)
    if isempty(rows)
        error('No rows were selected for plotting.');
    end
    figVisible = ternary_local(visibleFlag,'on','off');
    fig = figure('Name','SingleGeneratorDynamic paper figures', ...
        'Color','w','Visible',figVisible,'Renderer','painters');
    layout = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');

    nMethods = numel(methodOrder);
    [methodColors,methodLineStyles,methodMarkers,methodMarkerFaceModes] = build_method_styles_local(methodOrder);

    plotData = struct();
    plotData.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    plotData.methods = repmat(struct( ...
        'internalName','','displayName','','color',zeros(1,3),'lineStyle','-', ...
        'marker','o','markerFaceMode','open', ...
        'nTrain',[],'derivativeRMSE',[],'trajectoryNRMSE',[], ...
        'trajectoryStatus',''),1,nMethods);

    axDerivative = nexttile(layout);
    derivativeLegendHandles = gobjects(0);
    hold(axDerivative,'on');
    for iMethod = 1:nMethods
        internalName = char(methodOrder{iMethod});
        displayName = char(displayNames{iMethod});
        thisColor = methodColors(iMethod,:);
        thisLineStyle = methodLineStyles{iMethod};
        thisMarker = methodMarkers{iMethod};
        thisMarkerFaceMode = methodMarkerFaceModes{iMethod};
        thisMarkerFaceColor = 'none';
        if strcmpi(thisMarkerFaceMode,'filled')
            thisMarkerFaceColor = thisColor;
        end

        thisLineWidth = 1.4;
        if strcmpi(internalName,'SINDy')
            thisLineWidth = 2;
        end

        [nDerivative,yDerivative] = aggregate_metric_local(rows,internalName,'derivativeRMSE');
        valid = isfinite(yDerivative) & yDerivative>0;
        if any(valid)
            hDerivative = loglog(axDerivative,nDerivative(valid),yDerivative(valid), ...
                'LineStyle',thisLineStyle,'Marker',thisMarker, ...
                'Color',thisColor,'MarkerFaceColor',thisMarkerFaceColor, ...
                'DisplayName',displayName,'LineWidth',thisLineWidth,'MarkerSize',5);
            derivativeLegendHandles(end+1) = hDerivative; %#ok<AGROW>
        else
            hDerivative = loglog(axDerivative,NaN,NaN,thisMarker,'LineStyle','none', ...
                'Color',thisColor,'MarkerFaceColor',thisMarkerFaceColor, ...
                'DisplayName',[displayName ' (N/A)'],'MarkerSize',5);
            derivativeLegendHandles(end+1) = hDerivative; %#ok<AGROW>
        end
        plotData.methods(iMethod).internalName = internalName;
        plotData.methods(iMethod).displayName = displayName;
        plotData.methods(iMethod).color = thisColor;
        plotData.methods(iMethod).lineStyle = thisLineStyle;
        plotData.methods(iMethod).marker = thisMarker;
        plotData.methods(iMethod).markerFaceMode = thisMarkerFaceMode;
        plotData.methods(iMethod).nTrain = nDerivative;
        plotData.methods(iMethod).derivativeRMSE = yDerivative;
    end
    set(axDerivative,'XScale','log','YScale','log');
    grid(axDerivative,'on');
    xlabel(axDerivative,'Training samples');
    ylabel(axDerivative,'Vector-field RMSE');
    title(axDerivative,'Instantaneous derivative identification');

    axTrajectory = nexttile(layout);
    hold(axTrajectory,'on');
    rolloutFailures = struct('displayName',{},'color',{},'nTrain',{},'allFailed',{});
    for iMethod = 1:nMethods
        internalName = char(methodOrder{iMethod});
        displayName = char(displayNames{iMethod});
        thisColor = methodColors(iMethod,:);
        thisLineStyle = methodLineStyles{iMethod};
        thisMarker = methodMarkers{iMethod};
        thisMarkerFaceMode = methodMarkerFaceModes{iMethod};
        thisMarkerFaceColor = 'none';
        if strcmpi(thisMarkerFaceMode,'filled')
            thisMarkerFaceColor = thisColor;
        end

        thisLineWidth = 1.4;
        if strcmpi(internalName,'SINDy')
            thisLineWidth = 2;
        end

        [nTrajectory,yTrajectory] = aggregate_metric_local(rows,internalName,'trajectoryNRMSE');
        valid = isfinite(yTrajectory) & yTrajectory>0;
        failed = isinf(yTrajectory);
        if any(valid)
            loglog(axTrajectory,nTrajectory(valid),yTrajectory(valid), ...
                'LineStyle',thisLineStyle,'Marker',thisMarker, ...
                'Color',thisColor,'MarkerFaceColor',thisMarkerFaceColor, ...
                'DisplayName',displayName,'LineWidth',thisLineWidth,'MarkerSize',5);
            if any(failed)
                trajectoryStatus = 'partially_failed';
            else
                trajectoryStatus = 'available';
            end
        elseif any(failed)
            % Inf cannot be placed on a logarithmic axis. Keep the method in
            % the shared legend through its derivative curve and add an explicit
            % non-numeric failure annotation inside the rollout panel below.
            trajectoryStatus = 'failed';
        else
            loglog(axTrajectory,NaN,NaN,thisMarker,'LineStyle','none','Color',thisColor, ...
                'MarkerFaceColor',thisMarkerFaceColor,'LineWidth',1.0,'MarkerSize',6, ...
                'DisplayName',[displayName ' (N/A)']);
            trajectoryStatus = 'unavailable';
        end
        if any(failed)
            iFailure = numel(rolloutFailures)+1;
            rolloutFailures(iFailure).displayName = displayName;
            rolloutFailures(iFailure).color = thisColor;
            rolloutFailures(iFailure).nTrain = nTrajectory(failed);
            rolloutFailures(iFailure).allFailed = all(failed) && ~any(valid);
        end
        plotData.methods(iMethod).trajectoryNRMSE = yTrajectory;
        plotData.methods(iMethod).trajectoryStatus = trajectoryStatus;
    end
    set(axTrajectory,'XScale','log','YScale','log');
    grid(axTrajectory,'on');
    xlabel(axTrajectory,'Training samples');
    ylabel(axTrajectory,'Trajectory NRMSE');
    title(axTrajectory,'Short-horizon unseen-initial-condition rollout');
    add_rollout_failure_annotations_local(axTrajectory,rolloutFailures);

    positiveN = unique([rows.nTrain]);
    positiveN = positiveN(isfinite(positiveN) & positiveN>0);
    if ~isempty(positiveN)
        if numel(positiveN)==1
            commonXLim = positiveN(1)*[0.8 1.25];
        else
            % Add a small endpoint margin and use only actual sample counts as
            % major ticks. This avoids crowded automatic 1600/1800/2000 labels.
            commonXLim = [min(positiveN)/1.08, max(positiveN)*1.08];
        end
        set([axDerivative,axTrajectory], ...
            'XLim',commonXLim, ...
            'XTick',positiveN, ...
            'XTickLabel',compose('%g',positiveN), ...
            'XTickLabelRotation',0);
    end

    % Figure-2 readability aid: retain the full main derivative y-range (so
    % the near-machine-precision PhDN-G3 curve remains visible) and overlay a
    % compact in-panel zoom of the upper-error cluster. The inset excludes only
    % PhDN-G3 because its purpose is to separate the otherwise crowded curves.
    plotData.derivativeZoomInset = struct('enabled',false,'yLimits',[],'excludedMethod','');
    if nargin >= 6 && logical(addDerivativeZoomInset)
        plotData.derivativeZoomInset = add_derivative_zoom_inset_local( ...
            fig,axDerivative,plotData,'PhDN-G3');
    end

    % One shared, single-row legend beneath both subplots. The figure-level
    % title is intentionally omitted for manuscript use.
    if ~isempty(derivativeLegendHandles)
        sharedLegend = legend(axTrajectory,derivativeLegendHandles, ...
            get(derivativeLegendHandles,'DisplayName'), ...
            'Location','southoutside','Orientation','horizontal', ...
            'NumColumns',numel(derivativeLegendHandles), ...
            'Interpreter','none','FontSize',8,'Box','on');
        sharedLegend.Layout.Tile = 'south';
        try
            sharedLegend.ItemTokenSize = [12 8];
        catch
        end
    end

    set(fig,'Visible',figVisible);
    drawnow;
    % Preserve the original figure size/aspect ratio and axes layout.
    % Only nudge a y-axis label when its rendered left edge would be clipped.
    ensure_y_axis_labels_inside_figure_local([axDerivative,axTrajectory],4);
    drawnow;
end

function ensure_y_axis_labels_inside_figure_local(axArray,leftMarginPixels)
%ENSURE_Y_AXIS_LABELS_INSIDE_FIGURE_LOCAL
% Preserve the original figure proportion and tiledlayout geometry. This
% helper changes only the y-label position, and only when the label extends
% beyond the left edge of the figure during vector-PDF export.

    if nargin < 2 || isempty(leftMarginPixels)
        leftMarginPixels = 4;
    end

    drawnow;
    for iAx = 1:numel(axArray)
        ax = axArray(iAx);
        if ~isgraphics(ax,'axes') || isempty(ax.YLabel) || ~isgraphics(ax.YLabel)
            continue;
        end

        oldAxUnits = ax.Units;
        oldLabelUnits = ax.YLabel.Units;
        ax.Units = 'pixels';
        ax.YLabel.Units = 'pixels';

        axPosition = ax.Position;
        labelExtent = ax.YLabel.Extent;
        labelPosition = ax.YLabel.Position;
        labelLeftInFigure = axPosition(1) + labelExtent(1);

        if isfinite(labelLeftInFigure) && labelLeftInFigure < leftMarginPixels
            labelPosition(1) = labelPosition(1) + ...
                (leftMarginPixels - labelLeftInFigure);
            ax.YLabel.Position = labelPosition;
        end

        ax.YLabel.Units = oldLabelUnits;
        ax.Units = oldAxUnits;
    end
end

function [methodColors,methodLineStyles,methodMarkers,methodMarkerFaceModes] = build_method_styles_local(methodOrder)
%BUILD_METHOD_STYLES_LOCAL
% Pair each Stage0-SR-Gk curve with the same color as PhDN-Gk. Encode the
% G-level with marker shape and PhDN-vs-SR with line/fill style:
%   G1=o, G2=s, G3=^; PhDN=solid+filled; SR=dashed+open.
% Non-grouped methods retain solid lines and open circle markers.

    nMethods = numel(methodOrder);
    groupIndex = zeros(1,nMethods);
    isGroupedMethod = false(1,nMethods);
    isStage0SR = false(1,nMethods);

    for iMethod = 1:nMethods
        methodName = char(methodOrder{iMethod});
        token = regexpi(methodName,'^(?:PhDN-G|Stage0-SR-G)([123])$','tokens','once');
        if ~isempty(token)
            groupIndex(iMethod) = str2double(token{1});
            isGroupedMethod(iMethod) = true;
        end
        isStage0SR(iMethod) = startsWith(methodName,'Stage0-SR-','IgnoreCase',true);
    end

    nOtherMethods = sum(~isGroupedMethod);
    palette = lines(max(3+nOtherMethods,3));
    methodColors = zeros(nMethods,3);
    methodLineStyles = repmat({'-'},1,nMethods);
    methodMarkers = repmat({'o'},1,nMethods);
    methodMarkerFaceModes = repmat({'open'},1,nMethods);
    groupMarkers = {'o','s','^'};
    nextOtherColor = 4;

    for iMethod = 1:nMethods
        if isGroupedMethod(iMethod)
            group = groupIndex(iMethod);
            methodColors(iMethod,:) = palette(group,:);
            methodMarkers{iMethod} = groupMarkers{group};
            if isStage0SR(iMethod)
                methodLineStyles{iMethod} = '--';
                methodMarkerFaceModes{iMethod} = 'open';
            else
                methodLineStyles{iMethod} = '-';
                methodMarkerFaceModes{iMethod} = 'filled';
            end
        else
            methodColors(iMethod,:) = palette(nextOtherColor,:);
            nextOtherColor = nextOtherColor + 1;
        end
    end
end

function insetInfo = add_derivative_zoom_inset_local(fig,mainAx,plotData,excludedMethod)
%ADD_DERIVATIVE_ZOOM_INSET_LOCAL
% Add a zoom panel INSIDE the derivative axes footprint without resizing the
% tiled layout. The full main axis is unchanged; therefore the very-low-error
% PhDN-G3 curve remains visible and continues to demonstrate its advantage.

    insetInfo = struct('enabled',false,'yLimits',[],'excludedMethod',excludedMethod);
    if isempty(plotData.methods) || ~isgraphics(mainAx,'axes')
        return;
    end

    allUpperY = [];
    for iMethod = 1:numel(plotData.methods)
        item = plotData.methods(iMethod);
        if strcmpi(item.internalName,excludedMethod)
            continue;
        end
        y = item.derivativeRMSE(:).';
        y = y(isfinite(y) & y>0);
        allUpperY = [allUpperY,y]; %#ok<AGROW>
    end
    if isempty(allUpperY)
        return;
    end

    yMin = min(allUpperY);
    yMax = max(allUpperY);
    if ~(isfinite(yMin) && isfinite(yMax) && yMin>0 && yMax>0)
        return;
    end
    if yMax <= yMin
        yMin = yMin/2;
        yMax = yMax*2;
    else
        yMin = yMin/1.6;
        yMax = yMax*1.6;
    end

    drawnow;
    oldMainUnits = mainAx.Units;
    mainAx.Units = 'normalized';
    p = mainAx.Position;
    mainAx.Units = oldMainUnits;

    % Center the inset in the large blank log-error gap.  It is slightly wider
    % than the previous version for better horizontal resolution, and slightly
    % lower so the inset title stays clear of the upper SR-G3 dashed curve.
    % The main axes footprint is unchanged, so no paper-panel width is
    % lost. insetPosition = [x, y, w, h];
    insetPosition = [p(1)+0.26*p(3), p(2)+0.36*p(4), 0.56*p(3), 0.35*p(4)];
    axInset = axes('Parent',fig,'Units','normalized','Position',insetPosition, ...
        'Color','w','Box','on','FontSize',6,'Layer','top');
    hold(axInset,'on');

    for iMethod = 1:numel(plotData.methods)
        item = plotData.methods(iMethod);
        if strcmpi(item.internalName,excludedMethod)
            continue;
        end
        x = item.nTrain(:).';
        y = item.derivativeRMSE(:).';
        valid = isfinite(x) & x>0 & isfinite(y) & y>0;
        if ~any(valid)
            continue;
        end
        markerFaceColor = 'none';
        if strcmpi(item.markerFaceMode,'filled')
            markerFaceColor = item.color;
        end
        thisLineWidth = 1.0;
        if strcmpi(item.internalName,'SINDy')
            thisLineWidth = 1.3;
        end
        loglog(axInset,x(valid),y(valid), ...
            'LineStyle',item.lineStyle,'Marker',item.marker, ...
            'Color',item.color,'MarkerFaceColor',markerFaceColor, ...
            'LineWidth',thisLineWidth,'MarkerSize',3.5,'HandleVisibility','off');
    end

    set(axInset,'XScale','log','YScale','log','YLim',[yMin,yMax], ...
        'XLim',mainAx.XLim,'XTick',mainAx.XTick, ...
        'XTickLabel',mainAx.XTickLabel,'XTickLabelRotation',0, ...
        'TickDir','in','LineWidth',0.75);
    grid(axInset,'on');
    % title(axInset,'Upper-error zoom','FontSize',7,'FontWeight','normal', ...
    %     'Units','normalized','Position',[0.50 1.01 0]);

    insetInfo.enabled = true;
    insetInfo.yLimits = [yMin,yMax];
end

function add_rollout_failure_annotations_local(ax,failures)
%ADD_ROLLOUT_FAILURE_ANNOTATIONS_LOCAL Show failed methods without assigning
% an artificial finite NRMSE value on the logarithmic trajectory axis.
% Keep the annotation INSIDE the axes at the east-center so it does not
% consume extra figure width or compress the plotted data region.
    if isempty(failures)
        return;
    end

    yStep = 0.10;
    yCenter = 0.55;
    yTop = yCenter + 0.5*(numel(failures)-1)*yStep;
    for iFailure = 1:numel(failures)
        item = failures(iFailure);
        if item.allFailed
            label = sprintf('%s rollout: failed',item.displayName);
        else
            nText = format_numeric_list_local(item.nTrain);
            label = sprintf('%s rollout failed at N=%s',item.displayName,nText);
        end
        text(ax,0.985,yTop-(iFailure-1)*yStep,label, ...
            'Units','normalized','HorizontalAlignment','right', ...
            'VerticalAlignment','middle','Interpreter','none', ...
            'Color',item.color,'FontSize',8,'FontWeight','bold', ...
            'BackgroundColor','w','EdgeColor',item.color,'Margin',2, ...
            'Clipping','on');
    end
end

function out = format_numeric_list_local(values)
    values = values(:).';
    if isempty(values)
        out = 'unknown';
        return;
    end
    pieces = arrayfun(@(x)sprintf('%g',x),values,'UniformOutput',false);
    out = strjoin(pieces,',');
end

function [nValues,means] = aggregate_metric_local(rows,methodName,fieldName)
    idxMethod = strcmp({rows.method},methodName);
    nValues = unique([rows(idxMethod).nTrain]);
    nValues = sort(nValues);
    means = nan(size(nValues));
    for k = 1:numel(nValues)
        idx = idxMethod & [rows.nTrain] == nValues(k);
        rawValues = [rows(idx).(fieldName)];
        values = rawValues(isfinite(rawValues));
        if any(isinf(rawValues))
            means(k) = Inf;
        elseif ~isempty(values)
            means(k) = mean(values);
        end
    end
end

function export_figure_to_pdf_local(fig,pdfPath)
    drawnow;
    try
        exportgraphics(fig,pdfPath,'ContentType','vector');
    catch
        set(fig,'PaperPositionMode','auto');
        print(fig,pdfPath,'-dpdf','-painters');
    end
end

function out = ternary_local(cond,trueValue,falseValue)
    if cond
        out = trueValue;
    else
        out = falseValue;
    end
end
