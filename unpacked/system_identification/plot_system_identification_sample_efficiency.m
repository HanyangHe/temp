function [fig, plotData] = plot_system_identification_sample_efficiency(rows,caseLabel)
%PLOT_SYSTEM_IDENTIFICATION_SAMPLE_EFFICIENCY Plot mean curves over rounds.
%
% The same method color is reserved in both panels. Methods without an
% arbitrary-state rollout predictor remain visible in the rollout legend as
% N/A, while methods whose integrations failed are marked as failed.

    if nargin < 2 || isempty(caseLabel)
        caseLabel = 'SingleGeneratorDynamic';
    end
    caseLabel = char(string(caseLabel));

    methods = unique({rows.method},'stable');
    [methodColors,methodLineStyles,methodMarkers,methodMarkerFaceModes] = ...
        build_method_styles_local(methods);

    fig = figure('Name',[caseLabel ' sample efficiency'],'Color','w','Visible','on');
    tiledlayout(2,1,'TileSpacing','compact','Padding','compact');

    plotData = struct();
    plotData.methods = repmat(struct( ...
        'internalName','','displayName','','color',zeros(1,3), ...
        'lineStyle','-','marker','o','markerFaceMode','open', ...
        'nTrain',[],'derivativeRMSE',[],'trajectoryNRMSE',[], ...
        'trajectoryStatus',''),1,numel(methods));

    axDerivative = nexttile;
    hold(axDerivative,'on');
    for i = 1:numel(methods)
        internalName = methods{i};
        displayName = display_method_name_local(internalName);
        thisColor = methodColors(i,:);
        thisLineStyle = methodLineStyles{i};
        thisMarker = methodMarkers{i};
        thisMarkerFaceColor = 'none';
        if strcmpi(methodMarkerFaceModes{i},'filled'); thisMarkerFaceColor = thisColor; end
        [nDerivative,yDerivative] = aggregate_metric_local(rows,internalName,'derivativeRMSE');
        valid = isfinite(yDerivative) & yDerivative>0;
        if any(valid)
            loglog(axDerivative,nDerivative(valid),yDerivative(valid), ...
                'LineStyle',thisLineStyle,'Marker',thisMarker, ...
                'Color',thisColor,'MarkerFaceColor',thisMarkerFaceColor, ...
                'DisplayName',displayName,'LineWidth',1.2);
        else
            loglog(axDerivative,NaN,NaN,thisMarker,'LineStyle','none','Color',thisColor, ...
                'MarkerFaceColor',thisMarkerFaceColor,'DisplayName',[displayName ' (N/A)']);
        end

        plotData.methods(i).internalName = internalName;
        plotData.methods(i).displayName = displayName;
        plotData.methods(i).color = thisColor;
        plotData.methods(i).lineStyle = thisLineStyle;
        plotData.methods(i).marker = thisMarker;
        plotData.methods(i).markerFaceMode = methodMarkerFaceModes{i};
        plotData.methods(i).nTrain = nDerivative;
        plotData.methods(i).derivativeRMSE = yDerivative;
    end
    set(axDerivative,'XScale','log','YScale','log');
    grid(axDerivative,'on');
    xlabel(axDerivative,'Training samples');
    ylabel(axDerivative,'Vector-field RMSE');
    title(axDerivative,'Instantaneous derivative identification');
    legend(axDerivative,'Location','best','Interpreter','none');

    axTrajectory = nexttile;
    hold(axTrajectory,'on');
    rolloutFailures = struct('displayName',{},'color',{},'nTrain',{},'allFailed',{});
    for i = 1:numel(methods)
        internalName = methods{i};
        displayName = display_method_name_local(internalName);
        thisColor = methodColors(i,:);
        thisLineStyle = methodLineStyles{i};
        thisMarker = methodMarkers{i};
        thisMarkerFaceColor = 'none';
        if strcmpi(methodMarkerFaceModes{i},'filled'); thisMarkerFaceColor = thisColor; end
        [nTrajectory,yTrajectory] = aggregate_metric_local(rows,internalName,'trajectoryNRMSE');
        valid = isfinite(yTrajectory) & yTrajectory>0;
        failed = isinf(yTrajectory);

        if any(valid)
            loglog(axTrajectory,nTrajectory(valid),yTrajectory(valid), ...
                'LineStyle',thisLineStyle,'Marker',thisMarker, ...
                'Color',thisColor,'MarkerFaceColor',thisMarkerFaceColor, ...
                'DisplayName',displayName,'LineWidth',1.2);
            if any(failed)
                trajectoryStatus = 'partially_failed';
            else
                trajectoryStatus = 'available';
            end
        elseif any(failed)
            % Inf has no drawable location on a logarithmic axis. The failure
            % is shown explicitly as a non-numeric annotation below.
            loglog(axTrajectory,NaN,NaN,'x','LineStyle','none','Color',thisColor, ...
                'LineWidth',1.2,'MarkerSize',7, ...
                'DisplayName',[displayName ' (failed)']);
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
        plotData.methods(i).trajectoryNRMSE = yTrajectory;
        plotData.methods(i).trajectoryStatus = trajectoryStatus;
    end
    set(axTrajectory,'XScale','log','YScale','log');
    grid(axTrajectory,'on');
    xlabel(axTrajectory,'Training samples');
    ylabel(axTrajectory,'Trajectory NRMSE');
    title(axTrajectory,'Short-horizon unseen-initial-condition rollout');
    add_rollout_failure_annotations_local(axTrajectory,rolloutFailures);
    legend(axTrajectory,'Location','best','Interpreter','none');
    positiveN = unique([rows.nTrain]);
    positiveN = positiveN(isfinite(positiveN) & positiveN>0);
    if ~isempty(positiveN)
        if numel(positiveN)==1
            xlim(axDerivative,positiveN(1)*[0.8 1.25]);
            xlim(axTrajectory,positiveN(1)*[0.8 1.25]);
        else
            xlim(axDerivative,[min(positiveN) max(positiveN)]);
            xlim(axTrajectory,[min(positiveN) max(positiveN)]);
        end
    end
    hasFiniteTrajectory = false;
    for iMethod = 1:numel(plotData.methods)
        values = plotData.methods(iMethod).trajectoryNRMSE;
        if any(isfinite(values) & values>0)
            hasFiniteTrajectory = true;
            break;
        end
    end
    if ~hasFiniteTrajectory
        text(axTrajectory,0.5,0.5, ...
            'No finite rollout metric is currently available.', ...
            'Units','normalized','HorizontalAlignment','center', ...
            'VerticalAlignment','middle','Interpreter','none');
    end

    plotData.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    set(fig,'Visible','on');
    figure(fig);
    drawnow;
end

function [methodColors,methodLineStyles,methodMarkers,methodMarkerFaceModes] = build_method_styles_local(methodOrder)
%BUILD_METHOD_STYLES_LOCAL Shared paper marker semantics for SI replay figures.
    nMethods = numel(methodOrder);
    groupIndex = zeros(1,nMethods);
    isGrouped = false(1,nMethods);
    isSR = false(1,nMethods);
    for i = 1:nMethods
        name = char(methodOrder{i});
        token = regexpi(name,'^(?:PhDN-G|Stage0-SR-G)([123])$','tokens','once');
        if ~isempty(token)
            groupIndex(i) = str2double(token{1});
            isGrouped(i) = true;
        end
        isSR(i) = startsWith(name,'Stage0-SR-','IgnoreCase',true);
    end
    palette = lines(max(3+sum(~isGrouped),3));
    methodColors = zeros(nMethods,3);
    methodLineStyles = repmat({'-'},1,nMethods);
    methodMarkers = repmat({'o'},1,nMethods);
    methodMarkerFaceModes = repmat({'open'},1,nMethods);
    groupMarkers = {'o','s','^'};
    nextOther = 4;
    for i = 1:nMethods
        if isGrouped(i)
            g = groupIndex(i);
            methodColors(i,:) = palette(g,:);
            methodMarkers{i} = groupMarkers{g};
            if isSR(i)
                methodLineStyles{i} = '--';
            else
                methodMarkerFaceModes{i} = 'filled';
            end
        else
            methodColors(i,:) = palette(nextOther,:);
            nextOther = nextOther + 1;
        end
    end
end

function add_rollout_failure_annotations_local(ax,failures)
%ADD_ROLLOUT_FAILURE_ANNOTATIONS_LOCAL Show failed methods without assigning
%an artificial finite NRMSE value on the logarithmic trajectory axis.
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
            % Preserve failed rounds instead of silently averaging only the
            % finite successful rounds.
            means(k) = Inf;
        elseif ~isempty(values)
            means(k) = mean(values);
        end
    end
end

function label = display_method_name_local(methodName)
    label = char(methodName);
    if strcmpi(label,'KAN-pruned') || strcmpi(label,'kan-pruned') || strcmpi(label,'kan')
        label = 'KAN';
    end
end
