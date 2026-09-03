function [fig,figureData,summaryTable] = plot_soft_saturated_lorenz96_noise_robustness(rows,varargin)
%PLOT_SOFT_SATURATED_LORENZ96_NOISE_ROBUSTNESS Lorenz--96 manuscript robustness figure.
%
% The dedicated derivative-noise ablation reports clean ID-test vector-field
% RMSE versus derivative-label noise. The primary curves are PhDN-G1/G2/G3
% and MLP; optional Stage0-SR-G1/G2/G3 curves are plotted when those rows are
% supplied by the caller. Curves show mean +/- one standard deviation over
% rounds. OOD and rollout metrics are intentionally excluded.

    p = inputParser;
    addParameter(p,'Visible',true,@(x)islogical(x)||isnumeric(x));
    % Leave YLimits empty for data-driven log-axis limits; pass [ymin ymax] to override.
    addParameter(p,'YLimits',[],@(x)isempty(x)||(isnumeric(x)&&numel(x)==2&&all(isfinite(x))&&x(1)>0&&x(2)>x(1)));
    parse(p,varargin{:});
    visible = logical(p.Results.Visible);
    yLimits = double(p.Results.YLimits(:).');

    if isempty(rows)
        error('Noise robustness rows are empty.');
    end
    required = {'noiseLevel','roundIndex','method','derivativeRMSE'};
    for k = 1:numel(required)
        if ~isfield(rows,required{k})
            error('Noise robustness row field %s is missing.',required{k});
        end
    end

    preferredMethods = {'PhDN-G1','PhDN-G2','PhDN-G3', ...
        'Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3','MLP'};
    availableMethods = unique(string({rows.method}),'stable');
    methods = strings(0,1);
    for k = 1:numel(preferredMethods)
        if any(strcmpi(availableMethods,preferredMethods{k}))
            methods(end+1,1) = string(preferredMethods{k}); %#ok<AGROW>
        end
    end
    if isempty(methods)
        methods = availableMethods(:);
    end
    levels = unique([rows.noiseLevel]);
    levels = sort(levels(:).');

    nMax = numel(methods)*numel(levels);
    methodCol = strings(nMax,1);
    noiseLevelCol = nan(nMax,1);
    noisePercentCol = nan(nMax,1);
    nRunsCol = nan(nMax,1);
    idMeanCol = nan(nMax,1);
    idStdCol = nan(nMax,1);
    rowCounter = 0;
    for iMethod = 1:numel(methods)
        for iLevel = 1:numel(levels)
            mask = strcmpi(string({rows.method}),methods(iMethod)) & ...
                abs([rows.noiseLevel]-levels(iLevel)) < 1e-14;
            selected = rows(mask);
            if isempty(selected); continue; end
            id = [selected.derivativeRMSE];
            rowCounter = rowCounter+1;
            methodCol(rowCounter) = methods(iMethod);
            noiseLevelCol(rowCounter) = levels(iLevel);
            noisePercentCol(rowCounter) = 100*levels(iLevel);
            nRunsCol(rowCounter) = sum(isfinite(id));
            idMeanCol(rowCounter) = finite_mean_local(id);
            idStdCol(rowCounter) = finite_std_local(id);
        end
    end
    methodCol = methodCol(1:rowCounter);
    noiseLevelCol = noiseLevelCol(1:rowCounter);
    noisePercentCol = noisePercentCol(1:rowCounter);
    nRunsCol = nRunsCol(1:rowCounter);
    idMeanCol = idMeanCol(1:rowCounter);
    idStdCol = idStdCol(1:rowCounter);
    summaryTable = table(methodCol,noiseLevelCol,noisePercentCol,nRunsCol, ...
        idMeanCol,idStdCol, ...
        'VariableNames',{'method','noiseLevel','noisePercent','nRuns', ...
        'idRMSEMean','idRMSEStd'});

    fig = figure('Visible',onoff_local(visible),'Color','w', ...
        'Name','Derivative-noise robustness');
    ax = axes(fig); hold(ax,'on'); box(ax,'on'); grid(ax,'on');

    % Use one color per physical-prior level. The corresponding Stage-0 SR
    % curve deliberately reuses the PhDN color and differs only by line style:
    % PhDN = solid, SR = dashed. MLP receives a separate fourth color.
    colorOrder = get(ax,'ColorOrder');
    if size(colorOrder,1) < 4
        colorOrder = lines(4);
    end
    priorColors = struct('G1',colorOrder(1,:), ...
                         'G2',colorOrder(2,:), ...
                         'G3',colorOrder(3,:));
    mlpColor = colorOrder(4,:);

    for iMethod = 1:numel(methods)
        methodName = char(methods(iMethod));
        mask = strcmpi(summaryTable.method,methods(iMethod));
        S = summaryTable(mask,:);
        [~,order] = sort(S.noiseLevel);
        S = S(order,:);
        % Keep the lower error-bar endpoint positive on the logarithmic axis.
        % The summary table still contains the unmodified mean/std values.
        lowerErr = min(S.idRMSEStd,0.95*S.idRMSEMean);
        upperErr = S.idRMSEStd;

        lineStyle = '-';
        markerStyle = 'o';
        markerFaceColor = 'none';
        markerSize = 5;
        lineWidth = 1.4;
        if contains(methodName,'G1','IgnoreCase',true)
            markerStyle = 'o';
        elseif contains(methodName,'G2','IgnoreCase',true)
            markerStyle = 's';
        elseif contains(methodName,'G3','IgnoreCase',true)
            markerStyle = '^';
        end
        if startsWith(methodName,'Stage0-SR-','IgnoreCase',true)
            lineStyle = '--';
            % Use an opaque white marker face rather than transparent 'none'.
            % When SR and PhDN have identical values, transparent SR markers let
            % the filled PhDN markers show through and the two curves look like
            % one solid curve. White-filled open markers keep the SR point visible.
            markerFaceColor = 'white';
            markerSize = 6;
            lineWidth = 1.6;
        elseif startsWith(methodName,'PhDN-G','IgnoreCase',true)
            markerFaceColor = 'auto'; % replaced by curveColor after color assignment below
        end

        if contains(methodName,'G1','IgnoreCase',true)
            curveColor = priorColors.G1;
        elseif contains(methodName,'G2','IgnoreCase',true)
            curveColor = priorColors.G2;
        elseif contains(methodName,'G3','IgnoreCase',true)
            curveColor = priorColors.G3;
        elseif strcmpi(methodName,'MLP')
            curveColor = mlpColor;
        else
            curveColor = colorOrder(mod(iMethod-1,size(colorOrder,1))+1,:);
        end

        if ischar(markerFaceColor) && strcmpi(markerFaceColor,'auto')
            markerFaceColor = curveColor;
        end

        % Keep stored/internal labels unchanged but shorten the paper legend.
        legendName = methodName;
        if startsWith(methodName,'Stage0-SR-','IgnoreCase',true)
            legendName = strrep(methodName,'Stage0-SR-','SR-');
        end

        errorbar(ax,S.noisePercent,S.idRMSEMean,lowerErr,upperErr, ...
            'LineStyle',lineStyle,'Marker',markerStyle,'Color',curveColor, ...
            'MarkerFaceColor',markerFaceColor, ...
            'LineWidth',lineWidth,'MarkerSize',markerSize,'DisplayName',legendName);
    end

    set(ax,'YScale','log');
    if isempty(yLimits)
        finiteY = summaryTable.idRMSEMean(isfinite(summaryTable.idRMSEMean) & summaryTable.idRMSEMean>0);
        if ~isempty(finiteY)
            yMin = 10^floor(log10(min(finiteY)));
            yMax = 10^ceil(log10(max(finiteY)));
            if yMax <= yMin; yMax = 10*yMin; end
            yLimits = [yMin,yMax];
            ylim(ax,yLimits);
        end
    else
        ylim(ax,yLimits);
    end
    xlabel(ax,'Derivative-noise level (%)');
    ylabel(ax,'Clean ID-test vector-field RMSE');
    title(ax,'Derivative-noise robustness');
    xticks(ax,100*levels);
    legend(ax,'Location','best');

    figureData = struct();
    figureData.noiseLevels = levels;
    figureData.methods = cellstr(methods);
    figureData.summaryTable = summaryTable;
    figureData.metric = 'clean_id_test_vector_field_rmse';
    figureData.yLimits = yLimits;
    figureData.yAxisAutoScaled = isempty(p.Results.YLimits);
    figureData.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
end

function value = finite_mean_local(x)
    x = x(isfinite(x));
    if isempty(x); value = NaN; else; value = mean(x); end
end
function value = finite_std_local(x)
    x = x(isfinite(x));
    if numel(x) <= 1; value = 0; else; value = std(x,0); end
end
function s = onoff_local(tf)
    if tf; s='on'; else; s='off'; end
end
