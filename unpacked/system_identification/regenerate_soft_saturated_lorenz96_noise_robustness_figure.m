function info = regenerate_soft_saturated_lorenz96_noise_robustness_figure(resultsFile,exportDir,varargin)
%REGENERATE_SOFT_SATURATED_LORENZ96_NOISE_ROBUSTNESS_FIGURE Recompute RMSE summary and replot saved results.
%
% No training or rollout is executed. The function reloads the saved per-run
% noiseRows, recomputes mean/std from derivativeRMSE, and regenerates the RMSE
% figure/CSV/figure-data artifacts.

    p=inputParser;
    addParameter(p,'Visible',true,@(x)islogical(x)||isnumeric(x));
    addParameter(p,'ExportPDF',true,@(x)islogical(x)||isnumeric(x));
    addParameter(p,'IncludeStage0SRAblations',false,@(x)islogical(x)||isnumeric(x));
    addParameter(p,'YLimits',[],@(x)isempty(x)||(isnumeric(x)&&numel(x)==2&&all(isfinite(x))&&x(1)>0&&x(2)>x(1)));
    parse(p,varargin{:});
    if nargin < 2 || isempty(exportDir); exportDir=fileparts(resultsFile); end
    if exist(exportDir,'dir')~=7; mkdir(exportDir); end

    S=load(resultsFile,'noiseRows');
    if ~isfield(S,'noiseRows'); error('noiseRows missing from %s.',resultsFile); end
    rows = S.noiseRows;
    if isempty(rows) || ~isfield(rows,'derivativeRMSE')
        error(['Saved noiseRows do not contain derivativeRMSE. Re-run the lightweight ', ...
            'recorded-result replay once; retraining is not required.']);
    end

    comparisonMethods = {'PhDN-G1','PhDN-G2','PhDN-G3','MLP'};
    if logical(p.Results.IncludeStage0SRAblations)
        comparisonMethods = { ...
            'PhDN-G1','PhDN-G2','PhDN-G3', ...
            'Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3','MLP'};
    end
    if ~isempty(rows) && isfield(rows,'method')
        keep = ismember(lower(string({rows.method})),lower(string(comparisonMethods)));
        rows = rows(keep);
    end

    [fig,figureData,summaryTable]=plot_soft_saturated_lorenz96_noise_robustness( ...
        rows,'Visible',logical(p.Results.Visible),'YLimits',p.Results.YLimits);
    figPath=fullfile(exportDir,'lorenz96_noise_robustness.fig');
    pdfPath=fullfile(exportDir,'lorenz96_noise_robustness.pdf');
    dataPath=fullfile(exportDir,'lorenz96_noise_robustness_figure_data.mat');
    csvPath=fullfile(exportDir,'lorenz96_noise_robustness_mean_std.csv');
    savefig(fig,figPath);
    if logical(p.Results.ExportPDF)
        try
            exportgraphics(fig,pdfPath,'ContentType','vector');
        catch
            set(fig,'PaperPositionMode','auto'); print(fig,pdfPath,'-dpdf','-painters');
        end
    else
        pdfPath='';
    end
    save(dataPath,'figureData','summaryTable','-v7.3');
    writetable(summaryTable,csvPath);
    info=struct('figure',fig,'figPath',figPath,'pdfPath',pdfPath, ...
        'dataPath',dataPath,'csvPath',csvPath,'sourceResultsFile',resultsFile, ...
        'metric','clean_id_test_vector_field_rmse');
end
