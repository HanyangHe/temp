%% Replot SingleGeneratorDynamic derivative-noise robustness from saved rows
% Fast replay only: no training, no rollout, no method checkpoint scanning/loading.
% clear; clc;

projectRoot = pwd;

if ~exist(fullfile(projectRoot,'system_identification'),'dir')
    try
        activeFile = matlab.desktop.editor.getActiveFilename;
        projectRoot = fileparts(activeFile);
    catch
        error('Cannot locate project root.');
    end
end

addpath(genpath(projectRoot));
addpath(fullfile(projectRoot,'system_identification'),'-begin');
rehash;

NoiseLevels = [0,0.001,0.005,0.01];
NumRounds = 3;
IncludeStage0SRAblationsInComparison = true;

% [] lets the plotting function determine the full y-axis range automatically.
% No manuscript-oriented y-axis truncation is applied.
RMSEFigureYLimits = [1.000e-12, 1.000e+00];

methods = {'PhDN-G1','PhDN-G2','PhDN-G3','MLP'};

if IncludeStage0SRAblationsInComparison
    methods = {'PhDN-G1','PhDN-G2','PhDN-G3', ...
        'Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3','MLP'};
end

outputDir = fullfile(projectRoot,'outputs', ...
    'SingleGeneratorDynamic_SMIB_AVR_NoiseRobustness','summary');

compactFile = fullfile(outputDir,'generator_noise_rows.mat');
resultsFile = fullfile(outputDir,'single_generator_dynamic_noise_robustness_results.mat');

rows = struct([]);

if exist(compactFile,'file') == 2
    payload = load(compactFile,'noiseRows');
    if isfield(payload,'noiseRows') && isstruct(payload.noiseRows)
        rows = payload.noiseRows;
    end
elseif exist(resultsFile,'file') == 2
    payload = load(resultsFile,'noiseRows');
    if isfield(payload,'noiseRows') && isstruct(payload.noiseRows)
        rows = payload.noiseRows;
    end
else
    error('No generator noise robustness row cache/result MAT was found.');
end

if ~isempty(rows) && isfield(rows,'noiseLevel')
    keepNoise = false(1,numel(rows));
    for iNoise = 1:numel(NoiseLevels)
        keepNoise = keepNoise | abs([rows.noiseLevel]-NoiseLevels(iNoise)) < 1e-14;
    end
    rows = rows(keepNoise);
end

if ~isempty(rows) && isfield(rows,'roundIndex')
    rows = rows([rows.roundIndex] >= 1 & [rows.roundIndex] <= NumRounds);
end

if ~isempty(rows) && isfield(rows,'method')
    rows = rows(ismember(lower(string({rows.method})),lower(string(methods))));
end

if isempty(rows)
    error('No rows are available for the current low-noise grid.');
end

% Partial-result plotting is intentional: any available method/noise/round rows
% are plotted; missing PhDN or baseline cells do not abort the replay.
T = single_generator_dynamic_noise_rows_to_table(rows);

writetable(T,fullfile(outputDir,'generator_noise_robustness_all_runs.csv'));

[figNoise,figureData,summaryTable] = plot_single_generator_dynamic_noise_robustness( ...
    rows,'Visible',true,'YLimits',RMSEFigureYLimits);

writetable(summaryTable,fullfile(outputDir,'generator_noise_robustness_mean_std.csv'));

save(fullfile(outputDir,'generator_noise_robustness_figure_data.mat'), ...
    'figureData','summaryTable','-v7');

savefig(figNoise,fullfile(outputDir,'generator_noise_robustness.fig'));

exportgraphics(figNoise,fullfile(outputDir,'generator_noise_robustness.pdf'), ...
    'ContentType','vector');

fprintf('Fast generator low-noise robustness replot finished.\n');
