%% Replot SoftSaturatedLorenz96 derivative-noise robustness from compact rows
% Fast current-grid replay only; no model checkpoint loading and no training.
% The global compact cache is first refreshed from each tiny per-cell
% noise_rows.mat file. This repairs stale/missing rho rows (for example rho=0)
% without scanning or loading method_results checkpoints.
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
RMSEFigureYLimits = [];

methods = {'PhDN-G1','PhDN-G2','PhDN-G3','MLP'};
if IncludeStage0SRAblationsInComparison
    methods = {'PhDN-G1','PhDN-G2','PhDN-G3', ...
        'Stage0-SR-G1','Stage0-SR-G2','Stage0-SR-G3','MLP'};
end

outputRoot = fullfile(projectRoot,'outputs', ...
    'SoftSaturatedLorenz96_K10_F8_kappa1_NoiseRobustness');
outputDir = fullfile(outputRoot,'summary');
compactFile = fullfile(outputDir,'l96_noise_rows.mat');

allRows = struct([]);
if exist(compactFile,'file') == 2
    payload = load(compactFile,'noiseRows');
    if isfield(payload,'noiseRows') && isstruct(payload.noiseRows)
        allRows = payload.noiseRows;
    end
end

% Refresh from the lightweight per-cell row files. These files are tiny and
% authoritative for already completed round/rho cells. Never load anything
% from method_results here.
numCellRowFilesLoaded = 0;
for iRound = 1:NumRounds
    roundKey = sprintf('round_%02d',iRound);
    for iNoise = 1:numel(NoiseLevels)
        rho = NoiseLevels(iNoise);
        noiseKey = soft_saturated_lorenz96_noise_key(rho);
        cellRowsPath = fullfile(outputRoot,roundKey,noiseKey,'noise_rows.mat');
        if exist(cellRowsPath,'file') ~= 2
            continue;
        end
        try
            cellPayload = load(cellRowsPath,'rows');
            if isfield(cellPayload,'rows') && isstruct(cellPayload.rows) && ...
                    ~isempty(cellPayload.rows)
                allRows = merge_soft_saturated_lorenz96_noise_rows( ...
                    allRows,cellPayload.rows);
                numCellRowFilesLoaded = numCellRowFilesLoaded+1;
            end
            clear cellPayload;
        catch ME
            fprintf('Could not read lightweight row file %s: %s\n', ...
                cellRowsPath,ME.message);
        end
    end
end

if isempty(allRows)
    error('No saved compact rows are available for this robustness case.');
end

% Keep the repaired global cache synchronized for future near-instant replots.
if numCellRowFilesLoaded > 0
    save_soft_saturated_lorenz96_noise_compact_rows(compactFile,allRows);
    fprintf('Compact row cache refreshed from %d lightweight per-cell row files.\n', ...
        numCellRowFilesLoaded);
end

rows = filter_soft_saturated_lorenz96_noise_rows( ...
    allRows,NoiseLevels,NumRounds,methods);
if isempty(rows)
    error('No current-grid rows are available after lightweight cache refresh.');
end

% Partial replay remains allowed: plot every available method/round/rho row.
[complete,report] = validate_soft_saturated_lorenz96_noise_rows( ...
    rows,NoiseLevels,NumRounds,{'PhDN-G1','PhDN-G2','PhDN-G3'});
if ~complete
    fprintf('Current-grid PhDN rows are incomplete; plotting all available rows only.\n');
    for k = 1:numel(report.missing)
        fprintf('  Missing: %s\n',report.missing{k});
    end
end

T = soft_saturated_lorenz96_noise_rows_to_table(rows);
writetable(T,fullfile(outputDir,'lorenz96_noise_robustness_all_runs.csv'));

[figNoise,figureData,summaryTable] = plot_soft_saturated_lorenz96_noise_robustness( ...
    rows,'Visible',true,'YLimits',RMSEFigureYLimits);

writetable(summaryTable,fullfile(outputDir,'lorenz96_noise_robustness_mean_std.csv'));
save(fullfile(outputDir,'lorenz96_noise_robustness_figure_data.mat'), ...
    'figureData','summaryTable','-v7');
savefig(figNoise,fullfile(outputDir,'lorenz96_noise_robustness.fig'));
exportgraphics(figNoise,fullfile(outputDir,'lorenz96_noise_robustness.pdf'), ...
    'ContentType','vector');

fprintf('Fast low-noise robustness replot finished.\n');
