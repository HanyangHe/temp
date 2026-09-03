%% Replot paper figures for SoftSaturatedLorenz96 from saved results only
% This root-level convenience script does not retrain or rerun rollouts.
% It only reloads the previously saved aggregate summary MAT-file and
% redraws the manuscript figures as vector PDFs.

% clear; clc;

projectRoot = pwd;
if ~exist(fullfile(projectRoot,'core'),'dir') || ~exist(fullfile(projectRoot,'tasks'),'dir')
    try
        activeFile = matlab.desktop.editor.getActiveFilename;
        projectRoot = fileparts(activeFile);
    catch
        error('Cannot locate the project root. Set the current folder to the framework root.');
    end
end
addpath(genpath(projectRoot));
addpath(fullfile(projectRoot,'system_identification'),'-begin');
rehash;

LorenzDimension = 10;
SaturationKappa = 1;
LorenzForcing = 8;
caseToRun = sprintf('SS_L96_K%d',LorenzDimension);
caseDefinitionTask = task_soft_saturated_lorenz96( ...
    caseToRun,'general',LorenzForcing,SaturationKappa);

OutputCaseRoot = fullfile(projectRoot,'outputs',caseDefinitionTask.name);
OutputSummaryDir = fullfile(OutputCaseRoot,'summary');
SourceResultsFile = fullfile(OutputSummaryDir,'public_summary.mat');
LegacyResultsFile = fullfile(OutputSummaryDir,'soft_saturated_lorenz96_results.mat');
if exist(SourceResultsFile,'file') ~= 2 && exist(LegacyResultsFile,'file') == 2
    migrate_system_identification_summary_to_public( ...
        LegacyResultsFile,SourceResultsFile,'DeleteLegacy',true);
end

paperFigureInfo = regenerate_soft_saturated_lorenz96_paper_figures( ...
    SourceResultsFile,OutputSummaryDir,'Visible',true,'ExportPDF',true);

fprintf('\nSoftSaturatedLorenz96 plot-only regeneration finished.\n');
fprintf('Source summary MAT: %s\n',paperFigureInfo.sourceResultsFile);
fprintf('Export directory   : %s\n',paperFigureInfo.exportDir);
for iPaperFigure = 1:numel(paperFigureInfo.figures)
    fprintf('Generated figure %d: %s\n',iPaperFigure, ...
        paperFigureInfo.figures(iPaperFigure).pdfPath);
end
