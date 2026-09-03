%% Plot all SoftSaturatedLorenz96 rollout trajectories and save one selected PDF
% This helper script is intended to be run after the main demo has produced
% the variable `systemIdentificationRows` and the task `task`.
%
% It displays one trajectory figure for every available training sample size N.
% Only one selected N (default: the largest N) is exported to PDF.

assert(exist('task','var') == 1,'Variable "task" is not available in the workspace.');
assert(exist('systemIdentificationRows','var') == 1, ...
    'Variable "systemIdentificationRows" is not available in the workspace.');

availableN = unique([systemIdentificationRows.nTrain],'stable');
fprintf('Available N values for trajectory plotting: [%s]\n',num2str(availableN));

% Optional: manually choose which N to save to PDF.
SelectedNForPdf = max(availableN);

[figAllN,savedPdfPath] = plot_soft_saturated_lorenz96_trajectory_all_samples( ...
    task,systemIdentificationRows, ...
    'SampleList',availableN, ...
    'SaveSelectedNTrain',SelectedNForPdf, ...
    'SavePdf',true, ...
    'Visible','on');

disp(figAllN);
if ~isempty(savedPdfPath)
    fprintf('Trajectory PDF saved to:\n%s\n',savedPdfPath);
end
