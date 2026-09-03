function [figs, savedPdfPath] = ...
    plot_single_generator_dynamic_trajectory_all_samples(task, rows, varargin)
%PLOT_SINGLE_GENERATOR_DYNAMIC_TRAJECTORY_ALL_SAMPLES
% Display one rollout trajectory figure for every available training sample
% size N, where each method curve is the mean over successful rounds for the
% selected rollout IC. Only one selected N is exported to PDF by default.
%
% Usage:
%   [figs,savedPdfPath] = plot_single_generator_dynamic_trajectory_all_samples( ...
%       task,systemIdentificationRows);
%
%   [figs,savedPdfPath] = plot_single_generator_dynamic_trajectory_all_samples( ...
%       task,systemIdentificationRows, ...
%       'SampleList',[500 1000], ...
%       'SaveSelectedNTrain',1000, ...
%       'SavePdf',true, ...
%       'SavePdfPath',fullfile(pwd,'single_generator_dynamic_rollout_selected.pdf'));
%
% Name-value options:
%   'SampleList'         : explicit list of N to display.
%   'SaveSelectedNTrain' : only this N is exported to PDF. Default=max(N).
%   'SavePdf'            : true/false, default=true.
%   'SavePdfPath'        : explicit PDF path. Default is under pwd.
%   'Visible'            : 'on' or 'off' for generated figures. Default='on'.
%
% This helper does not alter metrics or model-selection logic; it only
% automates figure generation for multiple N values.

    parser = inputParser;
    parser.addParameter('SampleList',[],@(x) isnumeric(x) && isvector(x));
    parser.addParameter('SaveSelectedNTrain',[],@(x) isempty(x) || (isscalar(x) && isnumeric(x)));
    parser.addParameter('SavePdf',true,@(x) islogical(x) || isnumeric(x));
    parser.addParameter('SavePdfPath','',@(x) ischar(x) || isstring(x));
    parser.addParameter('Visible','on',@(x) any(strcmpi(char(x),{'on','off'})));
    parser.parse(varargin{:});
    opt = parser.Results;

    if isempty(rows)
        warning('Empty system-identification row collection.');
        figs = gobjects(0);
        savedPdfPath = '';
        return;
    end

    availableN = unique([rows.nTrain],'stable');
    if isempty(opt.SampleList)
        sampleList = availableN;
    else
        sampleList = reshape(opt.SampleList,1,[]);
        sampleList = sampleList(ismember(sampleList,availableN));
    end

    if isempty(sampleList)
        warning('No requested Ntrain values are available.');
        figs = gobjects(0);
        savedPdfPath = '';
        return;
    end

    if isempty(opt.SaveSelectedNTrain)
        saveSelectedNTrain = max(sampleList);
    else
        saveSelectedNTrain = opt.SaveSelectedNTrain;
    end

    figs = gobjects(1,numel(sampleList));
    for k = 1:numel(sampleList)
        currentN = sampleList(k);
        fig = plot_single_generator_dynamic_trajectory(task,rows,currentN);

        if isgraphics(fig)
            set(fig,'Visible',char(opt.Visible));
            set(fig,'Name',sprintf('SingleGeneratorDynamic rollout N=%d',currentN));
            figs(k) = fig;
        else
            figs(k) = gobjects(1);
        end
    end

    savedPdfPath = '';
    if logical(opt.SavePdf)
        if ~ismember(saveSelectedNTrain,sampleList)
            warning('Requested SaveSelectedNTrain=%d is not in the displayed sample list. No PDF saved.', ...
                saveSelectedNTrain);
            return;
        end

        matchIndex = find(sampleList == saveSelectedNTrain,1,'first');
        figSave = figs(matchIndex);
        if ~isgraphics(figSave)
            warning('Selected Ntrain=%d figure is not valid. No PDF saved.',saveSelectedNTrain);
            return;
        end

        if strlength(string(opt.SavePdfPath)) == 0
            savedPdfPath = fullfile(pwd, ...
                sprintf('single_generator_dynamic_rollout_N_%05d.pdf',saveSelectedNTrain));
        else
            savedPdfPath = char(opt.SavePdfPath);
        end

        local_save_pdf(figSave,savedPdfPath);
        fprintf('Saved selected rollout PDF: %s\n',savedPdfPath);
    end
end

function local_save_pdf(fig,pdfPath)
%LOCAL_SAVE_PDF Robust PDF export with fallback for older MATLAB versions.

    outDir = fileparts(pdfPath);
    if ~isempty(outDir) && exist(outDir,'dir') ~= 7
        mkdir(outDir);
    end

    try
        exportgraphics(fig,pdfPath,'ContentType','vector');
    catch
        set(fig,'PaperPositionMode','auto');
        print(fig,pdfPath,'-dpdf','-painters');
    end
end
