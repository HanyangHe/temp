function outputInfo = migrate_system_identification_summary_to_public( ...
    legacyResultsPath,publicSummaryPath,varargin)
%MIGRATE_SYSTEM_IDENTIFICATION_SUMMARY_TO_PUBLIC Extract a slim public summary.
%
% This utility is intentionally conservative: it reads only the small fields
% required for paper-table/figure regeneration from the legacy aggregate MAT
% file and never loads the very large allResults variable.  The legacy file is
% deleted only after the new public summary has been written and verified.
%
% Name-value option:
%   'DeleteLegacy' : delete legacyResultsPath after successful verification
%                    (default false).

    parser = inputParser;
    parser.addRequired('legacyResultsPath',@(x)ischar(x)||isstring(x));
    parser.addRequired('publicSummaryPath',@(x)ischar(x)||isstring(x));
    parser.addParameter('DeleteLegacy',false,@(x)islogical(x)||isnumeric(x));
    parser.parse(legacyResultsPath,publicSummaryPath,varargin{:});

    legacyResultsPath = char(parser.Results.legacyResultsPath);
    publicSummaryPath = char(parser.Results.publicSummaryPath);
    deleteLegacy = logical(parser.Results.DeleteLegacy);

    if exist(legacyResultsPath,'file') ~= 2
        error('Legacy summary MAT-file was not found: %s',legacyResultsPath);
    end

    parent = fileparts(publicSummaryPath);
    if exist(parent,'dir') ~= 7; mkdir(parent); end

    requiredNames = {'systemIdentificationRows','standardSummaryRows', ...
        'sampleEfficiencyTable','runMetadata'};
    fileInfo = whos('-file',legacyResultsPath);
    availableNames = {fileInfo.name};
    namesToLoad = intersect(requiredNames,availableNames,'stable');

    payload = struct();
    payload.systemIdentificationRows = struct([]);
    payload.standardSummaryRows = struct([]);
    payload.sampleEfficiencyTable = table();
    payload.runMetadata = struct();

    if ~isempty(namesToLoad)
        % Selective load is critical here: do NOT load allResults.
        loaded = load(legacyResultsPath,namesToLoad{:});
        for iName = 1:numel(namesToLoad)
            name = namesToLoad{iName};
            payload.(name) = loaded.(name);
        end
    end

    if isempty(payload.systemIdentificationRows)
        error(['Legacy summary does not contain usable systemIdentificationRows. ', ...
            'The legacy file has been left untouched: %s'],legacyResultsPath);
    end

    temporaryPath = [tempname(parent) '.mat'];
    cleanupObj = onCleanup(@()cleanup_temp_local(temporaryPath)); %#ok<NASGU>
    save(temporaryPath,'-struct','payload','-v7.3');
    [ok,message] = movefile(temporaryPath,publicSummaryPath,'f');
    if ~ok
        error('Could not create public summary %s: %s',publicSummaryPath,message);
    end

    % Verify both schema and the key plotting rows before touching the source.
    verifyInfo = whos('-file',publicSummaryPath);
    verifyNames = {verifyInfo.name};
    missingNames = setdiff(requiredNames,verifyNames);
    if ~isempty(missingNames)
        error(['Public summary verification failed; missing variable(s): %s. ', ...
            'Legacy file was NOT deleted.'],strjoin(missingNames,', '));
    end
    verifyRows = load(publicSummaryPath,'systemIdentificationRows');
    if ~isfield(verifyRows,'systemIdentificationRows') || ...
            isempty(verifyRows.systemIdentificationRows)
        error(['Public summary verification failed: systemIdentificationRows ', ...
            'is empty. Legacy file was NOT deleted.']);
    end

    legacyDeleted = false;
    if deleteLegacy
        delete(legacyResultsPath);
        legacyDeleted = exist(legacyResultsPath,'file') ~= 2;
        if ~legacyDeleted
            error(['Public summary was created successfully, but MATLAB could ', ...
                'not delete the legacy file: %s'],legacyResultsPath);
        end
    end

    outputInfo = struct();
    outputInfo.legacyResultsPath = legacyResultsPath;
    outputInfo.publicSummaryPath = publicSummaryPath;
    outputInfo.loadedVariables = namesToLoad;
    outputInfo.legacyDeleted = legacyDeleted;

    fprintf('Created slim public summary:\n%s\n',publicSummaryPath);
    if deleteLegacy
        fprintf('Verified public summary and deleted legacy aggregate:\n%s\n', ...
            legacyResultsPath);
    end
end

function cleanup_temp_local(path)
    if exist(path,'file') == 2
        try; delete(path); catch; end %#ok<CTCH>
    end
end
