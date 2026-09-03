function methodPath = save_soft_saturated_lorenz96_noise_method_checkpoint(outputDir,methodField,result)
%SAVE_SOFT_SATURATED_LORENZ96_NOISE_METHOD_CHECKPOINT Crash-safe per-method save.
%
% Uses a short temporary filename in the destination directory.  This avoids
% the Windows legacy MAX_PATH failure caused by tempname(parent), whose UUID
% basename can push a deeply nested robustness path to 260 characters.

    if nargin < 3 || ~isstruct(result) || isempty(result)
        error('A nonempty method result structure is required.');
    end
    methodField = char(methodField);
    if ~isvarname(methodField)
        error('Invalid method checkpoint field name: %s',methodField);
    end
    if exist(outputDir,'dir') ~= 7; mkdir(outputDir); end
    methodDir = fullfile(outputDir,'method_results');
    if exist(methodDir,'dir') ~= 7; mkdir(methodDir); end
    methodPath = fullfile(methodDir,[methodField '.mat']);
    atomic_save_result_local(methodPath,result);
end

function atomic_save_result_local(path,result)
    parent = fileparts(path);
    if exist(parent,'dir') ~= 7; mkdir(parent); end
    temporaryPath = short_temporary_path_local(parent);
    cleanupObj = onCleanup(@() cleanup_temp_local(temporaryPath)); %#ok<NASGU>
    save(temporaryPath,'result','-v7.3');
    [ok,message] = movefile(temporaryPath,path,'f');
    if ~ok
        error('Atomic method save failed for %s: %s',path,message);
    end
end

function temporaryPath = short_temporary_path_local(parent)
    % tempname(parent) may generate a ~40-character UUID basename.  On the
    % user's deep Windows project path that produced an exact 260-character
    % HDF5 filename.  Retain tempname entropy but keep only a short token.
    for attempt = 1:20
        [~,rawToken] = fileparts(tempname);
        nToken = numel(rawToken);
        tokenLength = min(12,nToken);
        token = rawToken(max(1,nToken-tokenLength+1):nToken);
        temporaryPath = fullfile(parent,['t' token '.mat']);
        if exist(temporaryPath,'file') ~= 2
            return;
        end
    end
    error('Could not allocate a short temporary MAT-file name under %s.',parent);
end

function cleanup_temp_local(path)
    if exist(path,'file') == 2
        try delete(path); catch, end %#ok<CTCH>
    end
end
