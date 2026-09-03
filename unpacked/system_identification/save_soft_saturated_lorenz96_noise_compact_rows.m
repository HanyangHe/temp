function save_soft_saturated_lorenz96_noise_compact_rows(filePath,noiseRows,rebuildInfo)
%SAVE_SOFT_SATURATED_LORENZ96_NOISE_COMPACT_ROWS
% Atomically cache the small Lorenz--96 robustness row index.
%
% The cache intentionally uses MATLAB v7 (not v7.3/HDF5) because it contains
% compact scalar rows only. A short temporary basename is used to stay safely
% below the legacy Windows MAX_PATH limit in deep project directories.

if nargin < 2
    error('filePath and noiseRows are required.');
end
if nargin < 3
    rebuildInfo = [];
end

cacheDir = fileparts(filePath);
if exist(cacheDir,'dir') ~= 7
    mkdir(cacheDir);
end

tmpPath = fullfile(cacheDir,'l96c_tmp.mat');
if exist(tmpPath,'file') == 2
    delete(tmpPath);
end

cleanupObj = onCleanup(@()cleanup_temp_file(tmpPath)); %#ok<NASGU>
if isempty(rebuildInfo)
    save(tmpPath,'noiseRows','-v7');
else
    save(tmpPath,'noiseRows','rebuildInfo','-v7');
end

% Validate the temporary cache before replacing any existing sidecar.
probe = load(tmpPath,'noiseRows');
if ~isfield(probe,'noiseRows') || ~isstruct(probe.noiseRows)
    error('Compact-row cache validation failed for temporary file: %s',tmpPath);
end
clear probe;

% Replace only after a complete, readable temporary file exists.
if exist(filePath,'file') == 2
    delete(filePath);
end
[ok,msg] = movefile(tmpPath,filePath,'f');
if ~ok
    error('Could not install compact-row cache at %s: %s',filePath,msg);
end
end

function cleanup_temp_file(tmpPath)
if exist(tmpPath,'file') == 2
    try
        delete(tmpPath);
    catch
    end
end
end
