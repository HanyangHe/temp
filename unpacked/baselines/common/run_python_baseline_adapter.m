function [pyResult, predictions, timeStats, runInfo] = run_python_baseline_adapter(adapterPath, pythonExe, workRoot, cfg, data)
%RUN_PYTHON_BASELINE_ADAPTER Export a shared split and run one Python adapter.

    if nargin < 5
        error('adapterPath, pythonExe, workRoot, cfg, and data are required.');
    end
    if ~exist(adapterPath, 'file')
        error('Python baseline adapter not found: %s', adapterPath);
    end
    commonDir = fileparts(mfilename('fullpath'));
    supervisorPath = fullfile(commonDir, 'python_matlab_supervisor.py');
    if ~exist(supervisorPath, 'file')
        error('Python baseline supervisor not found: %s', supervisorPath);
    end
    if isempty(pythonExe); pythonExe = 'python'; end
    if isempty(workRoot); workRoot = fullfile(tempdir, 'phdnn_python_baselines'); end
    if ~exist(workRoot, 'dir'); mkdir(workRoot); end

    runId = sprintf('%s_%s_%06d', datestr(now, 'yyyymmdd_HHMMSSFFF'), ...
        char(java.util.UUID.randomUUID()), randi(999999));
    runId = regexprep(runId, '[^A-Za-z0-9_\-]', '_');
    workDir = fullfile(workRoot, runId);
    mkdir(workDir);

    cfg.work_dir = workDir;
    cfg.paths = struct();
    fields = {'Xtr','Ytr','Xval','Yval','Xte','Yte','Xood','Yood'};
    names = {'X_train','Y_train','X_val','Y_val','X_test','Y_test','X_ood','Y_ood'};
    for k = 1:numel(fields)
        p = fullfile(workDir, [names{k}, '.csv']);
        cfg.paths.(names{k}) = p;
        if isfield(data, fields{k}) && ~isempty(data.(fields{k}))
            write_matrix_local(p, data.(fields{k}));
        else
            write_matrix_local(p, zeros(0, 0));
        end
    end
    configPath = fullfile(workDir, 'config.json');
    fid = fopen(configPath, 'w');
    if fid < 0; error('Cannot write baseline config: %s', configPath); end
    fwrite(fid, jsonencode(cfg, 'PrettyPrint', true), 'char'); fclose(fid);

    matlabPid = feature('getpid');
    controlPath = fullfile(workDir, 'matlab_run_alive.flag');
    adapterPidPath = fullfile(workDir, 'adapter_pid.txt');
    fid = fopen(controlPath, 'w');
    if fid >= 0
        fprintf(fid, 'MATLAB_PID=%d\nRUN_ID=%s\n', matlabPid, runId); fclose(fid);
    end
    args = {'--python', pythonExe, '--adapter', adapterPath, '--config', configPath, ...
        '--parent-pid', num2str(matlabPid), '--control-file', controlPath, ...
        '--adapter-pid-file', adapterPidPath};
    stdoutPath = fullfile(workDir, 'matlab_python_stdout.txt');
    tCall = tic;
    [status, cmdout, processInfo] = run_external_process_guarded( ...
        pythonExe, [{supervisorPath}, args], stdoutPath, configPath, controlPath, adapterPidPath);
    callTime = toc(tCall);
    if status ~= 0
        error(['Python baseline failed with status %d.\nAdapter: %s\n', ...
            'Adapter PID: %.0f\nOutput:\n%s\nWork dir: %s'], ...
            status, adapterPath, processInfo.adapterPid, cmdout, workDir);
    end

    resultPath = fullfile(workDir, 'result.json');
    if ~exist(resultPath, 'file')
        error('Python adapter did not create result.json. Work dir: %s\nOutput:\n%s', workDir, cmdout);
    end
    pyResult = jsondecode(fileread(resultPath));
    predictions = struct();
    predictions.train = read_matrix_local(fullfile(workDir, 'Yhat_train.csv'));
    predictions.val = read_matrix_local(fullfile(workDir, 'Yhat_val.csv'));
    predictions.test = read_matrix_local(fullfile(workDir, 'Yhat_test.csv'));
    predictions.ood = read_matrix_local(fullfile(workDir, 'Yhat_ood.csv'));
    timeStats = struct('pythonCallTime', callTime, ...
        'pyTotalTime', getfield_default_local(pyResult, 'total_time_seconds', NaN), ...
        'total', callTime);
    runInfo = struct('workDir', workDir, 'configPath', configPath, ...
        'resultJsonPath', resultPath, 'stdoutPath', stdoutPath, ...
        'adapterPid', processInfo.adapterPid, 'stdout', cmdout);
end

function write_matrix_local(path, A)
    if isempty(A)
        fid = fopen(path, 'w'); if fid >= 0; fclose(fid); end; return;
    end
    if exist('writematrix', 'file') == 2; writematrix(A, path); else; csvwrite(path, A); end
end

function A = read_matrix_local(path)
    if ~exist(path, 'file') || dir(path).bytes == 0; A = []; return; end
    if exist('readmatrix', 'file') == 2; A = readmatrix(path); else; A = csvread(path); end
    if isvector(A); A = A(:); end
end

function val = getfield_default_local(s, name, defaultVal)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name)); val = s.(name); else; val = defaultVal; end
end
