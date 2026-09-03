function [status, cmdout, info] = run_external_process_guarded(executable, arguments, stdoutPath, matchToken, controlPath, adapterPidPath)
%RUN_EXTERNAL_PROCESS_GUARDED Run and clean one MATLAB-owned PySR process.
%
% This v72c bugfix deliberately keeps cancellation simple and exact:
%   1) MATLAB starts one responsive supervisor.
%   2) The supervisor records the real PySR adapter PID.
%   3) If MATLAB Stop/Ctrl+C unwinds this function, onCleanup directly kills
%      the adapter PID tree and the supervisor PID tree.
%   4) A unique config.json command-line match is used as a final fallback.
%
% The onCleanup callback is a local subfunction called through an anonymous
% function that captures immutable values. It does not use a nested function
% sharing the exiting workspace. This avoids the verified MATLAB warning:
% "destroyed variable 'controlPath'".

    if nargin < 2 || isempty(arguments); arguments = {}; end
    if nargin < 3 || isempty(stdoutPath); stdoutPath = [tempname, '.log']; end
    if nargin < 4 || isempty(matchToken); matchToken = ''; end
    if nargin < 5 || isempty(controlPath); controlPath = ''; end
    if nargin < 6 || isempty(adapterPidPath); adapterPidPath = ''; end

    executableValue = char(executable);
    matchTokenValue = char(matchToken);
    controlPathValue = char(controlPath);
    adapterPidPathValue = char(adapterPidPath);
    stdoutPathValue = char(stdoutPath);

    outDir = fileparts(stdoutPathValue);
    if ~isempty(outDir) && ~exist(outDir, 'dir'); mkdir(outDir); end
    if exist(stdoutPathValue, 'file'); delete(stdoutPathValue); end

    cmdList = javaObject('java.util.ArrayList');
    cmdList.add(executableValue);
    for k = 1:numel(arguments)
        cmdList.add(char(arguments{k}));
    end

    builder = javaObject('java.lang.ProcessBuilder', cmdList);
    builder.redirectErrorStream(true);
    builder.redirectOutput(javaObject('java.io.File', stdoutPathValue));

    process = builder.start();
    supervisorPid = get_java_process_pid_local(process);
    info = struct('pid', supervisorPid, 'supervisorPid', supervisorPid, ...
        'adapterPid', NaN, 'stdoutPath', stdoutPathValue, ...
        'executable', executableValue, 'matchToken', matchTokenValue, ...
        'controlPath', controlPathValue, 'adapterPidPath', adapterPidPathValue);

    % Java handle state is retained independently of this function workspace.
    % The cleanup callback therefore knows whether this was a normal return.
    completedNormally = javaObject('java.util.concurrent.atomic.AtomicBoolean', false);

    % IMPORTANT: use a local subfunction with value-captured arguments. Do not
    % replace this with a nested cleanup function, which caused the verified
    % destroyed-controlPath warning when MATLAB Stop unwound the workspace.
    cleanupGuard = onCleanup(@() cleanup_interrupted_run_local( ...
        process, supervisorPid, adapterPidPathValue, matchTokenValue, ...
        controlPathValue, completedNormally)); %#ok<NASGU>

    % Do not use MATLAB pause() for process polling.  In the Live Editor,
    % repeated short pause() calls can make the toolbar look idle/paused even
    % though the external Python process is still running.  Sleep outside the
    % MATLAB pause state, then briefly service desktop events so Stop/Ctrl+C
    % remains responsive and the normal running indication is preserved.
    while java_process_alive_local(process)
        desktop_responsive_busy_wait_local(0.20);
        info.adapterPid = read_pid_file_local(adapterPidPathValue);
    end

    status = double(process.exitValue());
    completedNormally.set(true);
    delete_control_file_local(controlPathValue);
    info.adapterPid = read_pid_file_local(adapterPidPathValue);

    if exist(stdoutPathValue, 'file')
        cmdout = fileread(stdoutPathValue);
    else
        cmdout = '';
    end
end


function desktop_responsive_busy_wait_local(waitSeconds)
%DESKTOP_RESPONSIVE_BUSY_WAIT_LOCAL Poll without entering MATLAB pause state.
%
% java.lang.Thread.sleep keeps this MATLAB invocation synchronously active,
% while the following drawnow services desktop events and interruption
% requests.  The fallback preserves compatibility with unusual no-JVM runs.
    waitSeconds = max(0, double(waitSeconds));
    waitMilliseconds = max(1, round(1000 * waitSeconds));
    try
        javaMethod('sleep', 'java.lang.Thread', int64(waitMilliseconds));
    catch
        % Headless/no-JVM fallback.  Desktop MATLAB normally uses the Java path.
        pause(waitSeconds);
    end
    try
        drawnow limitrate;
    catch
        drawnow;
    end
end

function cleanup_interrupted_run_local(process, supervisorPid, adapterPidPath, matchToken, controlPath, completedNormally)
%CLEANUP_INTERRUPTED_RUN_LOCAL Stop only the current PySR run on interruption.
    try
        if completedNormally.get()
            delete_control_file_local(controlPath);
            return;
        end
    catch
        % If state inspection itself fails during unwinding, treat it as an
        % interrupted run and perform the safe exact-run cleanup.
    end

    delete_control_file_local(controlPath);
    adapterPid = read_pid_file_with_short_wait_local(adapterPidPath, 0.8);

    fprintf(2, '\nMATLAB interruption detected: terminating this PySR run');
    if isfinite(adapterPid); fprintf(2, ' (adapter PID %d)', round(adapterPid)); end
    if isfinite(supervisorPid); fprintf(2, ' (supervisor PID %d)', round(supervisorPid)); end
    fprintf(2, '.\n');

    if ispc
        % Kill the actual computation first, then its supervisor.
        kill_windows_pid_tree_local(adapterPid);

        % Exact config-path fallback catches this run even if Windows changed
        % the parent/child relation. Both supervisor and adapter command lines
        % contain the unique config.json path.
        terminate_matching_python_local(matchToken);

        kill_windows_pid_tree_local(supervisorPid);

        % Java fallback for the process directly launched by MATLAB.
        try
            if java_process_alive_local(process); process.destroyForcibly(); end
        catch
            try; process.destroy(); catch; end
        end

        % One quick verification/retry. This is intentionally short so MATLAB
        % Stop returns promptly instead of waiting for the abandoned search.
        pause(0.25);
        if process_pid_alive_local(adapterPid) || process_pid_alive_local(supervisorPid)
            kill_windows_pid_tree_local(adapterPid);
            terminate_matching_python_local(matchToken);
            kill_windows_pid_tree_local(supervisorPid);
        end
    else
        kill_posix_pid_tree_local(adapterPid);
        kill_posix_pid_tree_local(supervisorPid);
        try
            if java_process_alive_local(process); process.destroyForcibly(); end
        catch
            try; process.destroy(); catch; end
        end
    end
end

function tf = java_process_alive_local(process)
    tf = false;
    try; tf = logical(process.isAlive()); catch; end
end

function pid = get_java_process_pid_local(process)
    pid = NaN;
    try
        pid = double(process.pid());
        return;
    catch
    end
    try
        token = regexp(char(process.toString()), 'pid[= ](\d+)', 'tokens', 'once');
        if ~isempty(token); pid = str2double(token{1}); end
    catch
    end
end

function pid = read_pid_file_with_short_wait_local(pathValue, waitSeconds)
    pid = read_pid_file_local(pathValue);
    if isfinite(pid); return; end
    t0 = tic;
    while toc(t0) < waitSeconds
        pause(0.05);
        pid = read_pid_file_local(pathValue);
        if isfinite(pid); return; end
    end
end

function pid = read_pid_file_local(pathValue)
    pid = NaN;
    if isempty(pathValue) || ~exist(pathValue, 'file'); return; end
    try
        txt = strtrim(fileread(pathValue));
        value = str2double(txt);
        if isfinite(value) && value > 0; pid = value; end
    catch
    end
end

function tf = process_pid_alive_local(pid)
    tf = false;
    if ~isfinite(pid) || pid <= 0; return; end
    if ispc
        [status, out] = system(sprintf('tasklist /FI "PID eq %d" /FO CSV /NH', round(pid)));
        tf = status == 0 && contains(out, sprintf('"%d"', round(pid)));
    else
        tf = system(sprintf('kill -0 %d >/dev/null 2>&1', round(pid))) == 0;
    end
end

function kill_windows_pid_tree_local(pid)
    if ~isfinite(pid) || pid <= 0; return; end
    system(sprintf('taskkill /PID %d /T /F >NUL 2>&1', round(pid)));
end

function kill_posix_pid_tree_local(pid)
    if ~isfinite(pid) || pid <= 0; return; end
    system(sprintf('pkill -TERM -P %d >/dev/null 2>&1', round(pid)));
    system(sprintf('kill -TERM %d >/dev/null 2>&1', round(pid)));
    pause(0.15);
    system(sprintf('pkill -KILL -P %d >/dev/null 2>&1', round(pid)));
    system(sprintf('kill -KILL %d >/dev/null 2>&1', round(pid)));
end

function delete_control_file_local(controlPath)
    if isempty(controlPath); return; end
    try
        if exist(controlPath, 'file'); delete(controlPath); end
    catch
    end
end

function terminate_matching_python_local(matchToken)
%TERMINATE_MATCHING_PYTHON_LOCAL Kill only Python processes for this config.
    if isempty(matchToken); return; end

    % PowerShell single-quoted literal escaping. The entire PowerShell script
    % is enclosed in CMD double quotes, so its pipes are not interpreted by
    % cmd.exe. Do not use backslash quote escaping here; CMD does not treat
    % backslash as a quote escape character.
    token = strrep(char(matchToken), '''', '''''');
    ps = ['$t=''', token, '''; ', ...
        'Get-CimInstance Win32_Process | ', ...
        'Where-Object { $_.Name -match ''^python(w)?\.exe$'' -and ', ...
        '$_.CommandLine -like (''*''+$t+''*'') } | ', ...
        'ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }'];
    cmd = ['powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass ', ...
        '-Command "', ps, '" >NUL 2>&1'];
    system(cmd);
end
