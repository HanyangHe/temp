function info = save_single_generator_dynamic_noise_run(outputDir,resultPack,task,plan,noiseProtocol,rows)
%SAVE_SINGLE_GENERATOR_DYNAMIC_NOISE_RUN Persist one noise-level/round checkpoint.
%
% The noise robustness experiment uses an output key that contains both the
% round and noise level, so it must not use the sample-efficiency merge logic
% keyed only by (round,nTrain,method).

    if exist(outputDir,'dir') ~= 7; mkdir(outputDir); end
    methodDir = fullfile(outputDir,'method_results');
    if exist(methodDir,'dir') ~= 7; mkdir(methodDir); end

    info = struct();
    info.outputDir = outputDir;
    info.contextPath = fullfile(outputDir,'noise_context.mat');
    info.rowsPath = fullfile(outputDir,'noise_rows.mat');
    info.metricsCsvPath = fullfile(outputDir,'metrics.csv');
    info.methodPaths = struct();

    context = struct();
    context.task = task;
    context.samplingPlan = plan;
    context.noiseProtocol = noiseProtocol;
    atomic_save_local(info.contextPath,'context',context);

    % Incremental checkpoint behavior: merge the current method rows with any
    % previously saved rows for this same round/noise folder. Existing trained
    % method files are never deleted merely because they are absent from the
    % current resultPack.
    previousRows = struct([]);
    if exist(info.rowsPath,'file') == 2
        previousPayload = load(info.rowsPath,'rows');
        if isfield(previousPayload,'rows') && isstruct(previousPayload.rows)
            previousRows = previousPayload.rows;
        end
    end
    rows = merge_rows_local(previousRows,rows);
    atomic_save_local(info.rowsPath,'rows',rows);

    names = fieldnames(resultPack);
    for k = 1:numel(names)
        fieldName = names{k};
        if ~isstruct(resultPack.(fieldName)); continue; end
        methodPath = fullfile(methodDir,[fieldName '.mat']);
        result = resultPack.(fieldName); %#ok<NASGU>
        save(methodPath,'result','-v7.3');
        info.methodPaths.(fieldName) = methodPath;
    end

    % Report every method checkpoint currently present, including methods
    % preserved from earlier incremental runs.
    savedFiles = dir(fullfile(methodDir,'*.mat'));
    for k = 1:numel(savedFiles)
        [~,savedField] = fileparts(savedFiles(k).name);
        if isvarname(savedField)
            info.methodPaths.(savedField) = fullfile(methodDir,savedFiles(k).name);
        end
    end

    T = single_generator_dynamic_noise_rows_to_table(rows);
    writetable(T,info.metricsCsvPath);
end

function merged = merge_rows_local(previousRows,newRows)
    if isempty(previousRows); merged = newRows; return; end
    if isempty(newRows); merged = previousRows; return; end
    merged = previousRows;
    for i = 1:numel(newRows)
        replaceMask = false(1,numel(merged));
        if isfield(newRows,'method') && isfield(merged,'method')
            replaceMask = strcmpi({merged.method},newRows(i).method);
            if isfield(newRows,'roundIndex') && isfield(merged,'roundIndex')
                replaceMask = replaceMask & [merged.roundIndex] == newRows(i).roundIndex;
            end
            if isfield(newRows,'noiseLevel') && isfield(merged,'noiseLevel')
                replaceMask = replaceMask & ...
                    abs([merged.noiseLevel]-newRows(i).noiseLevel) < 1e-14;
            end
        end
        merged = merged(~replaceMask);
        if isempty(merged)
            merged = newRows(i);
        else
            merged(end+1) = newRows(i); %#ok<AGROW>
        end
    end
end

function atomic_save_local(path,variableName,value)
    parent = fileparts(path);
    if exist(parent,'dir') ~= 7; mkdir(parent); end
    temporaryPath = [tempname(parent) '.mat'];
    payload = struct(); payload.(variableName) = value;
    save(temporaryPath,'-struct','payload','-v7.3');
    [ok,message] = movefile(temporaryPath,path,'f');
    if ~ok
        if exist(temporaryPath,'file') == 2; delete(temporaryPath); end
        error('Atomic save failed for %s: %s',path,message);
    end
end
