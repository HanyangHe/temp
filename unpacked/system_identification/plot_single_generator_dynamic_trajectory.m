function [fig, plotData] = plot_single_generator_dynamic_trajectory(task, rows, selectedNTrain)
%PLOT_SINGLE_GENERATOR_DYNAMIC_TRAJECTORY Mean rollout over every IC and round.
% For a fixed N_train, each method curve is the ordinary pointwise mean of
% every rollout trajectory from every available training round and every
% rollout initial condition. Failed trajectories are retained: their valid
% prefix participates in the mean and their unavailable suffix is NaN. The
% mean is deliberately computed without 'omitnan', so the first NaN in any
% member propagates to the aggregate curve and MATLAB automatically stops
% drawing the invalid suffix. The reference curve is the corresponding mean
% of the true trajectories over the same rollout IC set.

    if nargin < 3 || isempty(selectedNTrain)
        selectedNTrain = max([rows.nTrain]);
    end

    yPaddingFraction = 0.10;

    plotData = struct('selectedNTrain',selectedNTrain, ...
        'availableRounds',[],'initialConditionCount',0, ...
        'reference',struct(),'methods',struct([]), ...
        'aggregationDefinition', ...
        'ordinary_pointwise_mean_over_all_rollout_ICs_and_rounds_with_nan_propagation', ...
        'generatedAt','');

    idxRows = find([rows.nTrain] == selectedNTrain);
    if isempty(idxRows)
        warning('No rows found for Ntrain=%d.',selectedNTrain);
        fig = [];
        return;
    end

    availableRounds = unique([rows(idxRows).roundIndex],'stable');
    plotData.availableRounds = availableRounds;
    nRounds = numel(availableRounds);

    rolloutRows = [];
    nCommon = Inf;
    for ii = idxRows
        r = rows(ii);
        if ~rollout_available_local(r)
            continue;
        end
        rolloutRows(end+1) = ii; %#ok<AGROW>
        nCommon = min(nCommon,numel(r.rollout.trajectories));
    end

    if isempty(rolloutRows) || ~isfinite(nCommon) || nCommon < 1
        warning('No rollout collection is available for Ntrain=%d.',selectedNTrain);
        fig = [];
        return;
    end
    plotData.initialConditionCount = nCommon;

    % Build one complete true trajectory for every IC. The true system and IC
    % design are common to all methods, so any row containing a full reference
    % trajectory can supply that IC.
    referenceTime = [];
    referenceCube = [];
    initialConditions = nan(nCommon,task.nx);
    for kk = 1:nCommon
        foundReference = false;
        for ii = rolloutRows
            tr = rows(ii).rollout.trajectories(kk);
            if ~reference_trajectory_available_local(tr)
                continue;
            end
            if isempty(referenceTime)
                referenceTime = tr.t(:);
                nTime = numel(referenceTime);
                referenceCube = nan(nTime,task.nx,nCommon);
            end
            if numel(tr.t) ~= numel(referenceTime) || ...
                    any(abs(tr.t(:)-referenceTime) > 1e-12)
                continue;
            end
            referenceCube(:,:,kk) = tr.trueState;
            if isfield(tr,'initialCondition') && numel(tr.initialCondition) == task.nx
                initialConditions(kk,:) = reshape(tr.initialCondition,1,[]);
            end
            foundReference = true;
            break;
        end
        if ~foundReference
            warning('No complete reference trajectory is available for rollout IC %d/%d.',kk,nCommon);
            fig = [];
            return;
        end
    end
    referenceMean = mean(referenceCube,3);

    % Stable method order from the selected-N subset.
    methodNames = cell(1,numel(idxRows));
    for jj = 1:numel(idxRows)
        methodNames{jj} = char(rows(idxRows(jj)).method);
    end
    [~,firstMethodIdx] = unique(lower(string(methodNames)),'stable');
    orderedMethodNames = methodNames(sort(firstMethodIdx));
    nMethods = numel(orderedMethodNames);

    methodColors = lines(max(nMethods,1));
    methodTemplate = struct('internalName','','displayName','','status','', ...
        'available',false,'reason','','roundIndices',[], ...
        'color',zeros(1,3),'t',[],'predictedState',[], ...
        'rawRMSE',NaN,'normalizedRMSE',NaN, ...
        'successfulTrajectories',0,'dataTrajectories',0, ...
        'totalTrajectories',0,'failedTrajectories',0);
    methodData = repmat(methodTemplate,1,nMethods);

    unavailableNames = {};
    noDataNames = {};

    for jj = 1:nMethods
        thisName = orderedMethodNames{jj};
        displayName = display_method_name_local(thisName);
        methodData(jj).internalName = thisName;
        methodData(jj).displayName = displayName;
        methodData(jj).color = methodColors(jj,:);

        methodRowIdx = [];
        for ii = idxRows
            if strcmpi(char(rows(ii).method),thisName)
                methodRowIdx(end+1) = ii; %#ok<AGROW>
            end
        end
        if isempty(methodRowIdx)
            methodData(jj).status = 'unavailable';
            methodData(jj).reason = 'No rows found for this method and N.';
            unavailableNames{end+1} = displayName; %#ok<AGROW>
            continue;
        end

        methodData(jj).roundIndices = [rows(methodRowIdx).roundIndex];
        availableMethodRows = methodRowIdx(arrayfun(@(ii) rollout_available_local(rows(ii)),methodRowIdx));
        if isempty(availableMethodRows)
            methodData(jj).status = 'unavailable';
            methodData(jj).reason = 'Rollout predictor unavailable for all rounds.';
            unavailableNames{end+1} = displayName; %#ok<AGROW>
            continue;
        end
        methodData(jj).available = true;

        nExpected = numel(availableMethodRows)*nCommon;
        trajectoryCube = nan(numel(referenceTime),task.nx,nExpected);
        slot = 0;
        reasonList = {};
        successCount = 0;
        dataCount = 0;

        for rr = 1:numel(availableMethodRows)
            row = rows(availableMethodRows(rr));
            for kk = 1:nCommon
                slot = slot+1;
                if numel(row.rollout.trajectories) < kk
                    reasonList{end+1} = sprintf('Missing rollout IC %d.',kk); %#ok<AGROW>
                    continue;
                end
                tr = row.rollout.trajectories(kk);
                [alignedState,hasData] = align_predicted_trajectory_local( ...
                    tr,referenceTime,task.nx);
                trajectoryCube(:,:,slot) = alignedState;
                dataCount = dataCount + double(hasData);
                if trajectory_success_local(tr)
                    successCount = successCount+1;
                elseif isfield(tr,'message') && ~isempty(tr.message)
                    reasonList{end+1} = char(tr.message); %#ok<AGROW>
                end
            end
        end

        methodData(jj).totalTrajectories = nExpected;
        methodData(jj).successfulTrajectories = successCount;
        methodData(jj).dataTrajectories = dataCount;
        methodData(jj).failedTrajectories = nExpected-successCount;
        if ~isempty(reasonList)
            methodData(jj).reason = strjoin(unique(reasonList,'stable'),' | ');
        end

        if dataCount < 1
            methodData(jj).status = 'no_data';
            if isempty(methodData(jj).reason)
                methodData(jj).reason = 'No rollout trajectory data stored.';
            end
            noDataNames{end+1} = displayName; %#ok<AGROW>
            continue;
        end

        % Intentionally do NOT use 'omitnan'. A failed member therefore makes
        % the aggregate NaN from its failure time onward, exactly preserving
        % failure visibility in the plotted mean trajectory.
        meanState = mean(trajectoryCube,3);
        meanState = apply_state_guard_to_mean_local(meanState,task.rollout.maxStateAbs);
        methodData(jj).t = referenceTime;
        methodData(jj).predictedState = meanState;

        validMask = all(isfinite(meanState),2) & all(isfinite(referenceMean),2);
        if any(validMask)
            err = meanState(validMask,:) - referenceMean(validMask,:);
            scale = reshape(task.rollout.stateScale,1,[]);
            scale(~isfinite(scale) | scale <= 0) = 1;
            nerr = err ./ scale;
            methodData(jj).rawRMSE = sqrt(mean(err(:).^2));
            methodData(jj).normalizedRMSE = sqrt(mean(nerr(:).^2));
        end
        methodData(jj).status = 'mean';
    end

    if ~isempty(unavailableNames)
        warning('Ntrain=%d: rollout-unavailable methods: %s.', ...
            selectedNTrain,strjoin(unavailableNames,', '));
    end
    if ~isempty(noDataNames)
        warning('Ntrain=%d: methods with no stored rollout data: %s.', ...
            selectedNTrain,strjoin(noDataNames,', '));
    end

    fig = figure('Name',sprintf('SingleGeneratorDynamic rollout mean over all ICs N=%d',selectedNTrain), ...
        'Color','w','Visible','on');
    tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
    labels = {'\delta [rad]','\Delta\omega [p.u.]','E''_q [p.u.]','E_{fd} [p.u.]'};

    for stateIndex = 1:task.nx
        ax = nexttile;
        hold(ax,'on');
        plot(ax,referenceTime,referenceMean(:,stateIndex),'k-', ...
            'LineWidth',1.8,'DisplayName','Reference');

        for jj = 1:nMethods
            thisColor = methodData(jj).color;
            legendName = methodData(jj).displayName;
            if strcmp(methodData(jj).status,'mean') && ~isempty(methodData(jj).predictedState)
                plot(ax,methodData(jj).t,methodData(jj).predictedState(:,stateIndex),'--', ...
                    'Color',thisColor,'LineWidth',1.2,'DisplayName',legendName);
            elseif stateIndex == 1 && strcmp(methodData(jj).status,'no_data')
                plot(ax,NaN,NaN,'x','LineStyle','none','Color',thisColor, ...
                    'LineWidth',1.2,'MarkerSize',7,'DisplayName',legendName);
            elseif stateIndex == 1 && strcmp(methodData(jj).status,'unavailable')
                plot(ax,NaN,NaN,'s','LineStyle','none','Color',thisColor, ...
                    'LineWidth',1.0,'MarkerSize',6,'DisplayName',legendName);
            end
        end

        grid(ax,'on');
        xlabel(ax,'Time [s]');
        ylabel(ax,labels{stateIndex});
        apply_reference_axis_limits_local(ax,referenceMean(:,stateIndex),yPaddingFraction);
        if stateIndex == 1
            legend(ax,'Location','best','Interpreter','none');
        end
    end

    sgtitle(sprintf(['Salient-pole SMIB--linear-AVR hard joint-OOD rollout ', ...
        '(ordinary mean over %d ICs and %d round(s); NaN-propagating failure aggregation), ', ...
        'N_{train}=%d'],nCommon,nRounds,selectedNTrain));

    plotData.reference = struct('t',referenceTime, ...
        'trueState',referenceMean, ...
        'initialConditions',initialConditions);
    plotData.methods = methodData;
    plotData.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    set(fig,'Visible','on');
    figure(fig);
    drawnow;
end

function tf = rollout_available_local(row)
    tf = isstruct(row) && isfield(row,'rolloutAvailable') && logical(row.rolloutAvailable) && ...
        isfield(row,'rollout') && isstruct(row.rollout) && ...
        isfield(row.rollout,'trajectories') && ~isempty(row.rollout.trajectories);
end

function tf = reference_trajectory_available_local(tr)
    tf = isstruct(tr) && isfield(tr,'t') && ~isempty(tr.t) && ...
        isfield(tr,'trueState') && ~isempty(tr.trueState) && ...
        all(isfinite(tr.t(:))) && all(isfinite(tr.trueState(:)));
    if tf
        tf = size(tr.trueState,1) == numel(tr.t);
    end
end

function [X,hasData] = align_predicted_trajectory_local(tr,referenceTime,nState)
    nTime = numel(referenceTime);
    X = nan(nTime,nState);
    hasData = false;
    if ~isstruct(tr) || ~isfield(tr,'predictedState') || isempty(tr.predictedState)
        return;
    end
    P = tr.predictedState;
    if size(P,2) ~= nState
        return;
    end
    nKeep = min(size(P,1),nTime);
    if isfield(tr,'t') && ~isempty(tr.t)
        nKeep = min(nKeep,numel(tr.t));
    end
    if nKeep < 1
        return;
    end
    P = P(1:nKeep,:);
    P(isinf(P)) = NaN;
    X(1:nKeep,:) = P;
    hasData = any(isfinite(P(:)));
end

function X = apply_state_guard_to_mean_local(X,maxStateAbs)
    if isempty(X)
        return;
    end
    lim = reshape(maxStateAbs,1,[]);
    if isscalar(lim)
        lim = repmat(lim,1,size(X,2));
    end
    if numel(lim) ~= size(X,2)
        error('task.rollout.maxStateAbs must be scalar or have one entry per state.');
    end
    for j = 1:size(X,2)
        invalid = ~isfinite(X(:,j)) | abs(X(:,j)) > lim(j);
        firstInvalid = find(invalid,1,'first');
        if ~isempty(firstInvalid)
            X(firstInvalid:end,j) = NaN;
        end
    end
end

function apply_reference_axis_limits_local(ax,referenceSeries,paddingFraction)
    y = referenceSeries(:);
    y = y(isfinite(y));
    if isempty(y)
        return;
    end
    yMin = min(y);
    yMax = max(y);
    ySpan = yMax-yMin;
    if ~(isfinite(ySpan) && ySpan > 0)
        refMag = max(abs([yMin,yMax,1]));
        ySpan = 0.1*refMag;
    end
    pad = max(1e-9,paddingFraction*ySpan);
    ylim(ax,[yMin-pad,yMax+pad]);
end

function tf = trajectory_success_local(tr)
    tf = isstruct(tr) && isfield(tr,'success') && logical(tr.success);
end

function label = display_method_name_local(methodName)
    label = char(methodName);
    if strcmpi(label,'KAN-pruned') || strcmpi(label,'kan-pruned') || strcmpi(label,'kan')
        label = 'KAN';
    end
end
