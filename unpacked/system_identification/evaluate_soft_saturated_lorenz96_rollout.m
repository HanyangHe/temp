function rollout = evaluate_soft_saturated_lorenz96_rollout(task, methodName, methodResult)
%EVALUATE_SOFT_SATURATED_LORENZ96_ROLLOUT Fixed-step ODE4 unseen-IC test.
%
% Both the exact and learned vector fields are integrated on the same fixed
% time grid by classical fourth-order Runge--Kutta (ODE4/RK4).  The fixed
% number of RHS evaluations prevents an unstable or pathological learned
% vector field from forcing an adaptive solver into arbitrarily small steps.

    [predictFcn, predictorInfo] = make_soft_saturated_lorenz96_predictor(methodName, methodResult);
    rollout = struct();
    rollout.method = methodName;
    rollout.available = predictorInfo.available;
    rollout.reason = predictorInfo.reason;
    % Primary trajectory metrics are the arithmetic means of the per-IC
    % trajectory RMSE values. The legacy pooled pointwise RMSE is retained
    % separately for diagnostics and backward comparison.
    rollout.rawRMSE = NaN;
    rollout.normalizedRMSE = NaN;
    rollout.pooledRawRMSE = NaN;
    rollout.pooledNormalizedRMSE = NaN;
    rollout.meanSuccessfulRawRMSE = NaN;
    rollout.meanSuccessfulNormalizedRMSE = NaN;
    rollout.perTrajectoryRMSE = nan(task.rollout.nInitialConditions,1);
    rollout.perTrajectoryNormalizedRMSE = nan(task.rollout.nInitialConditions,1);
    rollout.perStateRMSE = nan(1,task.nx);
    rollout.perStateNormalizedRMSE = nan(1,task.nx);
    rollout.aggregationDefinition = ...
        'arithmetic_mean_of_per_initial_condition_trajectory_RMSE';
    rollout.successCount = 0;
    rollout.totalCount = task.rollout.nInitialConditions;
    rollout.successRate = 0;
    rollout.solver = 'ode4';
    rollout.fixedStep = task.rollout.fixedStep;
    rollout.trajectories = struct([]);
    if ~predictorInfo.available
        return;
    end
    xBench = reshape(task.rollout.referenceInitialCondition, 1, []);

    % A compiled predictor is accepted only after a direct numerical check
    % against the original fixed-model predictor at the rollout IC. If the
    % compiled check errors or differs numerically, preserve correctness by
    % reverting to the original fixed inference path.
    if predictorInfo.compiled && isfield(predictorInfo, 'referencePredictFcn') && ...
            ~isempty(predictorInfo.referencePredictFcn)
        try
            yCompiled = predictFcn(xBench);
            yReference = predictorInfo.referencePredictFcn(xBench);
            delta = abs(yCompiled - yReference);
            maxAbsDifference = max(delta(:));
            scale = max(1, abs(yReference));
            relative = delta ./ scale;
            maxRelDifference = max(relative(:));
            predictorInfo.rolloutCheckMaxAbs = maxAbsDifference;
            predictorInfo.rolloutCheckMaxRel = maxRelDifference;
            if ~isfinite(maxAbsDifference) || ~isfinite(maxRelDifference) || ...
                    (maxAbsDifference > 1e-9 && maxRelDifference > 1e-8)
                warning('SoftSaturatedLorenz96:CompiledPredictorMismatch', ...
                    ['Compiled %s predictor differs from the original fixed model ', ...
                     '(max abs/rel %.6e / %.6e). Reverting to original inference.'], ...
                    methodName, maxAbsDifference, maxRelDifference);
                predictFcn = predictorInfo.referencePredictFcn;
                predictorInfo.compiled = false;
                predictorInfo.reason = sprintf(['Compiled predictor consistency ', ...
                    'check failed; original fixed inference is used ', ...
                    '(max abs/rel %.3e/%.3e).'], ...
                    maxAbsDifference, maxRelDifference);
            end
        catch ME
            warning('SoftSaturatedLorenz96:CompiledPredictorCheckFailed', ...
                ['Compiled %s predictor check raised an error: %s. ', ...
                 'Reverting to original fixed inference.'], methodName, ME.message);
            predictFcn = predictorInfo.referencePredictFcn;
            predictorInfo.compiled = false;
            predictorInfo.rolloutCheckError = ME.message;
            predictorInfo.reason = sprintf(['Compiled predictor check errored; ', ...
                'original fixed inference is used. Error: %s'], ME.message);
        end
    end

    % Do not retain a captured model function inside saved rollout results.
    if isfield(predictorInfo, 'referencePredictFcn')
        predictorInfo = rmfield(predictorInfo, 'referencePredictFcn');
    end
    rollout.predictorInfo = predictorInfo;

    benchmarkSucceeded = true;
    benchmarkMessage = '';
    try
        for kb = 1:3
            predictFcn(xBench);
        end
    catch ME
        benchmarkSucceeded = false;
        benchmarkMessage = ME.message;
    end

    nBench = 20;
    if benchmarkSucceeded
        benchTimer = tic;
        try
            for kb = 1:nBench
                predictFcn(xBench);
            end
            rollout.secondsPerRHS = toc(benchTimer) / nBench;
        catch ME
            rollout.secondsPerRHS = NaN;
            benchmarkMessage = ME.message;
        end
    else
        rollout.secondsPerRHS = NaN;
    end

    if isempty(benchmarkMessage)
        fprintf('[rollout predictor] %s | %.6g s/RHS | %s\n', ...
            methodName, rollout.secondsPerRHS, predictorInfo.reason);
    else
        fprintf('[rollout predictor] %s | benchmark unavailable (%s) | %s\n', ...
            methodName, benchmarkMessage, predictorInfo.reason);
    end

    tEval = (0:task.rollout.fixedStep:task.rollout.horizon).';
    if abs(tEval(end)-task.rollout.horizon) > 100*eps(max(1,task.rollout.horizon))
        error('Rollout horizon must be an integer multiple of fixedStep.');
    end

    initialConditions = make_initial_conditions_local(task);
    allRawSquared = zeros(0,task.nx);
    allNormSquared = zeros(0,task.nx);
    perIcRawRMSE = inf(size(initialConditions,1),1);
    perIcNormalizedRMSE = inf(size(initialConditions,1),1);
    perIcStateRMSE = inf(size(initialConditions,1),task.nx);
    perIcStateNormalizedRMSE = inf(size(initialConditions,1),task.nx);

    rowTemplate = struct('initialCondition',zeros(1,task.nx), ...
        'success',false,'message','','failureTime',NaN,'t',tEval, ...
        'trueState',[],'predictedState',[],'rawRMSE',Inf, ...
        'normalizedRMSE',Inf,'perStateRMSE',inf(1,task.nx), ...
        'perStateNormalizedRMSE',inf(1,task.nx), ...
        'trueRhsEvaluations',0, ...
        'predictedRhsEvaluations',0);
    trajectoryRows = repmat(rowTemplate,size(initialConditions,1),1);

    % Precompute every exact reference trajectory before evaluating the
    % learned model. A learned-model early abort must never remove the common
    % reference trajectories required by the report and trajectory figures.
    nIC = size(initialConditions,1);
    referenceStates = cell(nIC,1);
    referenceInfos = repmat(struct('success',true,'message','', ...
        'failureTime',NaN,'rhsEvaluations',0),nIC,1);
    for kRef = 1:nIC
        x0Ref = initialConditions(kRef,:);
        trajectoryRows(kRef).initialCondition = x0Ref;
        [xTrueRef,trueInfoRef] = ode4_fixed_local( ...
            @(x) true_rhs_local(x,task),tEval,x0Ref(:),task.rollout);
        referenceInfos(kRef) = trueInfoRef;
        trajectoryRows(kRef).trueRhsEvaluations = trueInfoRef.rhsEvaluations;
        trajectoryRows(kRef).t = tEval;
        if trueInfoRef.success
            referenceStates{kRef} = xTrueRef;
            trajectoryRows(kRef).trueState = xTrueRef;
        else
            trajectoryRows(kRef).message = sprintf( ...
                'Reference ODE4 rollout failed at t=%.6g s: %s', ...
                trueInfoRef.failureTime,trueInfoRef.message);
            trajectoryRows(kRef).failureTime = trueInfoRef.failureTime;
            rollout.available = false;
            rollout.reason = trajectoryRows(kRef).message;
            rollout.trajectories = trajectoryRows;
            return;
        end
    end

    methodTimer = tic;
    consecutiveFailures = 0;
    fprintf('\n[ODE4 rollout] method=%s | ICs=%d | T=%.3g s | dt=%.4g s\n', ...
        methodName,size(initialConditions,1),task.rollout.horizon,task.rollout.fixedStep);

    for k = 1:size(initialConditions,1)
        if toc(methodTimer) > task.rollout.maxWallTimePerMethod
            for kk = k:size(initialConditions,1)
                trajectoryRows(kk).initialCondition = initialConditions(kk,:);
                trajectoryRows(kk).message = sprintf( ...
                    'Skipped: method wall-time cap %.3g s exceeded.', ...
                    task.rollout.maxWallTimePerMethod);
                trajectoryRows(kk).failureTime = 0;
            end
            rollout.reason = sprintf('Method wall-time cap %.3g s exceeded.', ...
                task.rollout.maxWallTimePerMethod);
            fprintf('[ODE4 rollout] %s stopped before IC %d: method wall-time cap exceeded.\n', ...
                methodName,k);
            break;
        end
        x0 = initialConditions(k,:);
        row = rowTemplate;
        row.initialCondition = x0;
        if mod(k-1,max(1,task.rollout.progressEveryIC)) == 0
            fprintf('[ODE4 rollout] %s IC %d/%d start | x0=[%s]\n', ...
                methodName,k,size(initialConditions,1),num2str(x0,' %.5g'));
        end
        icTimer = tic;
        try
            xTrue = referenceStates{k};
            row.trueRhsEvaluations = referenceInfos(k).rhsEvaluations;
            row.t = tEval;
            row.trueState = xTrue;

            [xPred,predInfo] = ode4_fixed_local( ...
                @(x) learned_rhs_local(x,predictFcn,task.ny), ...
                tEval,x0(:),task.rollout);
            row.predictedRhsEvaluations = predInfo.rhsEvaluations;
            if ~predInfo.success
                % Preserve the complete reference trajectory and represent the
                % unavailable learned suffix explicitly by NaN.  This allows
                % trajectory plots to aggregate every rollout IC with ordinary
                % pointwise mean semantics: once any member has failed, the
                % aggregate becomes NaN from that time onward and MATLAB stops
                % drawing that part of the curve automatically.
                row.failureTime = predInfo.failureTime;
                row.t = tEval;
                row.trueState = xTrue;
                row.predictedState = nan(size(xTrue));
                nKeep = min(size(xPred,1),size(xTrue,1));
                if nKeep >= 1
                    row.predictedState(1:nKeep,:) = xPred(1:nKeep,:);
                end
                error('Learned ODE4 rollout failed at t=%.6g s: %s', ...
                    predInfo.failureTime,predInfo.message);
            end

            err = xPred-xTrue;
            scale = reshape(task.rollout.stateScale,1,[]);
            scale(~isfinite(scale) | scale<=0) = 1;
            normErr = err./scale;

            row.success = true;
            row.t = tEval;
            row.trueState = xTrue;
            row.predictedState = xPred;
            row.rawRMSE = sqrt(mean(err(:).^2));
            row.normalizedRMSE = sqrt(mean(normErr(:).^2));
            row.perStateRMSE = sqrt(mean(err.^2,1));
            row.perStateNormalizedRMSE = sqrt(mean(normErr.^2,1));
            perIcRawRMSE(k) = row.rawRMSE;
            perIcNormalizedRMSE(k) = row.normalizedRMSE;
            perIcStateRMSE(k,:) = row.perStateRMSE;
            perIcStateNormalizedRMSE(k,:) = row.perStateNormalizedRMSE;
            allRawSquared = [allRawSquared; err.^2]; %#ok<AGROW>
            allNormSquared = [allNormSquared; normErr.^2]; %#ok<AGROW>
            rollout.successCount = rollout.successCount+1;
            consecutiveFailures = 0;
            fprintf('[ODE4 rollout] %s IC %d success | elapsed=%.3f s | RMSE=%.3e\n', ...
                methodName,k,toc(icTimer),row.rawRMSE);
        catch ME
            row.message = ME.message;
            consecutiveFailures = consecutiveFailures+1;
            fprintf('[ODE4 rollout] %s IC %d failed | elapsed=%.3f s | %s\n', ...
                methodName,k,toc(icTimer),ME.message);
        end
        trajectoryRows(k,1) = row;

        if consecutiveFailures >= task.rollout.abortAfterConsecutiveFailures
            for kk = k+1:size(initialConditions,1)
                trajectoryRows(kk).initialCondition = initialConditions(kk,:);
                trajectoryRows(kk).message = sprintf( ...
                    'Skipped after %d consecutive rollout failures.', ...
                    task.rollout.abortAfterConsecutiveFailures);
                trajectoryRows(kk).failureTime = 0;
            end
            rollout.reason = sprintf('Aborted after %d consecutive failures.', ...
                task.rollout.abortAfterConsecutiveFailures);
            fprintf('[ODE4 rollout] %s aborted after %d consecutive failures.\n', ...
                methodName,task.rollout.abortAfterConsecutiveFailures);
            break;
        end
    end
    rollout.wallTime = toc(methodTimer);
    fprintf('[ODE4 rollout] method=%s finished | success=%d/%d | wall=%.3f s\n', ...
        methodName,rollout.successCount,rollout.totalCount,rollout.wallTime);

    rollout.trajectories = trajectoryRows;
    rollout.successRate = rollout.successCount/max(rollout.totalCount,1);
    rollout.perTrajectoryRMSE = perIcRawRMSE;
    rollout.perTrajectoryNormalizedRMSE = perIcNormalizedRMSE;

    successMask = reshape([trajectoryRows.success],[],1);
    if any(successMask)
        rollout.meanSuccessfulRawRMSE = mean(perIcRawRMSE(successMask));
        rollout.meanSuccessfulNormalizedRMSE = ...
            mean(perIcNormalizedRMSE(successMask));
    end
    if ~isempty(allRawSquared)
        rollout.pooledRawRMSE = sqrt(mean(allRawSquared(:)));
        rollout.pooledNormalizedRMSE = sqrt(mean(allNormSquared(:)));
    end

    if rollout.successCount == rollout.totalCount && rollout.totalCount > 0
        % Requested manuscript metric: first compute one trajectory RMSE per
        % unseen initial condition, then take their arithmetic mean.
        rollout.rawRMSE = mean(perIcRawRMSE);
        rollout.normalizedRMSE = mean(perIcNormalizedRMSE);
        rollout.perStateRMSE = mean(perIcStateRMSE,1);
        rollout.perStateNormalizedRMSE = ...
            mean(perIcStateNormalizedRMSE,1);
        fprintf(['[ODE4 rollout] %s aggregate | mean per-IC RMSE=%.3e | ', ...
            'mean per-IC NRMSE=%.3e | pooled RMSE=%.3e'], ...
            methodName,rollout.rawRMSE,rollout.normalizedRMSE, ...
            rollout.pooledRawRMSE);
    elseif rollout.successCount > 0
        % Preserve strict failure handling: a failed unseen-IC trajectory makes
        % the reported aggregate metric infinite. Successful-only means remain
        % available in the diagnostic fields above.
        rollout.rawRMSE = Inf;
        rollout.normalizedRMSE = Inf;
        rollout.reason = sprintf('%d/%d learned ODE4 trajectories failed.', ...
            rollout.totalCount-rollout.successCount, rollout.totalCount);
    else
        rollout.rawRMSE = Inf;
        rollout.normalizedRMSE = Inf;
        rollout.reason = 'All learned ODE4 trajectory integrations failed.';
    end
end

function [X,info] = ode4_fixed_local(rhsFcn,tEval,x0,rolloutOpts)
    nTime = numel(tEval);
    nx = numel(x0);
    X = nan(nTime,nx);
    X(1,:) = reshape(x0,1,[]);
    info = struct('success',true,'message','','failureTime',NaN, ...
        'rhsEvaluations',0);

    stateLimit = reshape(rolloutOpts.maxStateAbs,[],1);
    if isscalar(stateLimit)
        stateLimit = repmat(stateLimit,nx,1);
    end
    if numel(stateLimit) ~= nx
        error('rollout.maxStateAbs must be scalar or have one entry per state.');
    end
    derivativeLimit = rolloutOpts.maxDerivativeAbs;
    integrationTimer = tic;

    for i = 1:nTime-1
        if toc(integrationTimer) > rolloutOpts.maxWallTimePerIntegration
            info.success = false;
            info.message = sprintf('integration wall-time cap %.3g s exceeded', ...
                rolloutOpts.maxWallTimePerIntegration);
            info.failureTime = tEval(i);
            X = X(1:i,:);
            return;
        end
        if info.rhsEvaluations+4 > rolloutOpts.maxRhsEvaluationsPerIntegration
            info.success = false;
            info.message = sprintf('RHS evaluation cap %d exceeded', ...
                rolloutOpts.maxRhsEvaluationsPerIntegration);
            info.failureTime = tEval(i);
            X = X(1:i,:);
            return;
        end
        h = tEval(i+1)-tEval(i);
        x = X(i,:).';
        try
            k1 = checked_rhs_local(rhsFcn,x,stateLimit,derivativeLimit); info.rhsEvaluations = info.rhsEvaluations+1;
            k2 = checked_rhs_local(rhsFcn,x+0.5*h*k1,stateLimit,derivativeLimit); info.rhsEvaluations = info.rhsEvaluations+1;
            k3 = checked_rhs_local(rhsFcn,x+0.5*h*k2,stateLimit,derivativeLimit); info.rhsEvaluations = info.rhsEvaluations+1;
            k4 = checked_rhs_local(rhsFcn,x+h*k3,stateLimit,derivativeLimit); info.rhsEvaluations = info.rhsEvaluations+1;
            xNext = x+(h/6)*(k1+2*k2+2*k3+k4);
            if any(~isfinite(xNext))
                error('state became NaN or Inf');
            end
            if any(abs(xNext)>stateLimit)
                error('state safety envelope exceeded');
            end
            X(i+1,:) = xNext.';
        catch ME
            info.success = false;
            info.message = ME.message;
            info.failureTime = tEval(i);
            X = X(1:i,:);
            return;
        end
    end
end

function dx = checked_rhs_local(rhsFcn,x,stateLimit,derivativeLimit)
    if any(~isfinite(x))
        error('nonfinite state supplied to RHS');
    end
    if any(abs(x)>stateLimit)
        error('state safety envelope exceeded before RHS evaluation');
    end
    dx = reshape(rhsFcn(x),[],1);
    if numel(dx) ~= numel(x) || any(~isfinite(dx))
        error('RHS returned an invalid derivative');
    end
    if max(abs(dx)) > derivativeLimit
        error('derivative safety limit exceeded');
    end
end

function dx = true_rhs_local(x, task)
    dx = task.rhsFcn(reshape(x,1,[])).';
end

function dx = learned_rhs_local(x, predictFcn, ny)
    y = predictFcn(reshape(x,1,[]));
    if numel(y) ~= ny || any(~isfinite(y(:)))
        error('Learned vector field returned an invalid derivative.');
    end
    dx = reshape(y,[],1);
end

function X0 = make_initial_conditions_local(task)
    oldRng = rng;
    cleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
    rng(task.rollout.initialConditionSeed,'twister');
    X0 = sample_from_variable_domain(task.rollout.nInitialConditions, ...
        task.rollout.initialConditionDomain);
    if ~isempty(task.rollout.referenceInitialCondition)
        X0(1,:) = reshape(task.rollout.referenceInitialCondition,1,[]);
    end
end
