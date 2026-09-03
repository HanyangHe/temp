function [CoefOut, maskOut, stats] = post_bp_prune_refine_exact_phdn(CoefIn, maskIn, Xtr, Ytr, Xval, Yval, arch, normOpt, opts)
%POST_BP_PRUNE_REFINE_EXACT_PHDN v50a-style post-final-BP prune/refine pass.
%
% This helper is used after an exact dictionary-PhDN BP-LSQ refinement.  It
% prunes low-score active coefficients and then runs a short support-fixed
% BP-LSQ refinement on the pruned support.  A candidate pruning iteration is
% accepted only if the validation MSE is not worse than before pruning.
%
% The default score mode is the v50a-compatible mean absolute contribution:
%   score(A_ij) = mean(abs(A_ij * Phi_j(samples))).
%
% Required public helpers from the framework: model_forward,
% compute_regression_metrics, create_coef_template, pack_Coef_M_by_mask,
% unpack_Coef_M_by_mask, count_active_mask, residual_jacobian_masked_phdn.

    cfg = parse_post_bp_prune_cfg_local(opts);
    stats = struct();
    stats.applied = false;
    stats.acceptedAny = false;
    stats.originalActive = count_active_mask(maskIn);
    stats.finalActive = stats.originalActive;
    stats.elapsedTime = 0;
    stats.iterations = {};
    tAll = tic;

    CoefOut = CoefIn;
    maskOut = maskIn;

    if ~cfg.enable || cfg.numIterations <= 0 || stats.originalActive <= 0
        stats.elapsedTime = toc(tAll);
        return;
    end

    coefTemplate = create_coef_template(arch);
    thetaOut = pack_Coef_M_by_mask(CoefOut, maskOut);
    if isempty(thetaOut)
        stats.elapsedTime = toc(tAll);
        return;
    end

    for it = 1:cfg.numIterations
        activeBefore = count_active_mask(maskOut);
        trainBefore = eval_mse_postbp_local(CoefOut, Xtr, Ytr, arch, normOpt);
        valBefore = eval_mse_postbp_local(CoefOut, Xval, Yval, arch, normOpt);
        [maskCand, pruneInfo] = prune_mask_by_postbp_score_local(CoefOut, maskOut, Xtr, arch, normOpt, cfg);
        activeAfter = count_active_mask(maskCand);

        iterStats = struct();
        iterStats.iteration = it;
        iterStats.activeBefore = activeBefore;
        iterStats.activeAfterPrune = activeAfter;
        iterStats.nPruned = activeBefore - activeAfter;
        iterStats.trainBefore = trainBefore;
        iterStats.valBefore = valBefore;
        iterStats.trainAfter = NaN;
        iterStats.valAfter = NaN;
        iterStats.accepted = false;
        iterStats.reason = '';
        iterStats.pruneInfo = pruneInfo;

        if activeAfter >= activeBefore
            iterStats.reason = 'no_terms_below_score_threshold';
            stats.iterations{end + 1} = iterStats; %#ok<AGROW>
            if cfg.verbose
                fprintf('  post-BP prune iter %d/%d: active %d -> %d, no terms pruned [score=%s, thr %.3e, rel %.3e, belowThr %d]\n', ...
                    it, cfg.numIterations, activeBefore, activeAfter, pruneInfo.scoreMode, pruneInfo.threshold, cfg.relThreshold, pruneInfo.nBelowThresholdBeforeRowFloor);
            end
            break;
        end

        thetaCand0 = pack_Coef_M_by_mask(CoefOut, maskCand);
        [thetaCand, CoefCand, trainCand, valCand] = run_postbp_lsq_refine_local(thetaCand0, maskCand, coefTemplate, Xtr, Ytr, Xval, Yval, arch, normOpt, opts, cfg);
        iterStats.trainAfter = trainCand;
        iterStats.valAfter = valCand;

        accept = true;
        if cfg.acceptByValidation
            accept = isfinite(valCand) && valCand <= valBefore * (1 + cfg.maxRelValIncrease);
        end

        if accept
            thetaOut = thetaCand(:); %#ok<NASGU>
            CoefOut = CoefCand;
            maskOut = maskCand;
            iterStats.accepted = true;
            iterStats.reason = 'accepted';
            stats.acceptedAny = true;
        else
            iterStats.reason = 'validation_rejected';
        end

        stats.iterations{end + 1} = iterStats; %#ok<AGROW>
        if cfg.verbose
            fprintf('  post-BP prune iter %d/%d: active %d -> %d, pruned %d, train/val %.3e/%.3e -> %.3e/%.3e, accepted %d\n', ...
                it, cfg.numIterations, activeBefore, activeAfter, activeBefore - activeAfter, ...
                trainBefore, valBefore, trainCand, valCand, iterStats.accepted);
        end

        if ~iterStats.accepted
            break;
        end
    end

    stats.applied = true;
    stats.finalActive = count_active_mask(maskOut);
    stats.elapsedTime = toc(tAll);
end

function cfg = parse_post_bp_prune_cfg_local(opts)
    if isfield(opts, 'init') && isfield(opts.init, 'postBPPrune') && isstruct(opts.init.postBPPrune)
        src = opts.init.postBPPrune;
    else
        src = struct();
    end
    cfg = struct();
    cfg.enable = getfield_default_postbp_local(src, 'enable', true);
    cfg.numIterations = max(0, round(getfield_default_postbp_local(src, 'numIterations', 1)));
    cfg.scoreMode = lower(strtrim(char(getfield_default_postbp_local(src, 'scoreMode', 'contribution_abs_mean'))));
    if any(strcmpi(cfg.scoreMode, {'coef','coef_abs','coefficient','coefficient_abs','abs_coef','raw_coef_abs'}))
        cfg.scoreMode = 'coef_abs';
    elseif any(strcmpi(cfg.scoreMode, {'contribution','contribution_abs_mean','mean_abs_contribution','legacy_contribution'}))
        cfg.scoreMode = 'contribution_abs_mean';
    else
        cfg.scoreMode = 'contribution_abs_mean';
    end
    absThr = getfield_default_postbp_local(src, 'absThreshold', []);
    relThr = getfield_default_postbp_local(src, 'relThreshold', []);
    legacyAbs = getfield_default_postbp_local(src, 'contributionAbsThreshold', []);
    legacyRel = getfield_default_postbp_local(src, 'contributionRelThreshold', []);
    if isempty(absThr); absThr = legacyAbs; end
    if isempty(relThr); relThr = legacyRel; end
    if isempty(absThr) || ~isfinite(absThr); absThr = 1e-10; end
    if isempty(relThr) || ~isfinite(relThr); relThr = 0; end
    cfg.absThreshold = absThr;
    cfg.relThreshold = relThr;
    cfg.minTermsPerXiRow = max(0, round(getfield_default_postbp_local(src, 'minTermsPerXiRow', 0)));
    cfg.refineMaxIter = max(0, round(getfield_default_postbp_local(src, 'refineMaxIter', 50)));
    cfg.refineMaxFunEvals = max(100, round(getfield_default_postbp_local(src, 'refineMaxFunEvals', 5000)));
    cfg.acceptByValidation = getfield_default_postbp_local(src, 'acceptByValidation', true);
    cfg.maxRelValIncrease = getfield_default_postbp_local(src, 'maxRelValIncrease', 1e-4);
    cfg.verbose = getfield_default_postbp_local(src, 'verbose', true);
end

function [maskOut, info] = prune_mask_by_postbp_score_local(Coef, maskIn, X, arch, normOpt, cfg)
    maskOut = maskIn;
    info = struct('scoreMode', cfg.scoreMode, 'threshold', NaN, 'nRawPruned', 0, ...
        'nProtected', 0, 'nBelowThresholdBeforeRowFloor', 0, 'scoreMin', NaN, ...
        'scoreMedian', NaN, 'scoreMax', NaN);
    if isempty(maskIn) || count_active_mask(maskIn) == 0
        return;
    end

    cache = [];
    if strcmpi(cfg.scoreMode, 'contribution_abs_mean')
        try
            [~, cache] = model_forward(X, Coef, arch, normOpt);
        catch
            return;
        end
    end

    allScores = [];
    for ell = 1:size(maskIn, 2)
        for src = 1:ell
            M = maskIn{src, ell};
            if isempty(M) || ~any(M(:)), continue; end
            C = Coef{src, ell};
            active = find(M);
            [rowIdx, colIdx] = ind2sub(size(M), active);
            for k = 1:numel(active)
                s = postbp_term_score_local(C, cache, src, ell, rowIdx(k), colIdx(k), cfg.scoreMode);
                if isfinite(s)
                    allScores(end + 1, 1) = s; %#ok<AGROW>
                end
            end
        end
    end
    if isempty(allScores)
        return;
    end
    info.scoreMin = min(allScores);
    info.scoreMedian = median(allScores);
    info.scoreMax = max(allScores);
    info.threshold = max(cfg.absThreshold, cfg.relThreshold * info.scoreMax);

    for ell = 1:size(maskIn, 2)
        for src = 1:ell
            M = maskIn{src, ell};
            if isempty(M) || ~any(M(:)), continue; end
            C = Coef{src, ell};
            scoreMat = inf(size(M));
            active = find(M);
            [rowIdx, colIdx] = ind2sub(size(M), active);
            for k = 1:numel(active)
                scoreMat(rowIdx(k), colIdx(k)) = postbp_term_score_local(C, cache, src, ell, rowIdx(k), colIdx(k), cfg.scoreMode);
            end
            remove = M & (scoreMat <= info.threshold);
            info.nBelowThresholdBeforeRowFloor = info.nBelowThresholdBeforeRowFloor + nnz(remove);
            if cfg.minTermsPerXiRow > 0
                for r = 1:size(M, 1)
                    idxRow = find(M(r, :));
                    if isempty(idxRow), continue; end
                    willKeep = idxRow(~remove(r, idxRow));
                    if numel(willKeep) < cfg.minTermsPerXiRow
                        [~, ord] = sort(scoreMat(r, idxRow), 'descend');
                        keepNeed = idxRow(ord(1:min(cfg.minTermsPerXiRow, numel(idxRow))));
                        remove(r, keepNeed) = false;
                    end
                end
            end
            info.nRawPruned = info.nRawPruned + nnz(remove);
            maskOut{src, ell} = M & ~remove;
        end
    end
end

function s = postbp_term_score_local(C, cache, src, ell, r, c, scoreMode)
    if r > size(C, 1) || c > size(C, 2)
        s = Inf;
        return;
    end
    coefAbs = abs(C(r, c));
    if strcmpi(scoreMode, 'coef_abs')
        s = coefAbs;
        return;
    end
    if strcmpi(scoreMode, 'contribution_abs_mean')
        try
            Phi = cache.branch{src, ell}.Phi;
            if c <= size(Phi, 1)
                s = mean(abs(C(r, c) .* Phi(c, :)));
            else
                s = Inf;
            end
        catch
            s = Inf;
        end
        return;
    end
    s = coefAbs;
end

function [thetaFit, CoefFit, trainMSE, valMSE] = run_postbp_lsq_refine_local(theta0, mask, coefTemplate, Xtr, Ytr, Xval, Yval, arch, normOpt, opts, cfg)
    cfgObjective = struct();
    if isfield(opts, 'init') && isfield(opts.init, 'objective') && isstruct(opts.init.objective)
        cfgObjective = opts.init.objective;
    end
    cfgObjective.lambda1 = 0;
    if ~isfield(cfgObjective, 'lambda2') || isempty(cfgObjective.lambda2)
        cfgObjective.lambda2 = 0;
    end
    scaleY = make_residual_scale_postbp_local(Ytr, cfgObjective);
    useJac = true;
    if isfield(opts, 'init') && isfield(opts.init, 'lsq') && isfield(opts.init.lsq, 'useAnalyticJacobian')
        useJac = opts.init.lsq.useAnalyticJacobian;
    end
    obj = @(th) residual_jacobian_masked_phdn(th, mask, coefTemplate, Xtr, Ytr, arch, normOpt, cfgObjective, scaleY);
    thetaFit = theta0(:);
    try
        lsqOpts = optimoptions('lsqnonlin', 'Display', 'off', ...
            'MaxIterations', cfg.refineMaxIter, 'MaxFunctionEvaluations', cfg.refineMaxFunEvals, ...
            'StepTolerance', 1e-10, 'FunctionTolerance', 1e-10, ...
            'SpecifyObjectiveGradient', useJac);
        thetaFit = lsqnonlin(obj, theta0(:), [], [], lsqOpts);
    catch ME
        warning('Post-BP prune/refine LSQ failed (%s). Keeping the current coefficients.', ME.message);
    end
    CoefFit = unpack_Coef_M_by_mask(thetaFit, mask, coefTemplate);
    trainMSE = eval_mse_postbp_local(CoefFit, Xtr, Ytr, arch, normOpt);
    valMSE = eval_mse_postbp_local(CoefFit, Xval, Yval, arch, normOpt);
end

function mse = eval_mse_postbp_local(Coef, X, Y, arch, normOpt)
    try
        Yp = model_forward(X, Coef, arch, normOpt);
        m = compute_regression_metrics(Yp, Y);
        mse = m.mse;
    catch
        mse = Inf;
    end
end

function scaleY = make_residual_scale_postbp_local(Y, cfgObjective)
    if isfield(cfgObjective, 'normalizeResidual') && ~cfgObjective.normalizeResidual
        scaleY = 1;
        return;
    end
    scaleY = std(Y, 0, 1);
    scaleY(~isfinite(scaleY) | scaleY < 1e-12) = 1;
end

function val = getfield_default_postbp_local(s, name, defaultVal)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = defaultVal;
    end
end
