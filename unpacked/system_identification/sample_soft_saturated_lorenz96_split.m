function X = sample_soft_saturated_lorenz96_split(nSamples, domain, plan)
%SAMPLE_SOFT_SATURATED_LORENZ96_SPLIT Core--shell scrambled-Sobol data design.
%
% The SoftSaturatedLorenz96 sample-efficiency experiment uses four distinct,
% reproducible scrambled-Sobol streams per round:
%   1) one max-size ID training master pool sampled from the ID core;
%   2) one fixed ID validation pool sampled from the ID boundary shell;
%   3) one fixed ID test pool sampled over the complete original ID box;
%   4) one fixed OOD test pool sampled over the declared OOD box.
%
% Default ID geometry (all radii use the normalized L-infinity distance from
% the center of the original ID box):
%
%   training:   r_inf <= 0.90
%   validation: 0.92 <= r_inf <= 1.00
%   ID test:    complete original ID box
%
% Hence validation remains strictly inside the declared ID domain and never
% uses the OOD test set, while carrying a controlled covariate shift that is
% useful for selecting structures with better cross-region generalization.
%
% Optional plan fields:
%   idValidationSplitMode          : 'core_shell' (default) | 'iid_full_id'
%   trainingCoreRadius             : default 0.90
%   validationShellInnerRadius     : default 0.92
%   validationShellOuterRadius     : default 1.00
%
% The requested training set is always the prefix of the same max-size core
% master pool, preserving exact nesting across the sample-efficiency sweep.
% Validation, ID-test, and OOD-test points stay fixed within each round.
%
% sample_task_data() later invokes split_train_val_test(), which applies a
% random permutation. This helper predicts that permutation without consuming
% the caller's RNG state and inverse-arranges the combined ID matrix so the
% shared framework still returns the intended exact split.

    required = {'nTrain','nValidation','nTest','maxTrain', ...
        'trainSeed','validationSeed','testSeed','oodSeed'};
    for k = 1:numel(required)
        if ~isfield(plan,required{k}) || isempty(plan.(required{k}))
            error('Sampling plan field %s is missing.',required{k});
        end
    end

    samplingMethod = get_plan_field_local(plan,'samplingMethod','scrambled_sobol');
    if ~strcmpi(strtrim(char(samplingMethod)),'scrambled_sobol')
        error('SoftSaturatedLorenz96 requires samplingMethod=''scrambled_sobol''; received %s.', ...
            char(string(samplingMethod)));
    end
    scrambleMethod = get_plan_field_local(plan,'sobolScrambleMethod','MatousekAffineOwen');
    sobolSkip = get_plan_field_local(plan,'sobolSkip',1024);
    if ~isscalar(sobolSkip) || ~isfinite(sobolSkip) || sobolSkip < 0 || sobolSkip ~= floor(sobolSkip)
        error('plan.sobolSkip must be a nonnegative integer scalar.');
    end

    if exist('sobolset','file') ~= 2
        error(['sobolset is unavailable. The scrambled-Sobol design requires ', ...
            'MATLAB Statistics and Machine Learning Toolbox.']);
    end

    domain = normalize_task_domain(domain);
    isOodDomain = is_ood_domain_local(domain);
    callerState = rng;
    cleanup = onCleanup(@() rng(callerState)); %#ok<NASGU>

    % OOD sampling is independent of the ID split and is never used for
    % training, validation, early stopping, or structure selection.
    if isOodDomain
        X = generate_disjoint_sobol_pool_local( ...
            nSamples,domain,plan.oodSeed,zeros(0,numel(domain.intervals)), ...
            sobolSkip,scrambleMethod);
        assert_unique_rows_local(X,'OOD test');
        return;
    end

    expectedID = plan.nTrain + plan.nValidation + plan.nTest;
    if nSamples ~= expectedID
        error('nSamples=%d but the exact ID split plan requires %d.', ...
            nSamples,expectedID);
    end
    if plan.nTrain > plan.maxTrain
        error('Requested nTrain=%d exceeds maxTrain=%d.',plan.nTrain,plan.maxTrain);
    end

    nx = numel(domain.intervals);
    splitMode = lower(strtrim(char(get_plan_field_local( ...
        plan,'idValidationSplitMode','core_shell'))));

    switch splitMode
        case 'core_shell'
            coreRadius = get_plan_field_local(plan,'trainingCoreRadius',0.90);
            shellInnerRadius = get_plan_field_local( ...
                plan,'validationShellInnerRadius',0.92);
            shellOuterRadius = get_plan_field_local( ...
                plan,'validationShellOuterRadius',1.00);
            validate_core_shell_radii_local( ...
                coreRadius,shellInnerRadius,shellOuterRadius);

            % One fixed max-size training stream in the central ID core.
            % Smaller Ntrain settings use exact prefixes of this pool.
            XtrainMaster = generate_core_sobol_pool_local( ...
                plan.maxTrain,domain,plan.trainSeed,coreRadius, ...
                sobolSkip,scrambleMethod);

            % Independent validation stream in the boundary shell of the same
            % original ID box. The buffer between coreRadius and shellInnerRadius
            % prevents train/validation points from approaching the same boundary.
            Xvalidation = generate_shell_sobol_pool_local( ...
                plan.nValidation,domain,plan.validationSeed, ...
                shellInnerRadius,shellOuterRadius,XtrainMaster, ...
                sobolSkip,scrambleMethod);

            assert_core_shell_geometry_local( ...
                XtrainMaster,Xvalidation,domain,coreRadius, ...
                shellInnerRadius,shellOuterRadius);
            print_core_shell_design_once_local( ...
                coreRadius,shellInnerRadius,shellOuterRadius,plan);

        case {'iid_full_id','full_id_iid','legacy_iid'}
            % Explicit compatibility mode reproducing the former same-box
            % train/validation sampling design.
            XtrainMaster = generate_disjoint_sobol_pool_local( ...
                plan.maxTrain,domain,plan.trainSeed,zeros(0,nx), ...
                sobolSkip,scrambleMethod);
            Xvalidation = generate_disjoint_sobol_pool_local( ...
                plan.nValidation,domain,plan.validationSeed,XtrainMaster, ...
                sobolSkip,scrambleMethod);

        otherwise
            error(['Unknown plan.idValidationSplitMode=''%s''. Supported modes ', ...
                'are ''core_shell'' and ''iid_full_id''.'],splitMode);
    end

    % The ID test set continues to cover the complete original ID box. It is
    % independent of the shifted validation set and is used only for reporting.
    Xtest = generate_disjoint_sobol_pool_local( ...
        plan.nTest,domain,plan.testSeed,[XtrainMaster;Xvalidation], ...
        sobolSkip,scrambleMethod);

    assert_unique_rows_local(XtrainMaster,'training master');
    assert_unique_rows_local(Xvalidation,'validation');
    assert_unique_rows_local(Xtest,'ID test');
    assert_disjoint_rows_local(XtrainMaster,Xvalidation,'training master','validation');
    assert_disjoint_rows_local(XtrainMaster,Xtest,'training master','ID test');
    assert_disjoint_rows_local(Xvalidation,Xtest,'validation','ID test');

    Xtrain = XtrainMaster(1:plan.nTrain,:);
    desiredSplitOrder = [Xtrain;Xvalidation;Xtest];

    % Reproduce the next shared-framework randperm, then restore its start state.
    rng(callerState);
    splitStartState = rng;
    splitIndex = randperm(nSamples);
    rng(splitStartState);

    X = zeros(size(desiredSplitOrder));
    X(splitIndex,:) = desiredSplitOrder;
end

function X = generate_core_sobol_pool_local(nSamples,domain,seed,coreRadius,skip,scrambleMethod)
% Uniform scrambled-Sobol sampling in the centered rectangular ID core.

    domain = normalize_task_domain(domain);
    [lb,ub] = finite_box_bounds_local(domain);
    center = 0.5*(lb+ub);
    halfWidth = 0.5*(ub-lb);

    coreDomain = domain;
    coreDomain.lb = center-coreRadius*halfWidth;
    coreDomain.ub = center+coreRadius*halfWidth;
    for j = 1:numel(coreDomain.intervals)
        coreDomain.intervals{j} = [coreDomain.lb(j),coreDomain.ub(j)];
    end
    coreDomain.source = 'SoftSaturatedLorenz96_ID_training_core';

    X = generate_disjoint_sobol_pool_local( ...
        nSamples,coreDomain,seed,zeros(0,numel(lb)),skip,scrambleMethod);
end

function X = generate_shell_sobol_pool_local(nSamples,domain,seed,innerRadius,outerRadius,forbidden,skip,scrambleMethod)
% Uniform scrambled-Sobol rejection sampling from an L-infinity ID shell.

    domain = normalize_task_domain(domain);
    [lb,ub] = finite_box_bounds_local(domain);
    nx = numel(lb);
    center = 0.5*(lb+ub);
    halfWidth = 0.5*(ub-lb);

    if isempty(forbidden)
        forbidden = zeros(0,nx);
    elseif size(forbidden,2) ~= nx
        error('Forbidden-row dimension does not match the shell dimension.');
    end

    rng(seed,'twister');
    pointSet = sobolset(nx,'Skip',skip);
    pointSet = scramble(pointSet,char(scrambleMethod));

    shellFraction = max(eps,outerRadius^nx-innerRadius^nx);
    nCandidate = max(nSamples+128,ceil(1.35*nSamples/shellFraction));
    maxCandidate = max(64*nSamples,nSamples+16384);
    tolerance = 64*eps;

    while true
        U = net(pointSet,nCandidate);
        candidate = lb + U.*(ub-lb);
        Z = (candidate-center)./halfWidth;
        radius = max(abs(Z),[],2);
        keep = radius >= innerRadius-tolerance & ...
            radius <= outerRadius+tolerance;
        candidate = candidate(keep,:);
        candidate = unique(candidate,'rows','stable');
        if ~isempty(forbidden)
            candidate(ismember(candidate,forbidden,'rows'),:) = [];
        end
        if size(candidate,1) >= nSamples
            X = candidate(1:nSamples,:);
            return;
        end
        if nCandidate >= maxCandidate
            error(['Unable to construct %d validation-shell samples after ', ...
                'generating %d Sobol candidates.'],nSamples,nCandidate);
        end
        nCandidate = min(maxCandidate,2*nCandidate);
    end
end

function X = generate_disjoint_sobol_pool_local(nSamples,domain,seed,forbidden,skip,scrambleMethod)
% Generate a reproducible scrambled-Sobol pool and remove exact overlaps.

    domain = normalize_task_domain(domain);
    nx = numel(domain.intervals);
    nIntervals = cellfun(@(I) size(I,1),domain.intervals);
    if any(nIntervals ~= 1)
        error(['The SoftSaturatedLorenz96 Sobol sampler currently requires ', ...
            'one continuous interval per state variable.']);
    end

    [lb,ub] = finite_box_bounds_local(domain);

    if isempty(forbidden)
        forbidden = zeros(0,nx);
    elseif size(forbidden,2) ~= nx
        error('Forbidden-row dimension does not match the Sobol domain dimension.');
    end

    rng(seed,'twister');
    pointSet = sobolset(nx,'Skip',skip);
    pointSet = scramble(pointSet,char(scrambleMethod));

    % Duplicates are extremely unlikely after scrambling, but generate a small
    % reserve and enlarge deterministically if filtering ever removes points.
    nCandidate = max(nSamples+64,ceil(1.10*nSamples));
    maxCandidate = max(16*nSamples,nSamples+4096);
    while true
        U = net(pointSet,nCandidate);
        candidate = lb + U.*(ub-lb);
        candidate = unique(candidate,'rows','stable');
        if ~isempty(forbidden)
            candidate(ismember(candidate,forbidden,'rows'),:) = [];
        end
        if size(candidate,1) >= nSamples
            X = candidate(1:nSamples,:);
            return;
        end
        if nCandidate >= maxCandidate
            error(['Unable to construct %d unique, disjoint Sobol samples ', ...
                'after generating %d candidates.'],nSamples,nCandidate);
        end
        nCandidate = min(maxCandidate,2*nCandidate);
    end
end

function [lb,ub] = finite_box_bounds_local(domain)
    lb = reshape(domain.lb,1,[]);
    ub = reshape(domain.ub,1,[]);
    nx = numel(domain.intervals);
    if numel(lb) ~= nx || numel(ub) ~= nx || ...
            any(~isfinite(lb)) || any(~isfinite(ub)) || any(ub <= lb)
        error('Invalid finite box bounds supplied to the Sobol sampler.');
    end
end

function radius = normalized_linf_radius_local(X,domain)
    [lb,ub] = finite_box_bounds_local(domain);
    center = 0.5*(lb+ub);
    halfWidth = 0.5*(ub-lb);
    radius = max(abs((X-center)./halfWidth),[],2);
end

function validate_core_shell_radii_local(coreRadius,innerRadius,outerRadius)
    values = [coreRadius,innerRadius,outerRadius];
    if any(~isfinite(values)) || any(values <= 0) || ...
            ~(coreRadius < innerRadius && innerRadius < outerRadius) || ...
            outerRadius > 1
        error(['Core--shell radii must satisfy ', ...
            '0 < trainingCoreRadius < validationShellInnerRadius < ', ...
            'validationShellOuterRadius <= 1.']);
    end
end

function assert_core_shell_geometry_local(Xtrain,Xvalidation,domain,coreRadius,innerRadius,outerRadius)
    tolerance = 1e-12;
    trainRadius = normalized_linf_radius_local(Xtrain,domain);
    validationRadius = normalized_linf_radius_local(Xvalidation,domain);
    if any(trainRadius > coreRadius+tolerance)
        error('The generated training master contains points outside the ID core.');
    end
    if any(validationRadius < innerRadius-tolerance) || ...
            any(validationRadius > outerRadius+tolerance)
        error('The generated validation pool violates the requested ID shell.');
    end
end

function print_core_shell_design_once_local(coreRadius,innerRadius,outerRadius,plan)
    persistent printedKeys
    if isempty(printedKeys)
        printedKeys = {};
    end
    key = sprintf('%d_%d_%d_%.12g_%.12g_%.12g', ...
        plan.trainSeed,plan.validationSeed,plan.testSeed, ...
        coreRadius,innerRadius,outerRadius);
    if any(strcmp(printedKeys,key))
        return;
    end
    printedKeys{end+1} = key; %#ok<AGROW>
    fprintf(['Core--shell ID split active: train r_inf<=%.3f | ', ...
        'validation %.3f<=r_inf<=%.3f | ID test=full original ID box.\n'], ...
        coreRadius,innerRadius,outerRadius);
end

function tf = is_ood_domain_local(domain)
    tf = false;
    if isfield(domain,'source') && ~isempty(domain.source)
        tf = contains(lower(char(string(domain.source))),'ood');
    end
end

function value = get_plan_field_local(plan,name,defaultValue)
    if isfield(plan,name) && ~isempty(plan.(name))
        value = plan.(name);
    else
        value = defaultValue;
    end
end

function assert_unique_rows_local(X,label)
    if size(unique(X,'rows'),1) ~= size(X,1)
        error('The %s Sobol pool contains duplicate rows.',label);
    end
end

function assert_disjoint_rows_local(A,B,labelA,labelB)
    if any(ismember(A,B,'rows'))
        error('Sobol pools %s and %s overlap.',labelA,labelB);
    end
end
