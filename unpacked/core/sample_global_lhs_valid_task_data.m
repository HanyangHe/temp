function [X, Y, info] = sample_global_lhs_valid_task_data(task, nSamples, domain, seed, options)
%SAMPLE_GLOBAL_LHS_VALID_TASK_DATA Global 3-D LHS with validity rejection.
%
% [X,Y,INFO] = SAMPLE_GLOBAL_LHS_VALID_TASK_DATA(TASK,N,DOMAIN,SEED,OPTIONS)
% generates one randomized Latin-hypercube design over the complete Cartesian
% product of DOMAIN's per-variable interval unions. Candidate points are not
% drawn from a mixture of hand-selected subregions. A candidate is rejected
% only when TASK.sampleValidityFcn reports that it is outside the real-valued
% operator domain or too close to a declared singularity.
%
% The first candidate batch contains exactly N points. Therefore, when no
% rejection is needed, the returned set is an exact N-point randomized LHS.
% Additional independent global LHS batches are generated only to replace
% rejected candidates.

    if nargin < 3 || isempty(domain)
        domain = task.domain;
    end
    if nargin < 4 || isempty(seed)
        seed = 1;
    end
    if nargin < 5 || isempty(options)
        options = struct();
    end
    if ~isscalar(nSamples) || nSamples < 1 || nSamples ~= floor(nSamples)
        error('nSamples must be a positive integer.');
    end

    nx = task.nx;
    if isfield(task, 'variableNames')
        varNames = task.variableNames;
    else
        varNames = arrayfun(@(k) sprintf('x%d', k), 1:nx, 'UniformOutput', false);
    end
    domain = normalize_task_domain(domain, nx, varNames);

    options = apply_sampling_defaults_local(task, options);

    oldRng = rng;
    rngCleanup = onCleanup(@() rng(oldRng)); %#ok<NASGU>
    rng(seed, 'twister');

    X = zeros(0, nx);
    Y = zeros(0, task.ny);
    nGenerated = 0;
    nRejectedValidity = 0;
    nRejectedNonfiniteTarget = 0;
    batchSizes = zeros(0,1);

    for batchIndex = 1:options.maxBatches
        nRemaining = nSamples - size(X,1);
        if nRemaining <= 0
            break;
        end

        if batchIndex == 1
            nCandidate = nRemaining;
        else
            nCandidate = max(options.minimumReplacementBatchSize, ...
                ceil(options.replacementOversampleFactor * nRemaining));
        end

        U = randomized_lhs_local(nCandidate, nx);
        Xcandidate = map_unit_lhs_to_union_domain_local(U, domain);
        nGenerated = nGenerated + nCandidate;
        batchSizes(end+1,1) = nCandidate; %#ok<AGROW>

        [valid, validityDetails] = evaluate_validity_local(task, Xcandidate, options);
        if ~islogical(valid)
            valid = logical(valid);
        end
        valid = reshape(valid, [], 1);
        if numel(valid) ~= nCandidate
            error('task.sampleValidityFcn must return one logical value per candidate point.');
        end
        nRejectedValidity = nRejectedValidity + sum(~valid);

        Xvalid = Xcandidate(valid,:);
        if isempty(Xvalid)
            continue;
        end

        Yvalid = task.rhsFcn(Xvalid);
        finiteTarget = all(isfinite(Yvalid), 2);
        if isfinite(options.maximumAbsoluteTarget)
            finiteTarget = finiteTarget & all(abs(Yvalid) <= options.maximumAbsoluteTarget, 2);
        end
        nRejectedNonfiniteTarget = nRejectedNonfiniteTarget + sum(~finiteTarget);

        Xvalid = Xvalid(finiteTarget,:);
        Yvalid = Yvalid(finiteTarget,:);
        if isempty(Xvalid)
            continue;
        end

        nTake = min(nRemaining, size(Xvalid,1));
        X = [X; Xvalid(1:nTake,:)]; %#ok<AGROW>
        Y = [Y; Yvalid(1:nTake,:)]; %#ok<AGROW>

        %#ok<NASGU> validityDetails is intentionally evaluated for task-level
        % diagnostics even though only aggregate rejection counts are stored.
    end

    if size(X,1) < nSamples
        error(['Could not generate %d valid global LHS samples after %d batches. ', ...
            'Accepted %d of %d candidates. Check the domain or validity margins.'], ...
            nSamples, options.maxBatches, size(X,1), nGenerated);
    end

    info = struct();
    info.method = 'global_randomized_lhs_with_validity_rejection';
    info.seed = seed;
    info.domainSource = get_struct_field_local(domain, 'source', 'unspecified');
    info.requestedSamples = nSamples;
    info.acceptedSamples = size(X,1);
    info.generatedCandidates = nGenerated;
    info.rejectedByValidity = nRejectedValidity;
    info.rejectedByNonfiniteTarget = nRejectedNonfiniteTarget;
    info.totalRejected = nRejectedValidity + nRejectedNonfiniteTarget;
    info.rejectionRate = info.totalRejected / max(nGenerated, 1);
    info.nBatches = numel(batchSizes);
    info.batchSizes = batchSizes;
    info.options = options;
end

function options = apply_sampling_defaults_local(task, options)
    defaults = struct();
    defaults.maxBatches = 100;
    defaults.minimumReplacementBatchSize = 32;
    defaults.replacementOversampleFactor = 1.25;
    defaults.maximumAbsoluteTarget = Inf;

    if isfield(task, 'sampling') && isstruct(task.sampling)
        taskDefaults = task.sampling;
        fields = fieldnames(taskDefaults);
        for k = 1:numel(fields)
            defaults.(fields{k}) = taskDefaults.(fields{k});
        end
    end

    fields = fieldnames(defaults);
    for k = 1:numel(fields)
        if ~isfield(options, fields{k}) || isempty(options.(fields{k}))
            options.(fields{k}) = defaults.(fields{k});
        end
    end
end

function U = randomized_lhs_local(n, d)
    U = zeros(n,d);
    for j = 1:d
        stratumOrder = randperm(n).';
        U(:,j) = (stratumOrder - 1 + rand(n,1)) ./ n;
    end
end

function X = map_unit_lhs_to_union_domain_local(U, domain)
    [n, d] = size(U);
    X = zeros(n,d);

    for j = 1:d
        intervals = domain.intervals{j};
        lengths = intervals(:,2) - intervals(:,1);
        positive = lengths > 0;
        intervals = intervals(positive,:);
        lengths = lengths(positive);
        if isempty(intervals)
            error('Variable %d has no positive-length sampling interval.', j);
        end

        totalLength = sum(lengths);
        distance = min(U(:,j), 1-eps) .* totalLength;
        cumulative = cumsum(lengths);
        intervalIndex = ones(n,1);
        for r = 1:numel(lengths)-1
            intervalIndex(distance >= cumulative(r)) = r + 1;
        end
        previousCumulative = [0; cumulative(1:end-1)];
        localDistance = distance - previousCumulative(intervalIndex);
        X(:,j) = intervals(intervalIndex,1) + localDistance;
    end
end

function [valid, details] = evaluate_validity_local(task, X, options)
    if isfield(task, 'sampleValidityFcn') && ~isempty(task.sampleValidityFcn)
        try
            [valid, details] = task.sampleValidityFcn(X, options);
        catch ME_two_output
            try
                valid = task.sampleValidityFcn(X, options);
                details = struct();
            catch
                rethrow(ME_two_output);
            end
        end
    else
        valid = all(isfinite(X),2);
        details = struct();
    end
end

function value = get_struct_field_local(S, fieldName, defaultValue)
    if isstruct(S) && isfield(S, fieldName) && ~isempty(S.(fieldName))
        value = S.(fieldName);
    else
        value = defaultValue;
    end
end
