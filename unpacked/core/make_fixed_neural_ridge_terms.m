function [terms,meta] = make_fixed_neural_ridge_terms(H,nTarget,activation, ...
        quantiles,scales,poolRatio,seed,stdFloor,varThreshold,corrThreshold,ensureFullSpan)
%MAKE_FIXED_NEURAL_RIDGE_TERMS Build one reproducible fixed neural-ridge bank.
%
% H is d-by-N and contains the samples seen by one dictionary input block.
% The routine is shared by PhDN Stage-1 augmentation and Neural-SINDy so both
% methods use the same direction/location/scale construction and screening.

    validateattributes(H,{'numeric'},{'2d','real','finite','nonempty'},mfilename,'H');
    d = size(H,1);
    nSamples = size(H,2);
    nTarget = max(1,round(double(nTarget)));
    activation = lower(strtrim(char(activation)));
    if ~strcmp(activation,'tanh')
        error('Only tanh fixed neural-ridge bases are currently supported.');
    end
    quantiles = reshape(double(quantiles),1,[]);
    scales = reshape(double(scales),1,[]);
    quantiles = unique(quantiles(isfinite(quantiles) & quantiles>0 & quantiles<1),'stable');
    scales = unique(scales(isfinite(scales) & scales>0),'stable');
    if isempty(quantiles) || isempty(scales)
        error('Neural quantile and activation-scale sets must be nonempty.');
    end
    poolRatio = max(1,double(poolRatio));
    seed = round(double(seed));
    stdFloor = max(eps,double(stdFloor));
    varThreshold = max(0,double(varThreshold));
    corrThreshold = min(max(double(corrThreshold),0),1);
    ensureFullSpan = logical(ensureFullSpan);

    mu = mean(H,2);
    coordScale = std(H,0,2);
    coordScale(~isfinite(coordScale) | coordScale<stdFloor) = stdFloor;
    Hn = (H-mu)./coordScale;

    nCandidateTarget = ceil(poolRatio*nTarget);
    nPerDirection = numel(quantiles)*numel(scales);
    nDirection = ceil(nCandidateTarget/nPerDirection);
    if ensureFullSpan
        nDirection = max(nDirection,d);
    end
    nBlock = ceil(nDirection/max(d,1));

    U = zeros(d,nBlock*d);
    stream = RandStream('mt19937ar','Seed',max(0,mod(seed,2^31-1)));
    for b = 1:nBlock
        G = randn(stream,d,d);
        [Q,R] = qr(G,0);
        signs = sign(diag(R));
        signs(signs==0) = 1;
        Q = Q*diag(signs);
        U(:,(b-1)*d+(1:d)) = Q;
    end
    U = U(:,1:nDirection);

    nActual = nDirection*nPerDirection;
    values = zeros(nSamples,nActual);
    rawW = zeros(d,nActual);
    rawB = zeros(1,nActual);
    directionIndex = zeros(1,nActual);
    quantileValue = zeros(1,nActual);
    activationScale = zeros(1,nActual);
    transitionLocation = zeros(1,nActual);
    directionalSpread = zeros(1,nActual);

    c = 0;
    for j = 1:nDirection
        u = U(:,j);
        z = u.'*Hn;
        sj = std(z,0,2);
        if ~isfinite(sj) || sj<stdFloor; sj = stdFloor; end
        for iq = 1:numel(quantiles)
            q = quantiles(iq);
            tj = empirical_quantile_local(z,q);
            for ia = 1:numel(scales)
                alpha = scales(ia);
                c = c+1;
                argument = alpha*(z-tj)/sj;
                values(:,c) = tanh(argument(:));
                factor = alpha/sj;
                rawW(:,c) = factor*(u./coordScale);
                rawB(c) = -factor*(u.'*(mu./coordScale)+tj);
                directionIndex(c) = j;
                quantileValue(c) = q;
                activationScale(c) = alpha;
                transitionLocation(c) = tj;
                directionalSpread(c) = sj;
            end
        end
    end

    colStd = std(values,0,1);
    valid = find(isfinite(colStd) & colStd>=varThreshold & all(isfinite(values),1));
    if isempty(valid)
        valid = 1:size(values,2);
    end

    % Repeated orthogonal blocks can generate the same one-dimensional
    % direction. Remove exact symbolic duplicates before correlation screening
    % so every retained dictionary column has a distinct fixed parameter set.
    candidateTermNames = cell(1,nActual);
    for kCandidate = 1:nActual
        candidateTermNames{kCandidate} = ridge_term_text_local( ...
            activation,rawW(:,kCandidate),rawB(kCandidate));
    end
    [~,uniqueLocal] = unique(candidateTermNames(valid),'stable');
    valid = valid(uniqueLocal);

    corrKeep = zeros(1,0);
    for idx = valid
        if isempty(corrKeep)
            corrKeep(end+1) = idx; %#ok<AGROW>
            continue;
        end
        x = values(:,idx);
        accept = true;
        for kept = corrKeep
            y = values(:,kept);
            denom = norm(x-mean(x))*norm(y-mean(y));
            if denom <= eps
                rho = 1;
            else
                rho = abs(((x-mean(x)).'*(y-mean(y)))/denom);
            end
            if rho > corrThreshold
                accept = false;
                break;
            end
        end
        if accept
            corrKeep(end+1) = idx; %#ok<AGROW>
        end
    end
    strictCorrelationCount = numel(corrKeep);
    correlationScreeningRelaxed = false;
    if numel(corrKeep)<nTarget
        % The correlation threshold is a diversity preference, not a reason to
        % reject an otherwise valid Stage-0 structure. This situation is common
        % on one-dimensional hidden branches, where many useful translated tanh
        % features are necessarily strongly correlated. Restore the remaining
        % finite candidates and let column-pivoted selection choose the most
        % informative subset.
        correlationScreeningRelaxed = true;
        corrKeep = unique([corrKeep,valid],'stable');
    end
    if numel(corrKeep)<nTarget
        error(['Fixed neural-ridge candidate pool contains only %d finite unique ', ...
            'candidates for a target of %d. Increase neuralPoolRatio or enlarge ', ...
            'the quantile/scale design.'],numel(corrKeep),nTarget);
    end

    A = values(:,corrKeep);
    A = A-mean(A,1);
    sigma = std(A,0,1);
    sigma(~isfinite(sigma) | sigma<stdFloor) = 1;
    A = A./sigma;
    pivot = pivot_order_local(A);
    if numel(pivot)<nTarget
        error(['Fixed neural-ridge pivoting returned only %d candidate indices ', ...
            'for a target of %d.'],numel(pivot),nTarget);
    end
    selected = corrKeep(pivot(1:nTarget));
    if numel(selected) ~= nTarget || numel(unique(selected)) ~= nTarget
        error(['Fixed neural-ridge pivoting returned a non-permutation selection ', ...
            'for a target of %d unique candidates.'],nTarget);
    end

    terms = reshape(candidateTermNames(selected),[],1);
    if numel(unique(terms,'stable')) ~= nTarget
        error('Fixed neural-ridge term serialization produced duplicate basis names.');
    end

    selectedValues = values(:,selected);
    centeredValues = selectedValues-mean(selectedValues,1);
    numericalRank = rank(centeredValues);
    C = corrcoef(selectedValues);
    if isempty(C) || size(C,1)<=1
        maxAbsOffDiagonalCorrelation = 0;
    else
        C(1:size(C,1)+1:end) = 0;
        maxAbsOffDiagonalCorrelation = max(abs(C(:)));
    end
    if ~isfinite(maxAbsOffDiagonalCorrelation)
        maxAbsOffDiagonalCorrelation = 1;
    end
    % A selected bank may exceed the preferred pairwise-correlation threshold
    % after the controlled fallback above. Numerical rank remains the hard
    % safeguard; the realized correlation is retained in metadata for reporting.
    if numericalRank < min(nTarget,size(centeredValues,1)-1)
        error('Selected fixed neural-ridge feature matrix is rank deficient: rank %d of %d.', ...
            numericalRank,nTarget);
    end

    meta = struct();
    meta.inputDim = d;
    meta.nSamples = nSamples;
    meta.targetCount = nTarget;
    meta.retainedCount = numel(selected);
    meta.candidateTarget = nCandidateTarget;
    meta.actualCandidateCount = nActual;
    meta.directionCount = nDirection;
    meta.orthogonalBlockCount = nBlock;
    meta.quantiles = quantiles;
    meta.activationScales = scales;
    meta.normalizationMean = mu;
    meta.normalizationScale = coordScale;
    meta.selectedCandidateIndices = selected;
    meta.selectedDirectionIndices = directionIndex(selected);
    meta.selectedQuantiles = quantileValue(selected);
    meta.selectedActivationScales = activationScale(selected);
    meta.selectedTransitionLocations = transitionLocation(selected);
    meta.selectedDirectionalSpreads = directionalSpread(selected);
    meta.selectedRawWeights = rawW(:,selected);
    meta.selectedRawBiases = rawB(selected);
    meta.termNames = terms;
    meta.selectedFeatureNumericalRank = numericalRank;
    meta.selectedFeatureMaxAbsCorrelation = maxAbsOffDiagonalCorrelation;
    meta.requestedCorrelationThreshold = corrThreshold;
    meta.strictCorrelationScreenedCount = strictCorrelationCount;
    meta.correlationScreeningRelaxed = correlationScreeningRelaxed;
end

function qv = empirical_quantile_local(x,q)
    x = sort(double(x(:)));
    n = numel(x);
    if n==0; qv = 0; return; end
    if n==1; qv = x(1); return; end
    pos = 1+(n-1)*q;
    lo = floor(pos); hi = ceil(pos);
    if lo==hi
        qv = x(lo);
    else
        qv = x(lo)+(pos-lo)*(x(hi)-x(lo));
    end
end

function order = pivot_order_local(A)
% Return a complete, duplicate-free column-pivot order.
%
% MATLAB releases differ in whether the third output of economy QR is a
% permutation matrix or a permutation vector, and some releases do not accept
% qr(A,0,'vector').  The previous implementation assumed one representation in
% its fallback path, which could collapse the order to repeated index 1.  This
% routine accepts either representation, validates it explicitly, and uses a
% deterministic pivoted Gram--Schmidt fallback when necessary.
    n = size(A,2);
    if n==0
        order = zeros(1,0);
        return;
    end

    order = zeros(1,0);
    try
        [~,~,P] = qr(A,0);
        order = permutation_output_to_vector_local(P,n);
    catch
        order = zeros(1,0);
    end

    if ~is_valid_permutation_local(order,n)
        order = greedy_column_pivot_order_local(A);
    end
    if ~is_valid_permutation_local(order,n)
        error('Unable to construct a valid duplicate-free neural-feature pivot order.');
    end
end

function order = permutation_output_to_vector_local(P,n)
    order = zeros(1,0);
    if isvector(P) && numel(P)==n
        candidate = reshape(double(P),1,[]);
    elseif ismatrix(P) && size(P,1)==n && size(P,2)==n
        [peak,candidate] = max(abs(double(P)),[],1);
        if any(~isfinite(peak)) || any(peak<0.5)
            return;
        end
        candidate = reshape(candidate,1,[]);
    else
        return;
    end
    candidate = round(candidate);
    if is_valid_permutation_local(candidate,n)
        order = candidate;
    end
end

function tf = is_valid_permutation_local(order,n)
    order = reshape(double(order),1,[]);
    tf = numel(order)==n && all(isfinite(order)) && ...
        all(order==round(order)) && isequal(sort(order),1:n);
end

function order = greedy_column_pivot_order_local(A)
% Deterministic modified Gram--Schmidt column pivoting used only when the QR
% permutation representation cannot be decoded on the active MATLAB release.
    A = double(A);
    [m,n] = size(A);
    order = zeros(1,n);
    remaining = true(1,n);
    residual = A;
    Q = zeros(m,min(m,n));
    nQ = 0;
    scale = max(1,norm(A,'fro'));
    tol = max(size(A))*eps(scale);

    for k = 1:n
        score = sum(abs(residual).^2,1);
        score(~remaining) = -Inf;
        [bestScore,idx] = max(score);
        if isempty(idx) || ~isfinite(bestScore)
            idx = find(remaining,1,'first');
        end
        order(k) = idx;
        remaining(idx) = false;

        v = residual(:,idx);
        if nQ>0
            v = v-Q(:,1:nQ)*(Q(:,1:nQ).'*v);
        end
        nv = norm(v);
        if isfinite(nv) && nv>tol
            nQ = nQ+1;
            q = v/nv;
            Q(:,nQ) = q;
            rem = find(remaining);
            if ~isempty(rem)
                residual(:,rem) = residual(:,rem)-q*(q.'*residual(:,rem));
            end
        end
    end
end

function text = ridge_term_text_local(activation,w,b)
    pieces = cell(1,0);
    for i = 1:numel(w)
        if abs(w(i)) <= 1e-15; continue; end
        pieces{end+1} = sprintf('(%s)*v%d',number_text_local(w(i)),i); %#ok<AGROW>
    end
    if isempty(pieces)
        affine = number_text_local(b);
    else
        affine = strjoin(pieces,'+');
        if abs(b)>1e-15
            affine = [affine '+(' number_text_local(b) ')']; %#ok<AGROW>
        end
    end
    text = sprintf('%s(%s)',activation,affine);
end

function text = number_text_local(x)
    text = sprintf('%.17g',double(x));
end
