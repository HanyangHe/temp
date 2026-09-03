function [arch, info] = attach_fixed_neural_ridge_augmentation(arch, seedCoef, Xtr, normOpt, cfg)
%ATTACH_FIXED_NEURAL_RIDGE_AUGMENTATION Add fixed data-aware ridge features.
%
% The structurally compiled Stage-0 seed is evaluated first. For every active
% branch, the optional deterministic linear-coordinate passthrough and N
% fixed tanh ridge features are generated from the branch-state samples.
% Their internal projection/location/scale parameters remain frozen;
% only the outer PhDN coefficients are trainable. The generated ridge terms
% are folded back into the raw branch coordinates and stored as ordinary
% explicit dictionary terms, preserving the existing PhDN evaluator path.

    if nargin < 5 || isempty(cfg); cfg = struct(); end
    if nargin < 4 || isempty(normOpt); normOpt = default_norm_options(); end
    if isempty(Xtr) || ~isnumeric(Xtr) || size(Xtr,2) ~= arch.nx
        error('Neural augmentation requires finite training inputs with %d columns.', arch.nx);
    end
    if any(~isfinite(Xtr(:)))
        error('Neural augmentation training inputs contain nonfinite values.');
    end
    if ~isfield(arch,'caseDictionary') || ~isstruct(arch.caseDictionary)
        error('The compiled architecture has no caseDictionary metadata.');
    end

    D = arch.caseDictionary;
    if ~isfield(D,'augmentationTermsByBlock') || ~iscell(D.augmentationTermsByBlock)
        error('The compiled architecture has no branch augmentation container.');
    end

    nTarget = get_cfg_local(cfg,'neuralCount',arch.nx);
    if isempty(nTarget); nTarget = arch.nx; end
    nTarget = max(1,round(nTarget));
    activation = lower(strtrim(char(get_cfg_local(cfg,'neuralActivation','tanh'))));
    if ~strcmp(activation,'tanh')
        error('Only tanh fixed neural-ridge augmentation is currently supported.');
    end
    quantiles = reshape(double(get_cfg_local(cfg,'neuralQuantiles',[0.25,0.50,0.75])),1,[]);
    scales = reshape(double(get_cfg_local(cfg,'neuralScales',[0.5,1,2])),1,[]);
    quantiles = unique(quantiles(isfinite(quantiles) & quantiles>0 & quantiles<1),'stable');
    scales = unique(scales(isfinite(scales) & scales>0),'stable');
    if isempty(quantiles) || isempty(scales)
        error('Neural quantile and activation-scale sets must be nonempty.');
    end

    poolRatio = max(1,double(get_cfg_local(cfg,'neuralPoolRatio',3)));
    seedBase = round(double(get_cfg_local(cfg,'neuralSeed',1701)));
    stdFloor = max(eps,double(get_cfg_local(cfg,'neuralStdFloor',1e-10)));
    varianceThreshold = max(0,double(get_cfg_local(cfg,'neuralVarianceThreshold',1e-8)));
    correlationThreshold = double(get_cfg_local(cfg,'neuralCorrelationThreshold',0.995));
    correlationThreshold = min(max(correlationThreshold,0),1);
    ensureFullSpan = logical(get_cfg_local(cfg,'neuralEnsureFullDirectionalSpan',true));
    includeLinearTerms = logical(get_cfg_local(cfg,'neuralIncludeLinearTerms',false));

    branchStates = forward_inner_states(Xtr,seedCoef,arch,normOpt);
    dims = get_arch_dims(arch);
    L = arch.layer;
    blockInfo = cell(L,L);
    totalRetained = 0;

    for ell = 1:L
        for src = 1:ell
            if ~branch_active_local(arch,src,ell)
                continue;
            end
            stateIndex = ell-src+1;
            H = branchStates{stateIndex};
            inputDim = dims(stateIndex);
            if size(H,1) ~= inputDim
                error('Branch-state dimension mismatch at branch (%d,%d).',src,ell);
            end

            branchSeed = seedBase + 1009*ell + 9176*src;
            [terms,meta] = make_fixed_neural_ridge_terms(H,nTarget,activation, ...
                quantiles,scales,poolRatio,branchSeed,stdFloor, ...
                varianceThreshold,correlationThreshold,ensureFullSpan);
            linearTerms = {};
            if includeLinearTerms
                linearTerms = arrayfun(@(k) sprintf('v%d',k),1:inputDim,'UniformOutput',false).';
            end
            D.augmentationTermsByBlock{src,ell} = ...
                unique_stable_local([{'1'};linearTerms(:);terms(:)]);
            blockInfo{src,ell} = meta;
            totalRetained = totalRetained + numel(terms);
        end
    end

    D.augmentationMode = 'fixed_neural_ridge';
    D.augmentationActivation = activation;
    D.augmentationNeuralCount = nTarget;
    D.augmentationNeuralQuantiles = quantiles;
    D.augmentationNeuralScales = scales;
    D.augmentationNeuralPoolRatio = poolRatio;
    D.augmentationNeuralIncludeLinearTerms = includeLinearTerms;
    D.augmentationNeuralParametersByBlock = blockInfo;
    D.dictionaryExpansionMode = 'uniform_fixed_neural_ridge';
    if includeLinearTerms
        D.source = sprintf(['Per-output selected SINDy/PySR core DAG with uniform ', ...
            'constant+all linear coordinates+%d fixed %s ridge augmentations per active branch'], ...
            nTarget,activation);
    else
        D.source = sprintf(['Per-output selected SINDy/PySR core DAG with uniform ', ...
            'constant+%d fixed %s ridge augmentations per active branch'],nTarget,activation);
    end
    arch.caseDictionary = D;
    arch.dictionaryMode = 'sr_structural_dag_plus_uniform_fixed_neural_ridge_augmentation';
    arch.stage0CompileMode = 'sr_structure_score_core_uniform_fixed_neural_ridge_augmented_dag';

    info = struct();
    info.mode = 'fixed_neural_ridge';
    info.activation = activation;
    info.targetPerActiveBranch = nTarget;
    info.totalRetained = totalRetained;
    info.quantiles = quantiles;
    info.scales = scales;
    info.poolRatio = poolRatio;
    info.includeLinearTerms = includeLinearTerms;
    info.seed = seedBase;
    info.blocks = blockInfo;
end

function [terms,meta] = make_branch_neural_terms_local(H,nTarget,activation, ...
        quantiles,scales,poolRatio,seed,stdFloor,varThreshold,corrThreshold,ensureFullSpan)
    d = size(H,1);
    nSamples = size(H,2);
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
                switch activation
                    case 'tanh'
                        values(:,c) = tanh(argument(:));
                end
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
        % Preserve a usable finite bank for a degenerate branch state. The
        % outer coefficients remain zero at initialization, so exact Stage-0
        % reproduction is unaffected.
        valid = 1:size(values,2);
    end

    % Greedy correlation screening provides an interpretable first pass. If
    % it becomes too aggressive, remaining finite candidates are restored and
    % the final column-pivoted QR performs the actual numerical selection.
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
    if numel(corrKeep)<nTarget
        corrKeep = unique([corrKeep,valid],'stable');
    end

    A = values(:,corrKeep);
    A = A-mean(A,1);
    sigma = std(A,0,1);
    sigma(~isfinite(sigma) | sigma<stdFloor) = 1;
    A = A./sigma;
    pivot = pivot_order_local(A);
    selected = corrKeep(pivot(1:min(nTarget,numel(pivot))));
    if numel(selected)<nTarget
        remaining = setdiff(1:size(values,2),selected,'stable');
        selected = [selected,remaining(1:min(nTarget-numel(selected),numel(remaining)))]; %#ok<AGROW>
    end
    selected = selected(1:min(nTarget,numel(selected)));

    terms = cell(numel(selected),1);
    for k = 1:numel(selected)
        terms{k} = ridge_term_text_local(activation,rawW(:,selected(k)),rawB(selected(k)));
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
    n = size(A,2);
    if n==0; order = zeros(1,0); return; end
    try
        [~,~,order] = qr(A,0,'vector');
        order = reshape(order,1,[]);
    catch
        [~,~,P] = qr(A,0);
        [~,order] = max(P,[],1);
        order = reshape(order,1,[]);
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

function tf = branch_active_local(arch,src,ell)
    tf = true;
    if isfield(arch,'branchActiveMask') && ~isempty(arch.branchActiveMask)
        M = arch.branchActiveMask;
        if isnumeric(M) || islogical(M)
            if size(M,1)>=src && size(M,2)>=ell
                tf = logical(M(src,ell));
            end
        elseif iscell(M) && size(M,1)>=src && size(M,2)>=ell && ~isempty(M{src,ell})
            tf = logical(M{src,ell});
        end
    end
end

function value = get_cfg_local(cfg,name,defaultValue)
    if isstruct(cfg) && isfield(cfg,name) && ~isempty(cfg.(name))
        value = cfg.(name);
    else
        value = defaultValue;
    end
end

function out = unique_stable_local(in)
    out = cell(0,1);
    for i = 1:numel(in)
        value = char(in{i});
        if ~any(strcmp(out,value))
            out{end+1,1} = value; %#ok<AGROW>
        end
    end
end
