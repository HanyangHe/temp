function arch = make_sindy_neural_arch(task,opts,baseArch,Xtr)
%MAKE_SINDY_NEURAL_ARCH Build the matched Lorenz--96 Neural-SINDy dictionary.
%
% Exact library:
%   1 + raw inputs + N fixed neural-ridge bases + unary(raw inputs).
% No polynomial powers, unary-on-polynomial terms, or variable-operator cross
% terms are admitted. Neural terms are never silently deduplicated: the
% builder requires exactly N distinct symbolic and numerical columns.

    if nargin < 2 || isempty(opts); opts = sindy_default_options(); end
    if nargin < 3 || isempty(baseArch); baseArch = struct(); end
    if nargin < 4 || isempty(Xtr)
        error('Neural-SINDy requires the physical training inputs Xtr.');
    end
    validateattributes(Xtr,{'numeric'},{'2d','real','finite','nonempty'},mfilename,'Xtr');
    if size(Xtr,2) ~= task.nx
        error('Neural-SINDy Xtr has %d columns; task.nx=%d.',size(Xtr,2),task.nx);
    end

    arch = struct();
    arch.nx = task.nx;
    arch.ny = task.ny;
    arch.layer = 1;
    arch.hiddenDims = [];
    arch.operatorMode = 'true';
    arch.dictionaryMode = 'sindy_independent_neural_matched';
    arch.branchActiveMask = true(1,1);
    if isstruct(baseArch) && isfield(baseArch,'safety') && ~isempty(baseArch.safety)
        arch.safety = baseArch.safety;
    else
        arch.safety = struct('eps',1e-8);
    end
    if ~isfield(arch.safety,'eps') || isempty(arch.safety.eps)
        arch.safety.eps = 1e-8;
    end
    if isstruct(baseArch) && isfield(baseArch,'feasibility') && ~isempty(baseArch.feasibility)
        arch.feasibility = baseArch.feasibility;
    end

    nTarget = max(1,round(getfield_default_local(opts,'neuralCount',task.nx)));
    activation = lower(strtrim(char(getfield_default_local(opts,'neuralActivation','tanh'))));
    quantiles = getfield_default_local(opts,'neuralQuantiles',[0.25,0.50,0.75]);
    scales = getfield_default_local(opts,'neuralScales',[0.5,1,2]);
    poolRatio = getfield_default_local(opts,'neuralPoolRatio',3);
    seed = getfield_default_local(opts,'neuralSeed',11886);
    stdFloor = getfield_default_local(opts,'neuralStdFloor',1e-10);
    varianceThreshold = getfield_default_local(opts,'neuralVarianceThreshold',1e-8);
    correlationThreshold = getfield_default_local(opts,'neuralCorrelationThreshold',0.995);
    ensureFullSpan = getfield_default_local(opts,'neuralEnsureFullDirectionalSpan',true);

    [neuralTerms,neuralMeta] = make_fixed_neural_ridge_terms( ...
        Xtr.',nTarget,activation,quantiles,scales,poolRatio,seed,stdFloor, ...
        varianceThreshold,correlationThreshold,ensureFullSpan);
    if numel(neuralTerms) ~= nTarget
        error('Neural-SINDy generated %d neural terms; expected exactly %d.', ...
            numel(neuralTerms),nTarget);
    end
    assert_unique_terms_local(neuralTerms,'neural-ridge');

    % The first-order raw-input basis is mandatory. Neural ridges replace only
    % the degree-two polynomial block; they do not replace x1,...,xn.
    rawTerms = arrayfun(@(k) sprintf('v%d',k),1:task.nx,'UniformOutput',false).';
    if numel(rawTerms) ~= task.nx
        error('Neural-SINDy raw-input basis construction failed: got %d terms, expected %d.', ...
            numel(rawTerms),task.nx);
    end
    unaryOps = normalize_cellstr_local(getfield_default_local(opts,'unaryOperators',{'sqrt'}));
    unaryTerms = cell(0,1);
    for i = 1:numel(unaryOps)
        op = lower(strtrim(char(unaryOps{i})));
        if isempty(op); continue; end
        for k = 1:task.nx
            unaryTerms{end+1,1} = sprintf('%s(v%d)',op,k); %#ok<AGROW>
        end
    end

    % Do not call a silent unique() operation here. Any duplicate indicates a
    % construction error and must stop the run rather than erase neural bases.
    terms = [{'1'};rawTerms(:);neuralTerms(:);unaryTerms(:)];
    assert_unique_terms_local(terms,'complete Neural-SINDy');
    if ~all(ismember(rawTerms,terms))
        error('Neural-SINDy dictionary lost one or more mandatory raw-input terms x1,...,xn.');
    end

    stage0GuessTerms = normalize_cellstr_local(getfield_default_local( ...
        opts,'stage0InitialGuessTerms',{}));
    [terms,stage0GuessUnion] = union_stage0_guess_terms_local(terms,stage0GuessTerms);

    expectedSize = getfield_default_local(opts,'expectedLibrarySize',[]);
    if logical(getfield_default_local(opts,'strictLibraryAssertions',false)) && ...
            ~isempty(expectedSize) && numel(terms) ~= expectedSize
        error('Neural-SINDy dictionary size mismatch: constructed %d, expected %d.', ...
            numel(terms),expectedSize);
    end
    expectedNeural = getfield_default_local(opts,'expectedNeuralCount',[]);
    if logical(getfield_default_local(opts,'strictLibraryAssertions',false)) && ...
            ~isempty(expectedNeural) && numel(neuralTerms) ~= expectedNeural
        error('Neural-SINDy neural-count mismatch: constructed %d, expected %d.', ...
            numel(neuralTerms),expectedNeural);
    end

    D = struct();
    D.caseId = getfield_default_local(task,'name','Neural_SINDy');
    D.termsByDim = cell(1,max(1,task.nx));
    D.termsByDim{task.nx} = terms;
    D.noFallback = true;
    D.appendGlobalTerms = false;
    D.source = sprintf(['matched Neural-SINDy dictionary: constant + %d raw inputs + ', ...
        '%d distinct fixed %s ridge bases + %d direct-input unary terms {%s}'], ...
        task.nx,numel(neuralTerms),activation,numel(unaryTerms),strjoin(unaryOps,','));
    if stage0GuessUnion.enabled
        D.source = sprintf('%s + union(Stage0SRInitialGuesses: requested=%d, added=%d, duplicate=%d)', ...
            D.source,stage0GuessUnion.nRequested,stage0GuessUnion.nAdded, ...
            stage0GuessUnion.nDuplicate);
    end
    D.stage0InitialGuessUnion = stage0GuessUnion;
    D.neuralSindy = struct('enabled',true,'neuralTerms',{neuralTerms}, ...
        'parameters',neuralMeta,'replacesDegreeTwoPolynomialTerms',true, ...
        'includeRawInputs',true,'rawInputTerms',{rawTerms}, ...
        'inputSpace','raw_physical_state','sharedGenerator','make_fixed_neural_ridge_terms');
    arch.caseDictionary = D;
    arch.sindyDictionaryReport = struct( ...
        'version','matched_neural_sindy_v2', ...
        'nTerms',numel(terms), ...
        'rawInputCount',task.nx, ...
        'includeRawInputs',true, ...
        'rawInputTerms',{rawTerms}, ...
        'includePolynomialTerms',false, ...
        'polynomialTermsReplacedByNeural',true, ...
        'neuralCount',numel(neuralTerms), ...
        'neuralActivation',activation, ...
        'neuralTerms',{neuralTerms}, ...
        'neuralParameters',neuralMeta, ...
        'unaryOperators',{unaryOps}, ...
        'directInputUnaryCount',numel(unaryTerms), ...
        'includeUnaryOnMonomials',false, ...
        'includeOperatorCrossTerms',false, ...
        'stage0InitialGuessUnion',stage0GuessUnion, ...
        'source',D.source);
end

function assert_unique_terms_local(terms,label)
    keys = cellfun(@canonical_term_key_local,terms(:),'UniformOutput',false);
    [~,ia] = unique(keys,'stable');
    if numel(ia) ~= numel(keys)
        duplicateIdx = setdiff(1:numel(keys),ia,'stable');
        error('%s dictionary contains %d duplicate term name(s); first duplicate: %s.', ...
            label,numel(duplicateIdx),char(terms{duplicateIdx(1)}));
    end
end

function [out,info] = union_stage0_guess_terms_local(baseTerms,guessTerms)
    out = baseTerms(:);
    info = struct('enabled',~isempty(guessTerms),'requestedTerms',{guessTerms(:)}, ...
        'nRequested',numel(guessTerms),'nAdded',0,'nDuplicate',0, ...
        'addedTerms',{{}},'duplicateTerms',{{}}, ...
        'libraryIndices',zeros(numel(guessTerms),1), ...
        'libraryTerms',{cell(numel(guessTerms),1)});
    keys = cellfun(@canonical_term_key_local,out,'UniformOutput',false);
    for k = 1:numel(guessTerms)
        term = strrep(strtrim(char(guessTerms{k})),' ','');
        key = canonical_term_key_local(term);
        idx = find(strcmp(keys,key),1);
        if isempty(idx)
            out{end+1,1} = term; %#ok<AGROW>
            keys{end+1,1} = key; %#ok<AGROW>
            idx = numel(out);
            info.nAdded = info.nAdded+1;
            info.addedTerms{end+1,1} = term;
        else
            info.nDuplicate = info.nDuplicate+1;
            info.duplicateTerms{end+1,1} = term;
        end
        info.libraryIndices(k) = idx;
        info.libraryTerms{k} = out{idx};
    end
end

function key = canonical_term_key_local(term)
    key = lower(strrep(strtrim(char(term)),' ',''));
    key = strrep(key,'**','^');
    key = regexprep(key,'square\((v[0-9]+)\)','$1^2');
    key = regexprep(key,'sqr\((v[0-9]+)\)','$1^2');
    key = strip_outer_parentheses_local(key);
end

function out = strip_outer_parentheses_local(in)
    out = in;
    changed = true;
    while changed && numel(out)>=2 && out(1)=='(' && out(end)==')'
        changed = false; depth = 0; enclosesAll = true;
        for k = 1:numel(out)
            if out(k)=='('; depth=depth+1;
            elseif out(k)==')'
                depth=depth-1;
                if depth==0 && k<numel(out); enclosesAll=false; break; end
            end
        end
        if enclosesAll && depth==0; out=out(2:end-1); changed=true; end
    end
end

function out = normalize_cellstr_local(value)
    if isempty(value); out = {}; return; end
    if ischar(value); out = {value};
    elseif isstring(value); out = cellstr(value(:));
    elseif iscell(value); out = value(:);
    else; error('Expected a character vector, string array, or cell array.');
    end
    keep = true(size(out));
    for k = 1:numel(out)
        out{k} = strtrim(char(out{k}));
        keep(k) = ~isempty(out{k});
    end
    out = out(keep);
end

function value = getfield_default_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end
