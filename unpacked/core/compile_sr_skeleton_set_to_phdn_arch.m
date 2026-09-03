function [archCand, compileInfo] = compile_sr_skeleton_set_to_phdn_arch(coreTerms, varargin)
%COMPILE_SR_SKELETON_SET_TO_PHDN_ARCH Compile one selected symbolic core per output.
%
% New call:
%   compile_sr_skeleton_set_to_phdn_arch(coreTerms, archBase, task, modelRow, cfg)
%
% For backward source compatibility, the former six-input call is accepted;
% its second (simple-expression) argument is ignored.  Only coreTerms are
% compiled.  Each core expression is recursively decomposed into an operator
% tree, identical subexpressions are merged through the compiler node cache,
% and all outputs are aligned into one shared directed acyclic graph (DAG).
% PySR-required nonlinear operators are stored as exact structural channels.
% Every active branch receives the same dimension-dependent augmentation
% family. The default is constant plus total-degree Poly_2. An optional fixed
% neural-ridge mode builds data-aware tanh features from the Stage-0 branch
% states. Structural coefficients reproduce the selected SR core exactly and
% all added augmentation coefficients are initialized to zero.

    if numel(varargin) == 4
        archBase = varargin{1};
        task = varargin{2};
        modelRow = varargin{3};
        cfg = varargin{4};
    elseif numel(varargin) == 5
        % Legacy best/simple signature. The former simple tree is deliberately
        % ignored under the single-core Stage-0 policy.
        archBase = varargin{2};
        task = varargin{3};
        modelRow = varargin{4};
        cfg = varargin{5};
    else
        error(['compile_sr_skeleton_set_to_phdn_arch expects either ', ...
            '5 inputs (coreTerms, archBase, task, modelRow, cfg) or the ', ...
            'legacy 6-input form.']);
    end
    if isempty(cfg); cfg = struct(); end
    if ~iscell(coreTerms) || numel(coreTerms) ~= task.ny
        error('coreTerms must contain exactly one selected expression per output.');
    end

    compiler = init_compiler_state_local(task, archBase, cfg);
    coreRoots = cell(1,task.ny);
    coreRows = cell(1,task.ny);
    normalizedCoreTerms = cell(1,task.ny);
    originalCoreTerms = cell(1,task.ny);
    variableIndexBase = detect_stage0_variable_index_base_local(coreTerms, task, modelRow);
    maxSrcLayer = 1;
    expressionDepth = 1;
    for r = 1:task.ny
        originalCoreTerms{r} = normalize_stage0_term_local(coreTerms{r});
        normalizedCoreTerms{r} = normalize_stage0_term_local(coreTerms{r}, task, variableIndexBase);
        coreRoots{r} = parse_stage0_expr_local(normalizedCoreTerms{r});
        [coreRows{r}, compiler] = compile_linear_row_local(coreRoots{r}, compiler);
        maxSrcLayer = max(maxSrcLayer, max_source_layer_local(coreRows{r}));
        expressionDepth = max(expressionDepth, ast_depth_local(coreRoots{r}));
    end

    outputEll = max(1, maxSrcLayer);
    compiler = ensure_state_layer_local(compiler, outputEll); %#ok<NASGU>

    hiddenDims = zeros(1, max(outputEll - 1, 0));
    for s = 2:outputEll
        hiddenDims(s-1) = numel(compiler.stateNodes{s});
        if hiddenDims(s-1) < 1; hiddenDims(s-1) = 1; end
    end

    archCand = archBase;
    archCand.layer = outputEll;
    archCand.hiddenDims = hiddenDims;
    if isfield(archCand, 'dims'); archCand = rmfield(archCand, 'dims'); end
    archCand.nx = task.nx;
    archCand.ny = task.ny;
    archCand.operatorMode = 'true';
    archCand.branchActiveMask = false(outputEll, outputEll);
    archCand.branchActiveMode = 'sr_structure_score_core_augmented_dag';
    augmentationMode = normalize_augmentation_mode_local(cfg);
    if strcmp(augmentationMode,'fixed_neural_ridge')
        archCand.dictionaryMode = 'sr_structural_dag_plus_uniform_fixed_neural_ridge_augmentation';
        archCand.stage0CompileMode = 'sr_structure_score_core_uniform_fixed_neural_ridge_augmented_dag';
    else
        archCand.dictionaryMode = 'sr_structural_dag_plus_uniform_poly_augmentation';
        archCand.stage0CompileMode = 'sr_structure_score_core_uniform_poly_augmented_dag';
    end

    D = make_empty_dictionary_local(task, modelRow, strjoin(normalizedCoreTerms,'; '), 1, cfg);
    D.compileMode = archCand.stage0CompileMode;
    D.expressionDepth = expressionDepth;
    D.coreOperatorTreeRoots = coreRoots;
    D.bestOperatorTreeRoots = coreRoots; % backward-compatible alias
    D.decomposition = compiler.decomposition;
    D.unifiedCompactPrior = compiler.unifiedCompactPrior;
    D.stage0Expressions = originalCoreTerms;
    D.stage0CoreExpressions = originalCoreTerms;
    D.stage0BestScoreExpressions = originalCoreTerms;
    D.stage0CompilerExpressions = normalizedCoreTerms;
    D.stage0VariableAliases = stage0_variable_alias_map_local(task);
    D.stage0VariableIndexBase = variableIndexBase;
    D.structuralOperatorSemantics = 'official_pysr_raw';
    D.augmentationOperatorSemantics = 'phdn_protected';
    D.augmentationMode = augmentationMode;
    D.source = augmentation_source_local(augmentationMode, cfg);

    L = outputEll;
    D.structuralTermsByBlock = cell(L, L);
    D.augmentationTermsByBlock = cell(L, L);
    D.stage0SeedRowTerms = cell(L, L);
    D.stage0SeedRowCoef = cell(L, L);

    % Hidden DAG nodes.  The recursive compiler cache merges identical
    % subexpressions; each generated node is materialized once and may feed
    % multiple later branches or outputs.
    for s = 2:outputEll
        ell = s - 1;
        rowDim = numel(compiler.stateNodes{s});
        for srcState = 1:(s-1)
            src = ell - srcState + 1;
            exactRows = repmat({{}}, rowDim, 1);
            exactCoefs = repmat({[]}, rowDim, 1);
            hasAny = false;
            for rr = 1:rowDim
                contrib = compiler.stateNodes{s}{rr}.row;
                [termsR, coefsR] = row_terms_for_source_local(contrib, srcState);
                if ~isempty(termsR)
                    exactRows{rr} = termsR;
                    exactCoefs{rr} = coefsR;
                    hasAny = true;
                end
            end
            if hasAny
                inputDim = state_dim_local(compiler, srcState);
                structuralTerms = unique_terms_from_row_spec_local(exactRows);
                augmentationTerms = build_augmentation_terms_local(inputDim, compiler.unifiedCompactPrior, cfg);
                D.structuralTermsByBlock{src, ell} = structuralTerms;
                D.augmentationTermsByBlock{src, ell} = augmentationTerms;
                D.stage0SeedRowTerms{src, ell} = exactRows;
                D.stage0SeedRowCoef{src, ell} = exactCoefs;
                archCand.branchActiveMask(src, ell) = true;
            end
        end
    end

    % Output readout contains only the selected structure-score core path.  No
    % second/simple alternative is inserted into the DAG or parameter space.
    for srcState = 1:outputEll
        ell = outputEll;
        src = ell - srcState + 1;
        allowedRows = repmat({{}}, task.ny, 1);
        seedRows = repmat({[]}, task.ny, 1);
        hasAny = false;
        for r = 1:task.ny
            [coreR, coreC] = row_terms_for_source_local(coreRows{r}, srcState);
            if ~isempty(coreR)
                allowedRows{r} = coreR;
                seedRows{r} = coreC;
                hasAny = true;
            end
        end
        if hasAny
            inputDim = state_dim_local(compiler, srcState);
            structuralTerms = unique_terms_from_row_spec_local(allowedRows);
            augmentationTerms = build_augmentation_terms_local(inputDim, compiler.unifiedCompactPrior, cfg);
            D.structuralTermsByBlock{src, ell} = structuralTerms;
            D.augmentationTermsByBlock{src, ell} = augmentationTerms;
            D.stage0SeedRowTerms{src, ell} = allowedRows;
            D.stage0SeedRowCoef{src, ell} = seedRows;
            archCand.branchActiveMask(src, ell) = true;
        end
    end

    D.termsByDim = {};
    D.noFallback = true;
    D.appendGlobalTerms = false;
    if strcmp(augmentationMode,'fixed_neural_ridge')
        D.dictionaryExpansionMode = 'uniform_fixed_neural_ridge_pending_data_attachment';
    else
        D.dictionaryExpansionMode = 'uniform_constant_plus_total_degree_polynomial';
    end
    D.unifiedUnaryOperators = {};
    D.maxPolynomialOrder = compiler.unifiedCompactPrior.polyOrder;
    D.includeProducts = compiler.unifiedCompactPrior.includeProducts;
    archCand.caseDictionary = D;
    archCand.stage0Candidate = modelRow;
    archCand.srSkeletonSet = struct( ...
        'coreExpressions',{originalCoreTerms}, ...
        'bestScoreExpressions',{originalCoreTerms}, ...
        'bestExpressions',{originalCoreTerms});

    % Build a structural seed first. In neural-ridge mode, this seed is used
    % to evaluate branch-state samples before the fixed neural terms are added.
    seedCoef = make_seed_coef_from_row_seed_local(archCand);
    validate_seed_coef_size_local(seedCoef, archCand);
    neuralAugmentationInfo = struct();
    if strcmp(augmentationMode,'fixed_neural_ridge') && ...
            logical(get_struct_field_local(cfg,'buildNeuralAugmentation',true))
        XtrAug = get_struct_field_local(cfg,'trainingInputs',[]);
        normAug = get_struct_field_local(cfg,'normOpt',[]);
        if isempty(XtrAug)
            error(['Fixed neural-ridge augmentation requires cfg.trainingInputs ', ...
                'for branch-state construction.']);
        end
        neuralCfg = struct();
        neuralCfg.neuralCount = get_struct_field_local(cfg,'neuralCount',task.nx);
        neuralCfg.neuralActivation = get_struct_field_local(cfg,'neuralActivation','tanh');
        neuralCfg.neuralQuantiles = get_struct_field_local(cfg,'neuralQuantiles',[0.25,0.50,0.75]);
        neuralCfg.neuralScales = get_struct_field_local(cfg,'neuralScales',[0.5,1,2]);
        neuralCfg.neuralPoolRatio = get_struct_field_local(cfg,'neuralPoolRatio',3);
        neuralCfg.neuralSeed = get_struct_field_local(cfg,'neuralSeed',1701);
        neuralCfg.neuralStdFloor = get_struct_field_local(cfg,'neuralStdFloor',1e-10);
        neuralCfg.neuralVarianceThreshold = get_struct_field_local(cfg,'neuralVarianceThreshold',1e-8);
        neuralCfg.neuralCorrelationThreshold = get_struct_field_local(cfg,'neuralCorrelationThreshold',0.995);
        neuralCfg.neuralEnsureFullDirectionalSpan = get_struct_field_local(cfg,'neuralEnsureFullDirectionalSpan',true);
        neuralCfg.neuralIncludeLinearTerms = get_struct_field_local(cfg,'neuralIncludeLinearTerms',false);
        [archCand, neuralAugmentationInfo] = attach_fixed_neural_ridge_augmentation( ...
            archCand, seedCoef, XtrAug, normAug, neuralCfg);
        % Repack the same exact structural seed into the enlarged dictionaries.
        seedCoef = make_seed_coef_from_row_seed_local(archCand);
        validate_seed_coef_size_local(seedCoef, archCand);
        D = archCand.caseDictionary;
    end
    archCand.stage0SeedCoef = seedCoef;

    compileInfo = struct();
    compileInfo.compileMode = archCand.stage0CompileMode;
    compileInfo.expressionDepth = expressionDepth;
    compileInfo.nLayers = archCand.layer;
    compileInfo.hiddenDims = hiddenDims;
    compileInfo.compiledTerms = normalizedCoreTerms;
    compileInfo.originalTerms = originalCoreTerms;
    compileInfo.coreOriginalTerms = originalCoreTerms;
    compileInfo.variableAliases = stage0_variable_alias_map_local(task);
    compileInfo.variableIndexBase = variableIndexBase;
    compileInfo.nIntermediateNodes = count_intermediate_nodes_local(compiler);
    compileInfo.nExactActiveTerms = count_seed_nonzeros_local(seedCoef);
    compileInfo.unifiedCompactPrior = compiler.unifiedCompactPrior;
    compileInfo.augmentationMode = augmentationMode;
    compileInfo.neuralAugmentation = neuralAugmentationInfo;
    compileInfo.structuralOperatorSemantics = D.structuralOperatorSemantics;
    compileInfo.augmentationOperatorSemantics = D.augmentationOperatorSemantics;
    compileInfo.sharedSubexpressions = shared_subexpressions_local(compiler);
    compileInfo.nSharedSubexpressions = numel(compileInfo.sharedSubexpressions);
    D.sharedSubexpressions = compileInfo.sharedSubexpressions;
    archCand.caseDictionary = D;
    if strcmp(augmentationMode,'fixed_neural_ridge')
        augmentationSummary = sprintf('constant plus %d fixed neural-ridge bases', ...
            get_struct_field_local(cfg,'neuralCount',task.nx));
    else
        augmentationSummary = sprintf('constant plus total-degree Poly_%d bases', ...
            get_struct_field_local(cfg,'polyOrder',2));
    end
    compileInfo.summary = sprintf(['One selected symbolic core per output is recursively ', ...
        'decomposed into a shared compact PhDN DAG; identical subexpressions are ', ...
        'reused, SR operators remain exact structural channels, and every active ', ...
        'branch receives zero-initialized %s.'], augmentationSummary);
end

% =========================================================================
% Compiler state and recursive SINDy-branch decomposition
% =========================================================================
function compiler = init_compiler_state_local(task, archBase, cfg)
    compiler = struct();
    compiler.nx = task.nx;
    compiler.ny = task.ny;
    compiler.cfg = cfg;
    compiler.archBase = archBase;
    compiler.stateNodes = cell(1, 1);
    compiler.stateNodes{1} = {}; % state 1 is raw input x and has no generated nodes.
    compiler.nodeKeyMap = containers.Map('KeyType', 'char', 'ValueType', 'any');
    compiler.nodeUseCount = containers.Map('KeyType', 'char', 'ValueType', 'double');
    compiler.nodeKeyOrder = {};
    compiler.decomposition = struct('stateLayer', {}, 'rowIndex', {}, 'expr', {}, 'rowSummary', {});
    compiler.unifiedCompactPrior = init_unified_prior_local(archBase, cfg);
end

function prior = init_unified_prior_local(archBase, cfg)
    prior = struct();
    prior.augmentationMode = normalize_augmentation_mode_local(cfg);
    prior.polyOrder = get_struct_field_local(cfg, 'polyOrder', get_struct_field_local(archBase, 'polyOrder', 2));
    if isempty(prior.polyOrder) || ~isfinite(prior.polyOrder)
        prior.polyOrder = 2;
    end
    prior.polyOrder = max(1, round(prior.polyOrder));
    % SR-specific nonlinear operators remain in the structural DAG and are
    % never copied into either uniform augmentation family.
    prior.unaryOperators = {};
    prior.includeProducts = true;
    prior.includeConstant = true;
    prior.maxPowerObserved = 1;
    prior.exactExtraTerms = {};
end

function [row, compiler] = compile_linear_row_local(node, compiler)
    switch node.type
        case 'num'
            row = make_row_from_term_local(1, '1', node.value);
        case 'var'
            row = make_row_from_term_local(1, sprintf('v%d', node.value), 1);
        case 'unary'
            if strcmpi(node.op, 'neg')
                [row, compiler] = compile_linear_row_local(node.left, compiler);
                row = scale_row_local(row, -1);
            else
                [termSpec, compiler] = compile_basis_term_local(node, compiler);
                row = make_row_from_term_local(termSpec.sourceLayer, termSpec.term, termSpec.coef);
            end
        case 'binary'
            if strcmp(node.op, '+')
                [a, compiler] = compile_linear_row_local(node.left, compiler);
                [b, compiler] = compile_linear_row_local(node.right, compiler);
                row = combine_rows_local(a, b, 1);
            elseif strcmp(node.op, '-')
                [a, compiler] = compile_linear_row_local(node.left, compiler);
                [b, compiler] = compile_linear_row_local(node.right, compiler);
                row = combine_rows_local(a, b, -1);
            else
                [termSpec, compiler] = compile_basis_term_local(node, compiler);
                row = make_row_from_term_local(termSpec.sourceLayer, termSpec.term, termSpec.coef);
            end
        otherwise
            error('Unsupported AST node type in linear-row compilation: %s', node.type);
    end
    row = simplify_row_local(row);
end

function [spec, compiler] = compile_basis_term_local(node, compiler)
    spec = struct('sourceLayer', 1, 'term', '1', 'coef', 1);
    switch node.type
        case 'num'
            spec.sourceLayer = 1;
            spec.term = '1';
            spec.coef = node.value;
        case 'var'
            spec.sourceLayer = 1;
            spec.term = sprintf('v%d', node.value);
            spec.coef = 1;
        case 'unary'
            op = canonical_unary_name_local(node.op);
            if strcmpi(op, 'neg')
                [spec, compiler] = compile_basis_term_local(node.left, compiler);
                spec.coef = -spec.coef;
                return;
            end
            [ref, compiler] = compile_as_ref_local(node.left, compiler);
            if ref.isConst
                spec.sourceLayer = 1;
                spec.term = '1';
                spec.coef = apply_unary_const_local(op, ref.value);
            else
                spec.sourceLayer = ref.stateLayer;
                spec.term = sprintf('%s(v%d)', op, ref.index);
                spec.coef = 1;
                compiler = register_unary_op_local(compiler, op);
            end
        case 'binary'
            switch node.op
                case '*'
                    [spec, compiler] = compile_product_like_local({node.left, node.right}, [], compiler);
                case '/'
                    denom = make_unary_node_local('inv', node.right, ['inv(' node.right.expr ')']);
                    [spec, compiler] = compile_product_like_local({node.left, denom}, [], compiler);
                    compiler = register_unary_op_local(compiler, 'inv');
                case '^'
                    if ~strcmp(node.right.type, 'num')
                        error('Only numeric powers are supported in compact PhDN-DAG translation.');
                    end
                    p = node.right.value;
                    if strcmp(node.left.type, 'num')
                        spec.sourceLayer = 1;
                        spec.term = '1';
                        spec.coef = node.left.value .^ p;
                    else
                        [ref, compiler] = compile_as_ref_local(node.left, compiler);
                        if ref.isConst
                            spec.sourceLayer = 1;
                            spec.term = '1';
                            spec.coef = ref.value .^ p;
                        else
                            spec.sourceLayer = ref.stateLayer;
                            spec.term = power_term_local(ref.index, p);
                            spec.coef = 1;
                            compiler.unifiedCompactPrior.maxPowerObserved = max(compiler.unifiedCompactPrior.maxPowerObserved, abs(p));
                        end
                    end
                case {'+','-'}
                    [ref, compiler] = compile_as_ref_local(node, compiler);
                    if ref.isConst
                        spec.sourceLayer = 1;
                        spec.term = '1';
                        spec.coef = ref.value;
                    else
                        spec.sourceLayer = ref.stateLayer;
                        spec.term = sprintf('v%d', ref.index);
                        spec.coef = 1;
                    end
                otherwise
                    error('Unsupported binary operator: %s', node.op);
            end
        otherwise
            error('Unsupported AST node type in basis-term compilation: %s', node.type);
    end
end

function [spec, compiler] = compile_product_like_local(factorNodes, factorSpecsIn, compiler)
    if nargin < 2 || isempty(factorSpecsIn)
        factorSpecs = struct('sourceLayer', {}, 'term', {}, 'coef', {});
        for i = 1:numel(factorNodes)
            [s, compiler] = compile_basis_term_local(factorNodes{i}, compiler);
            factorSpecs(end+1) = s; %#ok<AGROW>
        end
    else
        factorSpecs = factorSpecsIn;
    end

    coef = 1;
    nonConst = struct('sourceLayer', {}, 'term', {}, 'coef', {});
    for i = 1:numel(factorSpecs)
        coef = coef .* factorSpecs(i).coef;
        if ~strcmp(factorSpecs(i).term, '1')
            factorSpecs(i).coef = 1;
            nonConst(end+1) = factorSpecs(i); %#ok<AGROW>
        end
    end
    if isempty(nonConst)
        spec = struct('sourceLayer', 1, 'term', '1', 'coef', coef);
        return;
    end

    srcLayers = [nonConst.sourceLayer];
    if numel(unique(srcLayers)) > 1
        refs = cell(1, numel(nonConst));
        for i = 1:numel(nonConst)
            % Each factor is materialized as a state node so the product can be
            % formed from one source vector in the next branch.
            row = make_row_from_term_local(nonConst(i).sourceLayer, nonConst(i).term, 1);
            [refs{i}, compiler] = add_state_node_local(row, sprintf('align product factor %s', nonConst(i).term), compiler);
        end
        targetLayer = max(cellfun(@(r) r.stateLayer, refs));
        refsAligned = cell(1, numel(refs));
        for i = 1:numel(refs)
            refsAligned{i} = refs{i};
        end
        if any(cellfun(@(r) r.stateLayer ~= targetLayer, refsAligned))
            % Align by copying all factors to a common new state.
            aligned = cell(1, numel(refsAligned));
            for i = 1:numel(refsAligned)
                r = refsAligned{i};
                row = make_row_from_term_local(r.stateLayer, sprintf('v%d', r.index), 1);
                [aligned{i}, compiler] = add_state_node_at_layer_local(row, targetLayer + 1, sprintf('copy factor v%d', r.index), compiler);
            end
            refsAligned = aligned;
        end
        terms = cell(1, numel(refsAligned));
        for i = 1:numel(refsAligned)
            terms{i} = sprintf('v%d', refsAligned{i}.index);
        end
        spec.sourceLayer = refsAligned{1}.stateLayer;
        spec.term = join_product_terms_local(terms);
        spec.coef = coef;
    else
        spec.sourceLayer = nonConst(1).sourceLayer;
        terms = {nonConst.term};
        spec.term = join_product_terms_local(terms);
        spec.coef = coef;
    end
    compiler.unifiedCompactPrior.includeProducts = true;
end

function [ref, compiler] = compile_as_ref_local(node, compiler)
    if strcmp(node.type, 'num')
        ref = struct('isConst', true, 'value', node.value, 'stateLayer', NaN, 'index', NaN);
        return;
    end
    if strcmp(node.type, 'var')
        ref = struct('isConst', false, 'value', NaN, 'stateLayer', 1, 'index', node.value);
        return;
    end
    key = canonical_node_key_local(node);
    if isKey(compiler.nodeKeyMap, key)
        ref = compiler.nodeKeyMap(key);
        if isKey(compiler.nodeUseCount, key)
            compiler.nodeUseCount(key) = compiler.nodeUseCount(key) + 1;
        else
            compiler.nodeUseCount(key) = 2;
        end
        return;
    end
    [row, compiler] = compile_linear_row_local(node, compiler);
    [ref, compiler] = add_state_node_local(row, node.expr, compiler);
    compiler.nodeKeyMap(key) = ref;
    compiler.nodeUseCount(key) = 1;
    compiler.nodeKeyOrder{end+1} = key;
end

function [ref, compiler] = add_state_node_local(row, expr, compiler)
    targetState = max(2, max_source_layer_local(row) + 1);
    [ref, compiler] = add_state_node_at_layer_local(row, targetState, expr, compiler);
end

function [ref, compiler] = add_state_node_at_layer_local(row, targetState, expr, compiler)
    compiler = ensure_state_layer_local(compiler, targetState);
    rowIndex = numel(compiler.stateNodes{targetState}) + 1;
    nodeRec = struct();
    nodeRec.expr = expr;
    nodeRec.row = row;
    nodeRec.rowIndex = rowIndex;
    nodeRec.stateLayer = targetState;
    compiler.stateNodes{targetState}{rowIndex} = nodeRec;
    ref = struct('isConst', false, 'value', NaN, 'stateLayer', targetState, 'index', rowIndex);
    compiler.decomposition(end+1).stateLayer = targetState; %#ok<AGROW>
    compiler.decomposition(end).rowIndex = rowIndex;
    compiler.decomposition(end).expr = expr;
    compiler.decomposition(end).rowSummary = summarize_row_local(row);
end

function compiler = ensure_state_layer_local(compiler, stateLayer)
    while numel(compiler.stateNodes) < stateLayer
        compiler.stateNodes{end+1} = {}; %#ok<AGROW>
    end
end

% =========================================================================
% Row contribution utilities
% =========================================================================
function row = make_row_from_term_local(sourceLayer, term, coef)
    row = struct('sourceLayer', {}, 'term', {}, 'coef', {});
    row(1).sourceLayer = sourceLayer;
    row(1).term = char(term);
    row(1).coef = coef;
    row = simplify_row_local(row);
end

function row = combine_rows_local(a, b, signB)
    if isempty(a); row = scale_row_local(b, signB); return; end
    if isempty(b); row = a; return; end
    b = scale_row_local(b, signB);
    row = [a(:); b(:)].';
    row = simplify_row_local(row);
end

function row = scale_row_local(row, scale)
    for i = 1:numel(row)
        row(i).coef = scale .* row(i).coef;
    end
    row = simplify_row_local(row);
end

function row = simplify_row_local(row)
    if isempty(row); return; end
    out = struct('sourceLayer', {}, 'term', {}, 'coef', {});
    for i = 1:numel(row)
        if isempty(row(i).term) || abs(row(i).coef) <= 0
            continue;
        end
        keyTerm = normalize_term_text_local(row(i).term);
        keySrc = row(i).sourceLayer;
        j = [];
        for q = 1:numel(out)
            if out(q).sourceLayer == keySrc && strcmp(out(q).term, keyTerm)
                j = q; break;
            end
        end
        if isempty(j)
            out(end+1).sourceLayer = keySrc; %#ok<AGROW>
            out(end).term = keyTerm;
            out(end).coef = row(i).coef;
        else
            out(j).coef = out(j).coef + row(i).coef;
        end
    end
    keep = true(1, numel(out));
    for i = 1:numel(out)
        keep(i) = abs(out(i).coef) > 0;
    end
    row = out(keep);
    if isempty(row)
        row = struct('sourceLayer', 1, 'term', '1', 'coef', 0);
    end
end

function [terms, coefs] = row_terms_for_source_local(row, sourceLayer)
    terms = {};
    coefs = [];
    if isempty(row); return; end
    for i = 1:numel(row)
        if row(i).sourceLayer == sourceLayer && abs(row(i).coef) > 0
            terms{end+1,1} = row(i).term; %#ok<AGROW>
            coefs(end+1) = row(i).coef; %#ok<AGROW>
        end
    end
    if ~isempty(terms)
        [terms, coefs] = combine_duplicate_terms_local(terms, coefs);
    end
end

function m = max_source_layer_local(row)
    if isempty(row)
        m = 1;
    else
        vals = [row.sourceLayer];
        vals = vals(isfinite(vals));
        if isempty(vals); m = 1; else; m = max(vals); end
    end
end

function txt = summarize_row_local(row)
    parts = cell(1, numel(row));
    for i = 1:numel(row)
        parts{i} = sprintf('%+.6g@h%d:%s', row(i).coef, row(i).sourceLayer, row(i).term);
    end
    txt = strjoin(parts, ' ');
end

% =========================================================================
% Unified compact dictionary construction
% =========================================================================
function terms = build_augmentation_terms_local(inputDim, prior, cfg)
    if ~logical(get_struct_field_local(cfg, 'enableAugmentation', true))
        terms = {};
        return;
    end
    mode = normalize_augmentation_mode_local(cfg);
    if strcmp(mode,'fixed_neural_ridge')
        % The data-aware neural terms are attached after the exact structural
        % seed has been constructed and evaluated. Keep only the bias channel
        % during this preliminary compilation.
        terms = {'1'};
    else
        terms = build_uniform_polynomial_terms_for_dim_local(inputDim, prior);
    end
end

function mode = normalize_augmentation_mode_local(cfg)
    mode = lower(strtrim(char(get_struct_field_local(cfg,'augmentationMode','polynomial'))));
    switch mode
        case {'poly','polynomial','uniform_poly','uniform_polynomial'}
            mode = 'polynomial';
        case {'neural','neural_ridge','fixed_neural_ridge','fixed-neural-ridge'}
            mode = 'fixed_neural_ridge';
        otherwise
            error('Unsupported Stage-1 augmentation mode: %s',mode);
    end
end

function text = augmentation_source_local(mode,cfg)
    if strcmp(mode,'fixed_neural_ridge')
        if logical(get_struct_field_local(cfg,'neuralIncludeLinearTerms',false))
            text = sprintf(['Per-output selected SINDy/PySR core DAG with uniform ', ...
                'constant+linear coordinates+%d fixed tanh ridge augmentation'], ...
                get_struct_field_local(cfg,'neuralCount',NaN));
        else
            text = sprintf(['Per-output selected SINDy/PySR core DAG with uniform ', ...
                'constant+%d fixed tanh ridge augmentation'], ...
                get_struct_field_local(cfg,'neuralCount',NaN));
        end
    else
        text = sprintf(['Per-output selected SINDy/PySR core DAG with uniform ', ...
            'constant+Poly_%d augmentation'],get_struct_field_local(cfg,'polyOrder',2));
    end
end

function terms = build_uniform_polynomial_terms_for_dim_local(inputDim, prior)
    terms = {'1'};
    polyOrder = max(1, round(get_struct_field_local(prior, 'polyOrder', 2)));
    terms = [terms(:); generate_polynomial_terms_local(inputDim, polyOrder, true).'];
    terms = unique_stable_local(terms);
end

function terms = generate_polynomial_terms_local(inputDim, order, includeProducts)
    terms = {};
    if nargin < 3; includeProducts = true; end
    if inputDim < 1 || order < 1; return; end
    if includeProducts
        for deg = 1:order
            exps = compositions_local(deg, inputDim);
            for r = 1:size(exps, 1)
                terms{end+1,1} = monomial_term_local(exps(r, :)); %#ok<AGROW>
            end
        end
    else
        for k = 1:inputDim
            for deg = 1:order
                exps = zeros(1,inputDim);
                exps(k) = deg;
                terms{end+1,1} = monomial_term_local(exps); %#ok<AGROW>
            end
        end
    end
    terms = unique_stable_local(terms);
end

function C = compositions_local(total, dim)
    if dim == 1
        C = total;
        return;
    end
    C = [];
    for a = total:-1:0
        rest = compositions_local(total - a, dim - 1);
        C = [C; [repmat(a, size(rest,1), 1), rest]]; %#ok<AGROW>
    end
end

function t = monomial_term_local(exps)
    parts = {};
    for k = 1:numel(exps)
        p = exps(k);
        if p == 0; continue; end
        if p == 1
            parts{end+1} = sprintf('v%d', k); %#ok<AGROW>
        else
            parts{end+1} = sprintf('v%d^%d', k, p); %#ok<AGROW>
        end
    end
    if isempty(parts)
        t = '1';
    else
        t = strjoin(parts, '*');
    end
end

function exactTerms = unique_terms_from_row_spec_local(rowSpec)
    exactTerms = {};
    for r = 1:numel(rowSpec)
        tr = rowSpec{r};
        if isempty(tr); continue; end
        if ischar(tr) || isstring(tr); tr = cellstr(tr); end
        exactTerms = [exactTerms; tr(:)]; %#ok<AGROW>
    end
    exactTerms = unique_stable_local(exactTerms);
end

function compiler = register_unary_op_local(compiler, op) %#ok<INUSD>
    % SR unary operators are represented by structural DAG channels only.
    % They must not be injected into the uniform augmentation dictionary.
end

% =========================================================================
% Architecture/dictionary seed construction
% =========================================================================
function D = make_empty_dictionary_local(task, row, term, idx, cfg) %#ok<INUSD>
    D = struct();
    D.caseId = task.name;
    D.priorLevel = get_struct_field_local(get_struct_field_local(task, 'prior', struct()), 'level', NaN);
    D.priorLevelName = get_struct_field_local(get_struct_field_local(task, 'prior', struct()), 'levelName', task.casemode);
    D.noFallback = true;
    D.appendGlobalTerms = false;
end

function seedCoef = make_seed_coef_from_row_seed_local(arch)
    Coef_template = create_coef_template(arch);
    seedCoef = zero_Coef_like(Coef_template);
    D = arch.caseDictionary;
    if isfield(D, 'stage0SeedRowTerms') && isfield(D, 'stage0SeedRowCoef')
        seedRowTerms = D.stage0SeedRowTerms;
        seedRowCoef = D.stage0SeedRowCoef;
    elseif isfield(D, 'rowTerms') && isfield(D, 'rowSeedCoef')
        seedRowTerms = D.rowTerms;
        seedRowCoef = D.rowSeedCoef;
    else
        return;
    end
    for ell = 1:arch.layer
        for src = 1:ell
            if src > size(Coef_template,1) || ell > size(Coef_template,2) || isempty(Coef_template{src,ell})
                continue;
            end
            if size(seedRowTerms,1) < src || size(seedRowTerms,2) < ell || isempty(seedRowTerms{src,ell})
                continue;
            end
            if size(seedRowCoef,1) < src || size(seedRowCoef,2) < ell || isempty(seedRowCoef{src,ell})
                continue;
            end
            dims = get_arch_dims(arch);
            inputState = ell - src + 1;
            inputDim = dims(inputState);
            termList = explicit_case_dictionary_terms(inputDim, arch, ell, src);
            A = seedCoef{src,ell};
            rowTerms = seedRowTerms{src,ell};
            rowCoefs = seedRowCoef{src,ell};
            for r = 1:min(size(A,1), numel(rowTerms))
                tr = rowTerms{r};
                cr = rowCoefs{r};
                if isempty(tr); continue; end
                if ischar(tr) || isstring(tr); tr = cellstr(tr); end
                cr = reshape(cr, 1, []);
                for q = 1:min(numel(tr), numel(cr))
                    col = find(strcmp(termList, char(tr{q})), 1);
                    if isempty(col)
                        error('Stage0-to-PhDN seed term "%s" is missing from dictionary block A{%d,%d}.', char(tr{q}), src, ell);
                    end
                    A(r, col) = cr(q);
                end
            end
            seedCoef{src,ell} = A;
        end
    end
end

function validate_seed_coef_size_local(seedCoef, arch)
    Coef_template = create_coef_template(arch);
    for ell = 1:size(Coef_template, 2)
        for src = 1:ell
            if isempty(Coef_template{src, ell}); continue; end
            if src > size(seedCoef,1) || ell > size(seedCoef,2) || isempty(seedCoef{src,ell})
                error('Stage0-to-PhDN seed is missing coefficient cell (%d,%d).', src, ell);
            end
            if ~isequal(size(seedCoef{src, ell}), size(Coef_template{src, ell}))
                error('Stage0-to-PhDN seed size mismatch at cell (%d,%d): seed [%s], template [%s].', ...
                    src, ell, num2str(size(seedCoef{src, ell})), num2str(size(Coef_template{src, ell})));
            end
        end
    end
end

function n = count_seed_nonzeros_local(seedCoef)
    n = 0;
    for i = 1:numel(seedCoef)
        if ~isempty(seedCoef{i})
            n = n + nnz(abs(seedCoef{i}) > 0);
        end
    end
end

function n = count_intermediate_nodes_local(compiler)
    n = 0;
    for s = 2:numel(compiler.stateNodes)
        n = n + numel(compiler.stateNodes{s});
    end
end

function d = state_dim_local(compiler, stateLayer)
    if stateLayer == 1
        d = compiler.nx;
    else
        if numel(compiler.stateNodes) < stateLayer
            d = 0;
        else
            d = numel(compiler.stateNodes{stateLayer});
        end
    end
end

% =========================================================================
% Term helpers
% =========================================================================
function term = power_term_local(idx, p)
    if abs(p - 1) < 1e-12
        term = sprintf('v%d', idx);
    elseif abs(p - 2) < 1e-12
        term = sprintf('v%d^2', idx);
    elseif abs(p - 3) < 1e-12
        term = sprintf('v%d^3', idx);
    else
        term = sprintf('v%d^%g', idx, p);
    end
end

function term = join_product_terms_local(terms)
    terms = terms(:).';
    if isempty(terms)
        term = '1';
    elseif numel(terms) == 1
        term = char(terms{1});
    else
        term = strjoin(terms, '*');
    end
end

function s = normalize_term_text_local(s)
    s = char(s);
    s = strrep(strtrim(s), ' ', '');
    s = strip_outer_parentheses_local(s);
end

function [termsOut, coefsOut] = combine_duplicate_terms_local(terms, coefs)
    termsOut = {};
    coefsOut = [];
    for i = 1:numel(terms)
        t = normalize_term_text_local(terms{i});
        j = find(strcmp(termsOut, t), 1);
        if isempty(j)
            termsOut{end+1,1} = t; %#ok<AGROW>
            coefsOut(end+1) = coefs(i); %#ok<AGROW>
        else
            coefsOut(j) = coefsOut(j) + coefs(i);
        end
    end
    keep = abs(coefsOut) > 0;
    termsOut = termsOut(keep);
    coefsOut = coefsOut(keep);
end

function key = canonical_node_key_local(node)
    if isfield(node, 'expr') && ~isempty(node.expr)
        key = normalize_term_text_local(node.expr);
    else
        key = node.type;
    end
end

function expressions = shared_subexpressions_local(compiler)
%SHARED_SUBEXPRESSIONS_LOCAL Return compiler-cache nodes referenced more than once.
    expressions = {};
    if ~isfield(compiler,'nodeKeyOrder') || ~isfield(compiler,'nodeUseCount')
        return;
    end
    for k = 1:numel(compiler.nodeKeyOrder)
        key = compiler.nodeKeyOrder{k};
        if isKey(compiler.nodeUseCount,key) && compiler.nodeUseCount(key) > 1
            expressions{end+1} = key; %#ok<AGROW>
        end
    end
end

% =========================================================================
% Parser
% =========================================================================
function node = parse_stage0_expr_local(s)
    s = strip_outer_parentheses_local(normalize_stage0_term_local(s));
    num = str2double(s);
    if isfinite(num)
        node = make_num_node_local(num, s); return;
    end

    % Top-level addition/subtraction must be resolved before a leading unary
    % sign.  Otherwise an expression such as -x0+x1 is incorrectly parsed as
    % -(x0+x1) instead of (-x0)+x1.  The splitter retains the leading sign in
    % the first part, which is then handled recursively as a true unary sign.
    [parts, signs] = split_top_level_add_sub_local(s);
    if numel(parts) > 1
        node = parse_stage0_expr_local(parts{1});
        if signs(1) < 0
            node = make_unary_node_local('neg', node, ['-' parts{1}]);
        end
        for i = 2:numel(parts)
            rhs = parse_stage0_expr_local(parts{i});
            if signs(i) >= 0
                node = make_binary_node_local('+', node, rhs, s);
            else
                node = make_binary_node_local('-', node, rhs, s);
            end
        end
        return;
    end

    if startsWith(s, '+') && numel(s) > 1
        node = parse_stage0_expr_local(s(2:end)); return;
    elseif startsWith(s, '-') && numel(s) > 1
        child = parse_stage0_expr_local(s(2:end));
        node = make_unary_node_local('neg', child, s); return;
    end

    [parts, ops] = split_top_level_mul_div_local(s);
    if numel(parts) > 1
        node = parse_stage0_expr_local(parts{1});
        for i = 2:numel(parts)
            rhs = parse_stage0_expr_local(parts{i});
            node = make_binary_node_local(ops(i-1), node, rhs, s);
        end
        return;
    end

    [base, exponent, hasPow] = split_power_local(s);
    if hasPow
        node = make_binary_node_local('^', parse_stage0_expr_local(base), make_num_node_local(exponent, num2str(exponent, 16)), s);
        return;
    end

    [fname, arg, isFun] = parse_function_local(s);
    if isFun
        node = make_unary_node_local(canonical_unary_name_local(fname), parse_stage0_expr_local(arg), s);
        return;
    end

    % Canonical PhDN variables are one-based (v1,v2,...). Official PySR
    % variables are zero-based (x0,x1,...), so an unnormalized xK token maps
    % to v(K+1). normalize_stage0_term_local normally performs this conversion
    % first; the separate parser branches are a defensive fallback.
    m = regexp(s, '^v(\d+)$', 'tokens', 'once');
    if ~isempty(m)
        idx = str2double(m{1});
        if idx < 1; error('Canonical PhDN variable index must be one-based: %s', s); end
        node = make_var_node_local(idx, s); return;
    end
    m = regexp(s, '^x(\d+)$', 'tokens', 'once');
    if ~isempty(m)
        node = make_var_node_local(str2double(m{1}) + 1, s); return;
    end

    error('Cannot parse Stage-0 expression for compact PhDN-DAG translation: %s', s);
end

function s = normalize_stage0_term_local(s, task, variableIndexBase)
    if nargin < 2 || isempty(task); task = struct(); end
    if nargin < 3 || isempty(variableIndexBase); variableIndexBase = 0; end
    s = char(s);
    s = strtrim(s);
    s = strrep(s, '**', '^');
    s = strrep(s, ' ', '');
    s = strrep(s, 'Abs(', 'abs(');

    % Current Stage-0 exports use one-based x1,...,xn, while legacy PySR results
    % may use zero-based x0,...,x(n-1). Convert according to the indexing mode
    % detected once from the complete core set. Placeholders prevent chained
    % replacements such as x1->v1 followed by another alias rewrite.
    if isstruct(task) && isfield(task, 'nx') && ~isempty(task.nx) && isfinite(task.nx)
        nxLocal = max(0, round(task.nx));
        if variableIndexBase == 0
            sourceIndices = 0:(nxLocal-1);
            canonicalIndices = 1:nxLocal;
        else
            sourceIndices = 1:nxLocal;
            canonicalIndices = 1:nxLocal;
        end
        for q = numel(sourceIndices):-1:1
            pattern = sprintf('(?<![A-Za-z0-9_])x%d(?![A-Za-z0-9_])', sourceIndices(q));
            s = regexprep(s, pattern, sprintf('__PYSRVAR%d__', canonicalIndices(q)));
        end
        for kk = 1:nxLocal
            s = strrep(s, sprintf('__PYSRVAR%d__', kk), sprintf('v%d', kk));
        end
    end

    % Convert task-specific PySR variable identifiers to the canonical PhDN
    % input notation before interpreting constants/functions.  Replacement is
    % token-aware, so a one-letter variable such as "a" is not substituted
    % inside names such as square, sqrt_abs, or exp.
    aliases = stage0_variable_alias_map_local(task);
    for i = 1:numel(aliases)
        alias = aliases(i).name;
        if isempty(alias); continue; end
        escaped = regexptranslate('escape', alias);
        pattern = ['(?<![A-Za-z0-9_])' escaped '(?![A-Za-z0-9_])'];
        s = regexprep(s, pattern, aliases(i).canonical);
    end

    s = strrep(s, 'π', 'pi');
    s = regexprep(s, '(?<![A-Za-z0-9_])pi(?![A-Za-z0-9_])', num2str(pi, 16));
end

function indexBase = detect_stage0_variable_index_base_local(coreTerms, task, modelRow)
%DETECT_STAGE0_VARIABLE_INDEX_BASE_LOCAL Distinguish x0-based and x1-based SR text.
    indexBase = [];
    if isstruct(modelRow) && isfield(modelRow,'variableIndexBase') && ...
            isscalar(modelRow.variableIndexBase) && any(modelRow.variableIndexBase == [0,1])
        indexBase = double(modelRow.variableIndexBase);
    end
    joined = strjoin(cellfun(@char,coreTerms,'UniformOutput',false),' ');
    if ~isempty(regexp(joined,'(?<![A-Za-z0-9_])x0(?![A-Za-z0-9_])','once'))
        indexBase = 0;
    end
    if isempty(indexBase)
        nxLocal = get_struct_field_local(task,'nx',0);
        namesAreCanonicalOneBased = false;
        if isstruct(task) && isfield(task,'variableNames') && numel(task.variableNames) >= nxLocal && nxLocal > 0
            names = task.variableNames;
            if ischar(names) || isstring(names); names = cellstr(names); end
            expected = arrayfun(@(k) sprintf('x%d',k),1:nxLocal,'UniformOutput',false);
            namesAreCanonicalOneBased = isequal(cellfun(@char,names(1:nxLocal),'UniformOutput',false),expected);
        end
        hasHighestOneBasedToken = nxLocal > 0 && ~isempty(regexp(joined, ...
            sprintf('(?<![A-Za-z0-9_])x%d(?![A-Za-z0-9_])',nxLocal),'once'));
        if namesAreCanonicalOneBased || hasHighestOneBasedToken
            indexBase = 1;
        else
            % Preserve compatibility for ambiguous legacy expressions which use
            % x1,... but contain neither x0 nor x_n in the selected core set.
            indexBase = 0;
        end
    end
end

function aliases = stage0_variable_alias_map_local(task)
    aliases = struct('name', {}, 'canonical', {}, 'index', {});
    if ~isstruct(task) || ~isfield(task, 'variableNames') || isempty(task.variableNames)
        return;
    end
    names = task.variableNames;
    if ischar(names) || isstring(names); names = cellstr(names); end
    names = names(:).';
    if isfield(task, 'nx') && ~isempty(task.nx) && isfinite(task.nx)
        names = names(1:min(numel(names), max(0, round(task.nx))));
    end

    % Longer aliases are processed first. This is mainly defensive for names
    % such as x and x_long; token boundaries already protect x from matching
    % the prefix of x_long.
    lengths = cellfun(@(x) numel(char(x)), names);
    [~, order] = sort(lengths, 'descend');
    for q = 1:numel(order)
        idx = order(q);
        name = strtrim(char(names{idx}));
        if isempty(name); continue; end
        aliases(end+1).name = name; %#ok<AGROW>
        aliases(end).canonical = sprintf('v%d', idx);
        aliases(end).index = idx;
    end
end

function [parts, signs] = split_top_level_add_sub_local(s)
    parts = {};
    signs = [];
    level = 0;
    start = 1;
    currentSign = 1;
    for i = 1:numel(s)
        ch = s(i);
        if ch == '('; level = level + 1; elseif ch == ')'; level = level - 1; end
        if level == 0 && (ch == '+' || ch == '-') && is_binary_add_sub_sign_local(s, i)
            piece = s(start:i-1);
            if ~isempty(piece)
                parts{end+1} = piece; %#ok<AGROW>
                signs(end+1) = currentSign; %#ok<AGROW>
            end
            currentSign = 1;
            if ch == '-'; currentSign = -1; end
            start = i + 1;
        end
    end
    piece = s(start:end);
    if ~isempty(piece)
        parts{end+1} = piece; %#ok<AGROW>
        signs(end+1) = currentSign; %#ok<AGROW>
    end
end

function tf = is_binary_add_sub_sign_local(s, pos)
    if pos <= 1; tf = false; return; end
    prev = s(pos - 1);
    if any(prev == ['(', '*', '/', '^', '+', '-', 'e', 'E'])
        tf = false;
    else
        tf = true;
    end
end

function [parts, ops] = split_top_level_mul_div_local(s)
    parts = {};
    ops = '';
    level = 0;
    start = 1;
    for i = 1:numel(s)
        ch = s(i);
        if ch == '('; level = level + 1; elseif ch == ')'; level = level - 1; end
        if level == 0 && (ch == '*' || ch == '/')
            if ch == '*' && ((i < numel(s) && s(i+1) == '*') || (i > 1 && s(i-1) == '*'))
                continue;
            end
            piece = s(start:i-1);
            if ~isempty(piece)
                parts{end+1} = piece; %#ok<AGROW>
                ops(end+1) = ch; %#ok<AGROW>
            end
            start = i + 1;
        end
    end
    piece = s(start:end);
    if ~isempty(piece); parts{end+1} = piece; end %#ok<AGROW>
    if numel(parts) <= 1; ops = ''; end
end

function [base, exponent, hasPow] = split_power_local(s)
    level = 0; pos = 0;
    for i = numel(s):-1:1
        ch = s(i);
        if ch == ')'; level = level + 1; elseif ch == '('; level = level - 1;
        elseif level == 0 && ch == '^'; pos = i; break;
        end
    end
    if pos == 0
        base = ''; exponent = NaN; hasPow = false; return;
    end
    base = s(1:pos-1);
    % Keep the compact-DAG parser consistent with
    % evaluate_explicit_case_terms.  PySR's SymPy export can serialize a
    % protected sqrt/cube composition as Abs(z)**(3/2); after normalization
    % that reaches this parser as abs(z)^(3/2).  str2double('(3/2)') is NaN,
    % even though the numerical branch evaluator already supports signed
    % rational powers.  Parse the complete numeric exponent syntax here so a
    % valid PySR expression is not incorrectly diverted to compile fallback.
    exponentText = strip_outer_parentheses_local(strtrim(s(pos+1:end)));
    exponent = parse_numeric_exponent_local(exponentText);
    hasPow = isfinite(exponent);
end

function exponent = parse_numeric_exponent_local(text)
%PARSE_NUMERIC_EXPONENT_LOCAL Parse decimal, integer, or signed rational powers.
% Supports PySR/SymPy forms such as ^(-1/4), ^(3/2), ^-2, and ^0.5.
    text = strip_outer_parentheses_local(strtrim(char(text)));
    exponent = str2double(text);
    if isfinite(exponent)
        return;
    end
    tok = regexp(text, ...
        '^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$', ...
        'tokens', 'once');
    if isempty(tok)
        exponent = NaN;
        return;
    end
    numerator = str2double(tok{1});
    denominator = str2double(tok{2});
    if ~isfinite(numerator) || ~isfinite(denominator) || denominator == 0
        exponent = NaN;
    else
        exponent = numerator / denominator;
    end
end

function [fname, arg, isFun] = parse_function_local(s)
    m = regexp(s, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
    if isempty(m)
        fname = ''; arg = ''; isFun = false;
    else
        fname = m{1}; arg = m{2}; isFun = true;
    end
end

function s = strip_outer_parentheses_local(s)
    changed = true;
    while changed && numel(s) >= 2 && s(1) == '(' && s(end) == ')'
        changed = false;
        level = 0; ok = true;
        for i = 1:numel(s)
            if s(i) == '('
                level = level + 1;
            elseif s(i) == ')'
                level = level - 1;
                if level == 0 && i < numel(s)
                    ok = false; break;
                end
            end
            if level < 0; ok = false; break; end
        end
        if ok && level == 0
            s = s(2:end-1);
            changed = true;
        end
    end
end

% =========================================================================
% AST and constants
% =========================================================================
function node = make_var_node_local(idx, expr)
    node = make_node_base_local('var', '', expr);
    node.value = idx;
end

function node = make_num_node_local(val, expr)
    node = make_node_base_local('num', '', expr);
    node.value = val;
end

function node = make_unary_node_local(op, child, expr)
    node = make_node_base_local('unary', op, expr);
    node.left = child;
end

function node = make_binary_node_local(op, left, right, expr)
    node = make_node_base_local('binary', op, expr);
    node.left = left;
    node.right = right;
end

function node = make_node_base_local(type, op, expr)
    node = struct();
    node.type = type;
    node.op = op;
    node.left = [];
    node.right = [];
    node.value = [];
    node.expr = expr;
end

function d = ast_depth_local(node)
    if any(strcmp(node.type, {'var','num'}))
        d = 1;
    elseif strcmp(node.type, 'unary')
        d = 1 + ast_depth_local(node.left);
    elseif strcmp(node.type, 'binary')
        d = 1 + max(ast_depth_local(node.left), ast_depth_local(node.right));
    else
        d = 1;
    end
end

function y = apply_unary_const_local(op, x)
% Constant folding belongs to the Stage-0 PySR structural path and therefore
% uses the same unprotected real operator definitions as the exported PySR
% expression. Invalid constants remain nonfinite/complex and are rejected by
% the normal Stage-0 reproduction checks instead of being silently clipped.
    switch lower(op)
        case 'inv'; y = 1 ./ x;
        case {'square','sqr'}; y = x.^2;
        case 'cube'; y = x.^3;
        case 'sqrt'; y = sqrt(x);
        case 'sqrt_abs'; y = sqrt(abs(x));
        case 'exp'; y = exp(x);
        case 'sin'; y = sin(x);
        case 'cos'; y = cos(x);
        case 'asin'; y = asin(x);
        case 'log'; y = log(x);
        case {'abs','Abs'}; y = abs(x);
        otherwise; y = x;
    end
end

function name = canonical_unary_name_local(name)
    name = char(name);
    switch lower(name)
        case 'sqr'
            name = 'square';
        case 'abs'
            name = 'abs';
        otherwise
            % keep original spelling
    end
end

function val = get_struct_field_local(s, name, defaultVal)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = defaultVal;
    end
end

function out = unique_stable_local(in)
    out = {};
    for i = 1:numel(in)
        if isempty(in{i}); continue; end
        name = char(in{i});
        if ~any(strcmp(out, name))
            out{end+1,1} = name; %#ok<AGROW>
        end
    end
    out = out(:).';
end
