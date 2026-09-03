function [arch, CoefExact, compileInfo] = build_manual_constructive_phdn(task, symbolicExpressions, spec)
%BUILD_MANUAL_CONSTRUCTIVE_PHDN Build one exact case-specific PhDN DAG.
%
% This helper is intentionally deterministic and does not perform symbolic
% search.  A case compiler supplies an explicit architecture, branch
% dictionaries, row supports, and exact coefficients through SPEC.

    if nargin < 1 || isempty(task)
        error('A registered task is required.');
    end
    if nargin < 2 || isempty(symbolicExpressions)
        symbolicExpressions = task.symbolicExpressions;
    end
    if nargin < 3 || ~isstruct(spec)
        error('A constructive case specification struct is required.');
    end

    validate_symbolic_input_local(task, symbolicExpressions);

    required = {'layer','hiddenDims','branchActiveMask','termsByBlock', ...
        'rowTerms','rowSeedCoef','nodeEquations'};
    for k = 1:numel(required)
        if ~isfield(spec, required{k})
            error('Constructive specification is missing field "%s".', required{k});
        end
    end

    arch = task.arch;
    arch.layer = spec.layer;
    arch.hiddenDims = reshape(spec.hiddenDims, 1, []);
    arch.nx = task.nx;
    arch.ny = task.ny;
    arch.operatorMode = 'true';
    arch.dictionaryMode = get_field_local(spec, 'dictionaryMode', ...
        'manual_constructive_exact_dag');
    if isfield(arch, 'dims')
        arch = rmfield(arch, 'dims');
    end

    branchActive = logical(spec.branchActiveMask);
    if ~isequal(size(branchActive), [arch.layer, arch.layer])
        error('branchActiveMask must have size %dx%d.', arch.layer, arch.layer);
    end
    arch.branchActiveMask = branchActive;
    arch.branchActiveMode = get_field_local(spec, 'branchActiveMode', ...
        'manual_constructive_case');

    D = struct();
    D.caseId = task.name;
    D.priorLevel = 4;
    D.priorLevelName = 'strong_prior';
    D.noFallback = true;
    D.appendGlobalTerms = false;
    D.source = get_field_local(spec, 'source', ...
        'Manual exact compilation of a symbolic-representability case');
    D.symbolicExpressions = symbolicExpressions(:).';
    D.termsByBlock = spec.termsByBlock;
    D.rowTerms = spec.rowTerms;
    D.rowSeedCoef = spec.rowSeedCoef;
    arch.caseDictionary = D;

    CoefExact = create_coef_template(arch);
    CoefExact = zero_Coef_like(CoefExact);
    for ell = 1:arch.layer
        for src = 1:ell
            if isempty(CoefExact{src,ell}) || ...
                    size(D.rowTerms,1) < src || size(D.rowTerms,2) < ell || ...
                    empty_row_spec_local(D.rowTerms{src,ell})
                continue;
            end
            CoefExact{src,ell} = fill_exact_block_local(CoefExact{src,ell}, ...
                D.termsByBlock{src,ell}, D.rowTerms{src,ell}, ...
                D.rowSeedCoef{src,ell}, src, ell);
        end
    end

    arch.exactConstructiveCoef = CoefExact;
    arch.exactSymbolicExpressions = symbolicExpressions(:).';

    compileInfo = struct();
    compileInfo.caseId = task.name;
    compileInfo.caseName = task.caseName;
    compileInfo.caseLabel = get_field_local(spec, 'caseLabel', task.description);
    compileInfo.compileMode = get_field_local(spec, 'compileMode', ...
        'manual_case_specific_constructive_compilation');
    compileInfo.layerDimensionVector = get_arch_dims(arch);
    compileInfo.nLayers = arch.layer;
    compileInfo.hiddenDims = arch.hiddenDims;
    compileInfo.nActiveBranches = nnz(branchActive);
    [compileInfo.activeBranchCoordinates, compileInfo.nonChainBranchCoordinates] = ...
        active_branch_coordinates_local(branchActive);
    compileInfo.nNonChainBranches = numel(compileInfo.nonChainBranchCoordinates);
    compileInfo.nExactActiveCoefficients = count_exact_nonzero_local(CoefExact);
    compileInfo.symbolicExpressions = symbolicExpressions(:).';
    compileInfo.sharedSubexpressions = get_field_local(spec, ...
        'sharedSubexpressions', {});
    compileInfo.nodeEquations = spec.nodeEquations(:).';
    compileInfo.summary = get_field_local(spec, 'summary', ...
        'The supplied expression is exactly represented by the registered finite PhDN DAG.');
end

function validate_symbolic_input_local(task, expressions)
    if ischar(expressions) || isstring(expressions)
        expressions = cellstr(expressions);
    end
    if ~iscell(expressions) || numel(expressions) ~= task.ny
        error('Case %s requires exactly %d symbolic expression(s).', ...
            task.name, task.ny);
    end
    expected = task.symbolicExpressions;
    for k = 1:task.ny
        actualK = canonical_expression_local(expressions{k});
        expectedK = canonical_expression_local(expected{k});
        if ~strcmp(actualK, expectedK)
            error(['Expression %d does not match registered case %s. ', ...
                'This compiler intentionally maps only the declared manuscript example.\n', ...
                'Expected: %s\nReceived: %s'], ...
                k, task.name, expected{k}, expressions{k});
        end
    end
end

function s = canonical_expression_local(s)
    s = char(s);
    s = strrep(s, ' ', '');
    s = strrep(s, '.*', '*');
    s = strrep(s, './', '/');
    s = strrep(s, '.^', '^');
    s = lower(s);
end

function tf = empty_row_spec_local(rowSpec)
    tf = isempty(rowSpec);
    if ~tf && iscell(rowSpec)
        tf = all(cellfun(@isempty, rowSpec));
    end
end

function A = fill_exact_block_local(A, dictionaryTerms, rowTerms, rowCoefs, src, ell)
    for r = 1:min(size(A,1), numel(rowTerms))
        tr = rowTerms{r};
        cr = rowCoefs{r};
        if isempty(tr)
            continue;
        end
        if ischar(tr) || isstring(tr)
            tr = cellstr(tr);
        end
        cr = reshape(cr, 1, []);
        if numel(tr) ~= numel(cr)
            error('Term/coefficient count mismatch in Coef_M{%d,%d}, row %d.', ...
                src, ell, r);
        end
        for q = 1:numel(tr)
            col = find(strcmp(dictionaryTerms, char(tr{q})), 1);
            if isempty(col)
                error('Constructive term %s is missing from Coef_M{%d,%d}.', ...
                    char(tr{q}), src, ell);
            end
            A(r,col) = cr(q);
        end
    end
end


function [allCoords, nonChainCoords] = active_branch_coordinates_local(branchActive)
    allCoords = {};
    nonChainCoords = {};
    L = size(branchActive,2);
    for ell = 1:L
        for src = 1:min(ell,size(branchActive,1))
            if ~branchActive(src,ell)
                continue;
            end
            label = sprintf('Coef_M{%d,%d}',src,ell);
            allCoords{end+1} = label; %#ok<AGROW>
            if src > 1
                nonChainCoords{end+1} = label; %#ok<AGROW>
            end
        end
    end
end

function n = count_exact_nonzero_local(Coef)
    n = 0;
    for k = 1:numel(Coef)
        if ~isempty(Coef{k})
            n = n + nnz(Coef{k});
        end
    end
end

function value = get_field_local(S, name, defaultValue)
    if isstruct(S) && isfield(S, name) && ~isempty(S.(name))
        value = S.(name);
    else
        value = defaultValue;
    end
end
