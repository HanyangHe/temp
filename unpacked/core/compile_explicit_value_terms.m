function plan = compile_explicit_value_terms(termNames, arch, layerIndex, branchIndex)
%COMPILE_EXPLICIT_VALUE_TERMS Parse explicit dictionary terms once for rollout.
%
% This compiler creates a small value-only abstract syntax tree (AST) for each
% retained dictionary term. During ODE integration, no term text is reparsed,
% no dictionary metadata is rebuilt, and no Jacobian is evaluated.
%
% Unsupported terms are marked for the original evaluator as a correctness
% fallback. The current SingleGeneratorDynamic PhDN/SINDy grammar is fully
% covered by the compiled path.

    if nargin < 3
        layerIndex = [];
    end
    if nargin < 4
        branchIndex = [];
    end

    termNames = normalize_term_names_local(termNames);
    rawMask = structural_raw_mask_local(termNames, arch, layerIndex, branchIndex);

    epsVal = 1e-8;
    if isstruct(arch) && isfield(arch, 'safety') && isstruct(arch.safety) && ...
            isfield(arch.safety, 'eps') && ~isempty(arch.safety.eps)
        epsVal = arch.safety.eps;
    end

    nodes = cell(numel(termNames), 1);
    fallback = false(numel(termNames), 1);
    fallbackReason = cell(numel(termNames), 1);

    for k = 1:numel(termNames)
        try
            nodes{k} = parse_expr_local(termNames{k});
        catch ME
            nodes{k} = empty_node_local();
            fallback(k) = true;
            fallbackReason{k} = ME.message;
        end
    end

    plan = struct();
    plan.names = termNames;
    plan.nodes = nodes;
    plan.fallback = fallback;
    plan.fallbackReason = fallbackReason;
    plan.rawMask = rawMask;
    plan.arch = arch;
    plan.layerIndex = layerIndex;
    plan.branchIndex = branchIndex;
    plan.eps = epsVal;
    plan.nTerms = numel(termNames);
    plan.nFallback = nnz(fallback);
end

function names = normalize_term_names_local(names)
    if isempty(names)
        names = cell(0, 1);
        return;
    end
    if ischar(names)
        names = {names};
    elseif isstring(names)
        names = cellstr(names);
    elseif ~iscell(names)
        error('compile_explicit_value_terms:InvalidTermContainer', ...
            'termNames must be a character vector, string array, or cell array.');
    end

    names = names(:);
    for k = 1:numel(names)
        value = names{k};
        if isstring(value)
            if ~isscalar(value)
                error('Each dictionary term must be scalar text.');
            end
            value = char(value);
        elseif isnumeric(value) && isscalar(value) && isfinite(value)
            value = sprintf('%.17g', value);
        elseif ~ischar(value)
            error('Each dictionary term must be text or a finite numeric scalar.');
        end
        names{k} = strtrim(value);
    end
end

function mask = structural_raw_mask_local(termNames, arch, layerIndex, branchIndex)
%STRUCTURAL_RAW_MASK_LOCAL Match terms that must retain official-PySR semantics.
    mask = false(numel(termNames), 1);
    if isempty(termNames) || isempty(layerIndex) || isempty(branchIndex) || ...
            ~isstruct(arch) || ~isfield(arch, 'caseDictionary') || ...
            ~isstruct(arch.caseDictionary)
        return;
    end

    D = arch.caseDictionary;
    if ~isfield(D, 'structuralOperatorSemantics') || ...
            ~strcmpi(strtrim(char(D.structuralOperatorSemantics)), 'official_pysr_raw') || ...
            ~isfield(D, 'structuralTermsByBlock') || ...
            ~iscell(D.structuralTermsByBlock) || ...
            size(D.structuralTermsByBlock, 1) < branchIndex || ...
            size(D.structuralTermsByBlock, 2) < layerIndex
        return;
    end

    structural = D.structuralTermsByBlock{branchIndex, layerIndex};
    % Empty cells are normal for augmentation-only branches. The previous
    % patch attempted CELLFUN on numeric [], which caused the reported error.
    if isempty(structural)
        return;
    end
    if ischar(structural)
        structural = {structural};
    elseif isstring(structural)
        structural = cellstr(structural);
    elseif ~iscell(structural)
        % Nontext legacy placeholders (for example 0) do not define raw terms.
        return;
    end

    normalizedStructural = cell(numel(structural), 1);
    nValid = 0;
    structural = structural(:);
    for k = 1:numel(structural)
        value = structural{k};
        if isstring(value) && isscalar(value)
            value = char(value);
        end
        if ~ischar(value)
            continue;
        end
        nValid = nValid + 1;
        normalizedStructural{nValid} = normalize_term_text_local(value);
    end
    normalizedStructural = normalizedStructural(1:nValid);
    if isempty(normalizedStructural)
        return;
    end

    for k = 1:numel(termNames)
        mask(k) = any(strcmp(normalizedStructural, ...
            normalize_term_text_local(termNames{k})));
    end
end

function out = normalize_term_text_local(in)
    out = strrep(strtrim(char(in)), ' ', '');
end

function node = parse_expr_local(expr)
    expr = strrep(strtrim(char(expr)), ' ', '');
    expr = strrep(expr, '**', '^');
    expr = strip_outer_parentheses_local(expr);
    if isempty(expr)
        error('Empty expression.');
    end

    [parts, signs] = split_top_level_add_sub_local(expr);
    if numel(parts) > 1
        children = cell(numel(parts), 1);
        for k = 1:numel(parts)
            children{k} = parse_expr_local(parts{k});
        end
        node = make_node_local('sum');
        node.children = children;
        node.signs = signs(:).';
        return;
    end

    if expr(1) == '+' && numel(expr) > 1
        node = parse_expr_local(expr(2:end));
        return;
    elseif expr(1) == '-' && numel(expr) > 1 && ~isfinite(str2double(expr))
        node = make_node_local('neg');
        node.children = {parse_expr_local(expr(2:end))};
        return;
    end

    [parts, ops] = split_top_level_mul_div_local(expr);
    if numel(parts) > 1
        children = cell(numel(parts), 1);
        for k = 1:numel(parts)
            children{k} = parse_expr_local(parts{k});
        end
        node = make_node_local('muldiv');
        node.children = children;
        node.ops = ops;
        return;
    end

    [base, exponent, hasPower] = split_power_local(expr);
    if hasPower
        node = make_node_local('power');
        node.children = {parse_expr_local(base)};
        node.exponent = exponent;
        return;
    end

    [fname2, ~, ~, isBinary] = parse_binary_function_local(expr);
    if isBinary
        error('Unsupported explicit binary operator: %s', fname2);
    end

    [fname, arg, isUnary] = parse_function_local(expr);
    if isUnary
        supported = {'re','real','conj','conjugate','im','imag','inv', ...
            'square','sqr','cube','abs','sqrt','sqrt_abs','exp', ...
            'sin','cos','tanh','asin','log'};
        if ~any(strcmpi(fname, supported))
            chebOrder = parse_chebyshev_order_local(fname);
            if isempty(chebOrder)
                error('Unsupported explicit unary operator: %s', fname);
            end
        end
        node = make_node_local('unary');
        node.name = lower(fname);
        node.children = {parse_expr_local(arg)};
        return;
    end

    token = regexp(expr, '^v(\d+)$', 'tokens', 'once');
    if ~isempty(token)
        node = make_node_local('variable');
        node.index = str2double(token{1});
        return;
    end

    value = str2double(expr);
    if isfinite(value)
        node = make_node_local('constant');
        node.value = value;
        return;
    end

    error('Cannot parse explicit dictionary term: %s', expr);
end

function node = make_node_local(kind)
    node = struct('kind', kind, 'value', [], 'index', [], 'name', '', ...
        'children', {{}}, 'ops', '', 'signs', [], 'exponent', []);
end

function node = empty_node_local()
    node = make_node_local('fallback');
end

function [parts, ops] = split_top_level_mul_div_local(s)
    parts = {};
    ops = '';
    level = 0;
    startIndex = 1;
    for k = 1:numel(s)
        ch = s(k);
        if ch == '('
            level = level + 1;
        elseif ch == ')'
            level = level - 1;
        end
        if level == 0 && (ch == '*' || ch == '/')
            if ch == '*' && ((k < numel(s) && s(k+1) == '*') || ...
                    (k > 1 && s(k-1) == '*'))
                continue;
            end
            piece = s(startIndex:k-1);
            if ~isempty(piece)
                parts{end+1} = piece; %#ok<AGROW>
                ops(end+1) = ch; %#ok<AGROW>
            end
            startIndex = k + 1;
        end
    end
    piece = s(startIndex:end);
    if ~isempty(piece)
        parts{end+1} = piece; %#ok<AGROW>
    end
    if numel(parts) <= 1
        ops = '';
    end
end

function [parts, signs] = split_top_level_add_sub_local(s)
    parts = {};
    signs = [];
    level = 0;
    startIndex = 1;
    currentSign = 1;
    for k = 1:numel(s)
        ch = s(k);
        if ch == '('
            level = level + 1;
        elseif ch == ')'
            level = level - 1;
        end
        if level == 0 && (ch == '+' || ch == '-') && ...
                is_binary_add_sub_sign_local(s, k)
            piece = s(startIndex:k-1);
            if ~isempty(piece)
                parts{end+1} = piece; %#ok<AGROW>
                signs(end+1) = currentSign; %#ok<AGROW>
            end
            currentSign = 1;
            if ch == '-'
                currentSign = -1;
            end
            startIndex = k + 1;
        end
    end
    piece = s(startIndex:end);
    if ~isempty(piece)
        parts{end+1} = piece; %#ok<AGROW>
        signs(end+1) = currentSign; %#ok<AGROW>
    end
end

function tf = is_binary_add_sub_sign_local(s, position)
    if position <= 1
        tf = false;
        return;
    end
    previous = s(position - 1);
    tf = ~any(previous == ['(', '*', '/', '^', '+', '-', 'e', 'E']);
end

function [base, exponent, hasPower] = split_power_local(s)
    level = 0;
    position = 0;
    for k = numel(s):-1:1
        ch = s(k);
        if ch == ')'
            level = level + 1;
        elseif ch == '('
            level = level - 1;
        elseif level == 0 && ch == '^'
            position = k;
            break;
        end
    end
    if position == 0
        base = '';
        exponent = NaN;
        hasPower = false;
        return;
    end
    base = s(1:position-1);
    exponentText = strip_outer_parentheses_local(strtrim(s(position+1:end)));
    exponent = parse_numeric_exponent_local(exponentText);
    hasPower = isfinite(exponent);
end

function exponent = parse_numeric_exponent_local(text)
    text = strip_outer_parentheses_local(strtrim(char(text)));
    exponent = str2double(text);
    if isfinite(exponent)
        return;
    end
    token = regexp(text, ...
        '^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$', ...
        'tokens', 'once');
    if isempty(token)
        exponent = NaN;
        return;
    end
    numerator = str2double(token{1});
    denominator = str2double(token{2});
    if ~isfinite(numerator) || ~isfinite(denominator) || denominator == 0
        exponent = NaN;
    else
        exponent = numerator / denominator;
    end
end

function [fname, arg1, arg2, isFunction] = parse_binary_function_local(s)
    match = regexp(s, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
    if isempty(match)
        fname = '';
        arg1 = '';
        arg2 = '';
        isFunction = false;
        return;
    end
    inside = match{2};
    level = 0;
    commaIndex = [];
    for k = 1:numel(inside)
        if inside(k) == '('
            level = level + 1;
        elseif inside(k) == ')'
            level = level - 1;
        elseif inside(k) == ',' && level == 0
            if ~isempty(commaIndex)
                fname = '';
                arg1 = '';
                arg2 = '';
                isFunction = false;
                return;
            end
            commaIndex = k;
        end
    end
    if isempty(commaIndex) || level ~= 0
        fname = '';
        arg1 = '';
        arg2 = '';
        isFunction = false;
        return;
    end
    fname = match{1};
    arg1 = inside(1:commaIndex-1);
    arg2 = inside(commaIndex+1:end);
    isFunction = ~isempty(arg1) && ~isempty(arg2) && ...
        parentheses_balanced_local(arg1) && parentheses_balanced_local(arg2);
end

function [fname, arg, isFunction] = parse_function_local(s)
    match = regexp(s, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
    if isempty(match)
        fname = '';
        arg = '';
        isFunction = false;
        return;
    end
    fname = match{1};
    arg = match{2};
    isFunction = parentheses_balanced_local(arg);
end

function order = parse_chebyshev_order_local(fname)
    token = regexp(char(fname), '^[Tt](\d+)$', 'tokens', 'once');
    if isempty(token)
        order = [];
        return;
    end
    order = str2double(token{1});
    if ~isfinite(order) || order < 0 || abs(order - round(order)) > 0
        order = [];
    else
        order = round(order);
    end
end

function tf = parentheses_balanced_local(s)
    level = 0;
    tf = true;
    for k = 1:numel(s)
        if s(k) == '('
            level = level + 1;
        elseif s(k) == ')'
            level = level - 1;
        end
        if level < 0
            tf = false;
            return;
        end
    end
    tf = level == 0;
end

function s = strip_outer_parentheses_local(s)
    while numel(s) >= 2 && s(1) == '(' && s(end) == ')' && ...
            parentheses_balanced_local(s(2:end-1))
        s = s(2:end-1);
    end
end
