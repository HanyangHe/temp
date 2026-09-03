function [archOut, info] = augment_case_dictionary_for_normalization(archIn, normCfg)
%AUGMENT_CASE_DICTIONARY_FOR_NORMALIZATION Close compact priors under affine IO normalization.
%
% Case dictionaries are usually written in raw physical coordinates.  When the
% model is trained in normalized coordinates, the same physical support may need
% additional affine/bias terms.  This helper keeps normalization enabled and
% augments only the compact dictionary support needed by the current coordinate
% system.
%
% Semantics:
%   - rowTerms, if present, remain hard row-wise required support.  They are
%     handled by augment_rowwise_support_for_normalization().
%   - termsByBlock / termsByLayer / termsByDim are block/layer/dimension
%     candidate supports.  They are augmented as candidate supports, not as
%     row-wise hard masks.
%   - No old full-dictionary structure is restored here.

    archOut = archIn;
    info = struct('applied', false, 'mode', 'none', ...
        'inputAffineClosure', false, 'outputAffineClosure', false, ...
        'nTermsAdded', 0, 'message', '');

    if nargin < 2 || isempty(normCfg)
        info.message = 'No normalization configuration supplied.';
        return;
    end
    if ~isfield(archOut, 'caseDictionary') || ~isstruct(archOut.caseDictionary)
        info.message = 'No caseDictionary field.';
        return;
    end

    % Strong-prior rowTerms need row-wise hard-support closure, not blockwise
    % candidate closure.  Keep that existing logic untouched.
    if isfield(archOut.caseDictionary, 'rowTerms') && ~isempty(archOut.caseDictionary.rowTerms)
        [archOut, rowInfo] = augment_rowwise_support_for_normalization(archOut, normCfg);
        info.applied = isfield(rowInfo, 'applied') && rowInfo.applied;
        info.mode = 'rowwise_hard_support';
        if isfield(rowInfo, 'inputAffineClosure'); info.inputAffineClosure = rowInfo.inputAffineClosure; end
        if isfield(rowInfo, 'outputAffineClosure'); info.outputAffineClosure = rowInfo.outputAffineClosure; end
        if isfield(rowInfo, 'nTermsAddedToRows'); info.nTermsAdded = rowInfo.nTermsAddedToRows; end
        if isfield(rowInfo, 'message'); info.message = rowInfo.message; else; info.message = 'rowwise closure complete.'; end
        return;
    end

    useIOnorm = get_norm_flag_local(normCfg, 'useInputOutputNorm', false) || ...
        get_norm_flag_local(normCfg, 'applyToMLPSurrogate', false);
    useInputNorm = useIOnorm || get_norm_flag_local(normCfg, 'useInputNorm', false);
    useOutputNorm = useIOnorm || get_norm_flag_local(normCfg, 'useOutputNorm', false);
    if ~useInputNorm && ~useOutputNorm
        info.message = 'Affine IO normalization is disabled; no compact support augmentation needed.';
        return;
    end

    D = archOut.caseDictionary;
    L = archOut.layer;
    dims = get_arch_dims(archOut);
    nAdded = 0;

    % Branchwise compact dictionaries.
    if isfield(D, 'termsByBlock') && iscell(D.termsByBlock) && ~isempty(D.termsByBlock)
        for ell = 1:L
            for src = 1:ell
                if ~is_branch_active_local(archOut, src, ell)
                    continue;
                end
                if size(D.termsByBlock,1) < src || size(D.termsByBlock,2) < ell || isempty(D.termsByBlock{src,ell})
                    continue;
                end
                k = ell - src + 1;
                terms = canonical_term_cell_local(D.termsByBlock{src,ell});
                oldN = numel(terms);
                if useInputNorm && k == 1
                    terms = add_input_affine_closure_terms_local(terms, dims(k));
                    if numel(terms) > oldN
                        info.inputAffineClosure = true;
                    end
                end
                oldN2 = numel(terms);
                if useOutputNorm && ell == L
                    terms = ensure_constant_local(terms);
                    if numel(terms) > oldN2
                        info.outputAffineClosure = true;
                    end
                end
                D.termsByBlock{src,ell} = terms;
                nAdded = nAdded + max(0, numel(terms) - oldN);
            end
        end
    end

    % Layerwise compact dictionaries.  A layer dictionary is shared by all
    % active branches in that layer; if any input-source or final-output branch
    % needs affine closure, augment the shared layer support.
    if isfield(D, 'termsByLayer') && iscell(D.termsByLayer) && ~isempty(D.termsByLayer)
        for ell = 1:min(L, numel(D.termsByLayer))
            if isempty(D.termsByLayer{ell})
                continue;
            end
            terms = canonical_term_cell_local(D.termsByLayer{ell});
            oldN = numel(terms);
            needInputClosure = false;
            for src = 1:ell
                if ~is_branch_active_local(archOut, src, ell)
                    continue;
                end
                k = ell - src + 1;
                if useInputNorm && k == 1 && terms_need_input_affine_closure_local(terms, dims(k))
                    needInputClosure = true;
                end
            end
            if needInputClosure
                terms = add_input_affine_closure_terms_local(terms, max_dim_for_layer_local(dims, ell));
                info.inputAffineClosure = true;
            end
            if useOutputNorm && ell == L
                terms = ensure_constant_local(terms);
                info.outputAffineClosure = true;
            end
            D.termsByLayer{ell} = terms;
            nAdded = nAdded + max(0, numel(terms) - oldN);
        end
    end

    % Dimension-indexed compact dictionaries.  Add closure for dimensions that
    % are used by an input-source branch or by the final output layer.
    if isfield(D, 'termsByDim') && iscell(D.termsByDim) && ~isempty(D.termsByDim)
        for d = 1:numel(D.termsByDim)
            if isempty(D.termsByDim{d})
                continue;
            end
            terms = canonical_term_cell_local(D.termsByDim{d});
            oldN = numel(terms);
            [needInputClosure, needOutputClosure] = dim_needs_closure_local(archOut, dims, d, useInputNorm, useOutputNorm, terms);
            if needInputClosure
                terms = add_input_affine_closure_terms_local(terms, d);
                info.inputAffineClosure = true;
            end
            if needOutputClosure
                terms = ensure_constant_local(terms);
                info.outputAffineClosure = true;
            end
            D.termsByDim{d} = terms;
            nAdded = nAdded + max(0, numel(terms) - oldN);
        end
    end

    info.applied = nAdded > 0;
    info.mode = 'compact_candidate_support';
    info.nTermsAdded = nAdded;
    if info.applied
        info.message = sprintf('Added %d compact dictionary term(s) for affine normalization closure.', nAdded);
    else
        info.message = 'Compact dictionary already closed under affine normalization.';
    end

    D.normalizationAwareCompactTerms = true;
    D.normalizationAwareCompactTermsInfo = info;
    archOut.caseDictionary = D;
end

function tf = get_norm_flag_local(s, name, defaultVal)
    tf = defaultVal;
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        tf = logical(s.(name));
    end
end

function tf = is_branch_active_local(arch, src, ell)
    tf = true;
    if ~isfield(arch, 'branchActiveMask') || isempty(arch.branchActiveMask)
        return;
    end
    M = arch.branchActiveMask;
    try
        if iscell(M)
            if size(M,1) >= src && size(M,2) >= ell && ~isempty(M{src,ell})
                tf = logical(M{src,ell});
            end
        elseif isnumeric(M) || islogical(M)
            if size(M,1) >= src && size(M,2) >= ell
                tf = logical(M(src,ell));
            end
        end
    catch
        tf = true;
    end
end

function terms = canonical_term_cell_local(terms)
    if isempty(terms)
        terms = {};
        return;
    end
    if ischar(terms) || isstring(terms)
        terms = cellstr(terms);
    end
    terms = terms(:);
    out = {};
    for i = 1:numel(terms)
        name = strrep(strtrim(char(terms{i})), ' ', '');
        if isempty(name)
            continue;
        end
        % Accept x/h aliases, but store canonical v names.
        m = regexp(name, '[xh](\d+)', 'tokens');
        if ~isempty(m)
            idx = cellfun(@(z) str2double(z{1}), m);
            for k = max(idx):-1:1
                name = regexprep(name, sprintf('(?<![A-Za-z0-9_])[xh]%d(?![A-Za-z0-9_])', k), sprintf('v%d', k));
            end
        end
        out{end+1,1} = name; %#ok<AGROW>
    end
    terms = unique_stable_cell_local(out);
end

function terms = add_input_affine_closure_terms_local(terms, inputDim)
    terms = canonical_term_cell_local(terms);
    closure = terms;
    for i = 1:numel(terms)
        lowerTerms = lower_order_polynomial_terms_local(terms{i}, inputDim);
        closure = [closure(:); lowerTerms(:)]; %#ok<AGROW>
    end
    if terms_need_input_affine_closure_local(terms, inputDim)
        closure = [{'1'}; closure(:)];
    end
    terms = unique_stable_cell_local(closure);
end

function tf = terms_need_input_affine_closure_local(terms, inputDim)
    tf = false;
    for i = 1:numel(terms)
        name = char(terms{i});
        if strcmp(name, '1')
            continue;
        end
        if contains_polynomial_variable_local(name, inputDim)
            tf = true;
            return;
        end
    end
end

function tf = contains_polynomial_variable_local(name, inputDim)
    tf = false;
    if contains(name, '(')
        % Operator terms such as inv(v1) are not finite affine-polynomial
        % closures.  They are left unchanged here.
        return;
    end
    for k = 1:inputDim
        if ~isempty(regexp(name, sprintf('(?<![A-Za-z0-9_])v%d(?![A-Za-z0-9_])', k), 'once'))
            tf = true;
            return;
        end
    end
end

function lowerTerms = lower_order_polynomial_terms_local(termName, inputDim)
    lowerTerms = {};
    name = char(termName);
    if strcmp(name, '1') || contains(name, '(')
        return;
    end
    exps = zeros(1, inputDim);
    factors = strsplit(name, '*');
    for i = 1:numel(factors)
        f = strtrim(factors{i});
        if isempty(f); continue; end
        tok = regexp(f, '^v(\d+)(?:\^(\d+))?$', 'tokens', 'once');
        if isempty(tok)
            return;
        end
        idx = str2double(tok{1});
        if idx < 1 || idx > inputDim
            return;
        end
        if numel(tok) >= 2 && ~isempty(tok{2})
            p = str2double(tok{2});
        else
            p = 1;
        end
        if ~isfinite(p) || p < 0 || abs(p-round(p)) > 0
            return;
        end
        exps(idx) = exps(idx) + round(p);
    end
    if ~any(exps > 0)
        return;
    end
    activeIdx = find(exps > 0);
    gridVals = cell(1, numel(activeIdx));
    for k = 1:numel(activeIdx)
        gridVals{k} = 0:exps(activeIdx(k));
    end
    combos = cell(1, numel(activeIdx));
    [combos{:}] = ndgrid(gridVals{:});
    for ii = 1:numel(combos{1})
        subExp = zeros(1, inputDim);
        for k = 1:numel(activeIdx)
            subExp(activeIdx(k)) = combos{k}(ii);
        end
        if isequal(subExp, exps)
            continue;
        end
        lowerTerms{end+1,1} = monomial_name_local(subExp); %#ok<AGROW>
    end
end

function name = monomial_name_local(exps)
    if ~any(exps > 0)
        name = '1';
        return;
    end
    parts = {};
    for k = 1:numel(exps)
        p = exps(k);
        if p <= 0
            continue;
        elseif p == 1
            parts{end+1} = sprintf('v%d', k); %#ok<AGROW>
        else
            parts{end+1} = sprintf('v%d^%d', k, p); %#ok<AGROW>
        end
    end
    name = strjoin(parts, '*');
end

function terms = ensure_constant_local(terms)
    terms = canonical_term_cell_local(terms);
    if ~any(strcmp(terms, '1'))
        terms = unique_stable_cell_local([{'1'}; terms(:)]);
    end
end

function [needInputClosure, needOutputClosure] = dim_needs_closure_local(arch, dims, d, useInputNorm, useOutputNorm, terms)
    needInputClosure = false;
    needOutputClosure = false;
    L = arch.layer;
    for ell = 1:L
        for src = 1:ell
            if ~is_branch_active_local(arch, src, ell)
                continue;
            end
            k = ell - src + 1;
            if dims(k) ~= d
                continue;
            end
            if useInputNorm && k == 1 && terms_need_input_affine_closure_local(terms, d)
                needInputClosure = true;
            end
            if useOutputNorm && ell == L
                needOutputClosure = true;
            end
        end
    end
end

function dmax = max_dim_for_layer_local(dims, ell)
    dmax = 1;
    for src = 1:ell
        k = ell - src + 1;
        dmax = max(dmax, dims(k));
    end
end

function out = unique_stable_cell_local(in)
    out = {};
    for i = 1:numel(in)
        v = char(in{i});
        if isempty(v)
            continue;
        end
        if ~any(strcmp(out, v))
            out{end+1,1} = v; %#ok<AGROW>
        end
    end
end
