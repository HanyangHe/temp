function [archOut, info] = augment_rowwise_support_for_normalization(archIn, normCfg)
%AUGMENT_ROWWISE_SUPPORT_FOR_NORMALIZATION Make rowwise hard support closed under IO normalization.
%
% Row-wise strong priors are written in the task file as physical support
% identities.  When affine input/output normalization is enabled, the same
% physical identities may require bias channels in the normalized-coordinate
% coefficient model.  This helper keeps the current compact rowwise interface
% but augments only the necessary constant terms before coefficient templates
% and masks are created.
%
% It does NOT disable normalization and it does NOT turn rowwise support into a
% blockwise optional dictionary.  It only preserves the hard-support semantics
% under the coordinate system actually used by model_forward.

    archOut = archIn;
    info = struct('applied', false, 'inputAffineClosure', false, 'outputAffineClosure', false, ...
        'nTermsAddedToDictionary', 0, 'nTermsAddedToRows', 0, 'message', '');

    if nargin < 2 || isempty(normCfg)
        info.message = 'No normalization configuration supplied.';
        return;
    end
    if ~isfield(archOut, 'caseDictionary') || ~isstruct(archOut.caseDictionary) || ...
            ~isfield(archOut.caseDictionary, 'rowTerms') || isempty(archOut.caseDictionary.rowTerms)
        info.message = 'No rowTerms hard support.';
        return;
    end

    useIOnorm = get_norm_flag_local(normCfg, 'useInputOutputNorm', false) || ...
        get_norm_flag_local(normCfg, 'applyToMLPSurrogate', false);
    useInputNorm = useIOnorm || get_norm_flag_local(normCfg, 'useInputNorm', false);
    useOutputNorm = useIOnorm || get_norm_flag_local(normCfg, 'useOutputNorm', false);
    if ~useInputNorm && ~useOutputNorm
        info.message = 'Affine IO normalization is disabled; no rowwise support augmentation needed.';
        return;
    end

    D = archOut.caseDictionary;
    if ~isfield(D, 'termsByBlock') || isempty(D.termsByBlock)
        info.message = 'caseDictionary.termsByBlock is unavailable.';
        return;
    end
    R = D.rowTerms;
    L = archOut.layer;
    dims = get_arch_dims(archOut);

    addedDict = 0;
    addedRows = 0;

    for ell = 1:L
        for src = 1:ell
            if ~is_branch_active_local(archOut, src, ell)
                continue;
            end
            if size(R,1) < src || size(R,2) < ell || isempty(R{src, ell}) || ~iscell(R{src, ell})
                continue;
            end
            rowSpec = R{src, ell};
            k = ell - src + 1;
            sourceIsInput = (k == 1);
            isFinalLayer = (ell == L);

            % Input affine normalization: a physical affine-linear row such as
            % x3-x4 becomes c0 + c1*x3n + c2*x4n in normalized coordinates.
            if useInputNorm && sourceIsInput
                for r = 1:numel(rowSpec)
                    termsR = canonical_term_cell_local(rowSpec{r});
                    if isempty(termsR)
                        continue;
                    end
                    if row_needs_input_bias_closure_local(termsR, dims(k)) && ~any(strcmp(termsR, '1'))
                        [D, addedD] = add_term_to_block_dictionary_local(D, src, ell, '1');
                        termsR = [{'1'}; termsR(:)];
                        rowSpec{r} = unique_stable_cell_local(termsR);
                        addedDict = addedDict + addedD;
                        addedRows = addedRows + 1;
                        info.inputAffineClosure = true;
                    end
                end
            end

            % Output affine normalization: model_forward denormalizes the final
            % layer output.  A normalized-coordinate final output may need an
            % additive constant to represent raw y exactly after denormalization.
            if useOutputNorm && isFinalLayer
                for r = 1:numel(rowSpec)
                    termsR = canonical_term_cell_local(rowSpec{r});
                    if isempty(termsR)
                        continue;
                    end
                    if ~any(strcmp(termsR, '1'))
                        [D, addedD] = add_term_to_block_dictionary_local(D, src, ell, '1');
                        termsR = [{'1'}; termsR(:)];
                        rowSpec{r} = unique_stable_cell_local(termsR);
                        addedDict = addedDict + addedD;
                        addedRows = addedRows + 1;
                        info.outputAffineClosure = true;
                    end
                end
            end

            R{src, ell} = rowSpec;
        end
    end

    D.rowTerms = R;
    D.normalizationAwareRowTerms = true;
    D.normalizationAwareRowTermsInfo = info;
    archOut.caseDictionary = D;
    info.applied = (addedDict > 0 || addedRows > 0);
    info.nTermsAddedToDictionary = addedDict;
    info.nTermsAddedToRows = addedRows;
    if info.applied
        info.message = sprintf('Added %d dictionary term(s) and %d row support bias channel(s) for affine normalization closure.', addedDict, addedRows);
    else
        info.message = 'Rowwise support already closed under affine normalization.';
    end
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
        out{end+1,1} = name; %#ok<AGROW>
    end
    terms = unique_stable_cell_local(out);
end

function tf = row_needs_input_bias_closure_local(termsR, inputDim)
    tf = false;
    for i = 1:numel(termsR)
        name = char(termsR{i});
        if strcmp(name, '1')
            continue;
        end
        if is_first_order_variable_term_local(name, inputDim)
            tf = true;
            return;
        end
    end
end

function tf = is_first_order_variable_term_local(name, inputDim)
    tf = false;
    for k = 1:inputDim
        if strcmp(name, sprintf('v%d', k)) || strcmp(name, sprintf('x%d', k)) || strcmp(name, sprintf('h%d', k))
            tf = true;
            return;
        end
    end
end

function [D, added] = add_term_to_block_dictionary_local(D, src, ell, termName)
    added = 0;
    if ~isfield(D, 'termsByBlock') || isempty(D.termsByBlock)
        return;
    end
    if size(D.termsByBlock,1) < src || size(D.termsByBlock,2) < ell || isempty(D.termsByBlock{src,ell})
        D.termsByBlock{src,ell} = {termName};
        added = 1;
        return;
    end
    termsB = canonical_term_cell_local(D.termsByBlock{src,ell});
    if ~any(strcmp(termsB, termName))
        termsB = [{termName}; termsB(:)];
        D.termsByBlock{src,ell} = unique_stable_cell_local(termsB);
        added = 1;
    end
end

function out = unique_stable_cell_local(in)
    out = {};
    for i = 1:numel(in)
        v = char(in{i});
        if ~any(strcmp(out, v))
            out{end+1,1} = v; %#ok<AGROW>
        end
    end
end
