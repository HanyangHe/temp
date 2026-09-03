function [sindyOpts, info] = sync_sindy_stage0_initial_guesses(sindyOpts, phdnOpts, inputDim)
%SYNC_SINDY_STAGE0_INITIAL_GUESSES Mirror Stage-0 SR guesses into SINDy.
%
% Fair-prior invariant for system-identification experiments:
% every expression supplied to official PySR through
%   phdnOpts.stage0.pysr.initialGuesses
% is also made available as one flat candidate function in the SINDy
% dictionary.  The expressions are shared across all outputs in both methods.
%
% PySR-facing xK/fixed-atom syntax is converted to the explicit SINDy vK
% syntax.  Only exact/canonical duplicate columns are removed; sparse sums are
% intentionally retained as grouped candidate functions even when their
% components also appear separately in the broad SINDy library.

    if nargin < 1 || isempty(sindyOpts)
        sindyOpts = sindy_default_options();
    end
    if nargin < 2
        phdnOpts = [];
    end
    if nargin < 3 || isempty(inputDim)
        inputDim = Inf;
    end

    info = struct();
    info.enabled = false;
    info.source = 'phdnOpts.stage0.pysr.initialGuesses';
    info.scope = 'shared_all_outputs';
    info.requestedTerms = {};
    info.normalizedTerms = {};
    info.nRequested = 0;
    info.nNormalizedUnique = 0;
    info.reason = 'Stage-0 PySR initial guesses are disabled or unavailable.';

    pysrOpts = get_nested_struct_local(phdnOpts, {'stage0','pysr'});
    if isempty(pysrOpts)
        sindyOpts.stage0InitialGuessTerms = {};
        sindyOpts.stage0InitialGuessSyncInfo = info;
        return;
    end

    enabled = logical(getfield_default_local(pysrOpts, 'initialGuessesEnable', false));
    raw = getfield_default_local(pysrOpts, 'initialGuesses', {});
    raw = normalize_cellstr_local(raw);
    info.requestedTerms = raw;
    info.nRequested = numel(raw);
    if ~enabled || isempty(raw)
        sindyOpts.stage0InitialGuessTerms = {};
        sindyOpts.stage0InitialGuessSyncInfo = info;
        return;
    end

    normalized = cell(0,1);
    keys = cell(0,1);
    for k = 1:numel(raw)
        term = normalize_guess_term_local(raw{k}, inputDim);
        key = canonical_key_local(term);
        if ~any(strcmp(keys, key))
            normalized{end+1,1} = term; %#ok<AGROW>
            keys{end+1,1} = key; %#ok<AGROW>
        end
    end

    existing = normalize_cellstr_local(getfield_default_local( ...
        sindyOpts, 'stage0InitialGuessTerms', {}));
    for k = 1:numel(existing)
        key = canonical_key_local(existing{k});
        if ~any(strcmp(keys, key))
            normalized{end+1,1} = existing{k}; %#ok<AGROW>
            keys{end+1,1} = key; %#ok<AGROW>
        end
    end

    info.enabled = true;
    info.normalizedTerms = normalized;
    info.nNormalizedUnique = numel(normalized);
    info.reason = sprintf(['Synchronized %d unique Stage-0 SR initial expression(s) ', ...
        'into the SINDy candidate-function union.'], numel(normalized));

    sindyOpts.stage0InitialGuessTerms = normalized;
    sindyOpts.stage0InitialGuessSyncInfo = info;
end

function term = normalize_guess_term_local(value, inputDim)
    term = strtrim(char(value));
    if isempty(term)
        error('Stage-0 initial guesses may not contain empty expressions.');
    end
    term = strrep(term, '**', '^');

    % Fixed typed atoms used by the PySR adapter become ordinary physical
    % functions in the flat SINDy dictionary.
    term = regexprep(term, 'sin_x([0-9]+)_atom', 'sin(x$1)');
    term = regexprep(term, 'cos_x([0-9]+)_atom', 'cos(x$1)');

    % Convert all supported user-facing coordinates to the explicit dictionary
    % convention v1,...,vd.  Descending replacement prevents x1 matching x10.
    if isfinite(inputDim)
        maxIndex = max(1, round(inputDim));
    else
        tokens = regexp(term, '(?<![A-Za-z0-9_])[xXhHvV]([0-9]+)(?![A-Za-z0-9_])', 'tokens');
        maxIndex = 0;
        for i = 1:numel(tokens)
            maxIndex = max(maxIndex, str2double(tokens{i}{1}));
        end
    end
    for idx = maxIndex:-1:1
        term = regexprep(term, sprintf('(?<![A-Za-z0-9_])[xXhH]%d(?![A-Za-z0-9_])', idx), sprintf('v%d', idx));
    end
    term = strrep(term, ' ', '');

    if contains(term, '_atom')
        error('Unsupported unresolved PySR fixed atom in SINDy guess term: %s', term);
    end

    tokens = regexp(term, '(?<![A-Za-z0-9_])v([0-9]+)(?![A-Za-z0-9_])', 'tokens');
    for i = 1:numel(tokens)
        idx = str2double(tokens{i}{1});
        if isfinite(inputDim) && idx > inputDim
            error('Stage-0 guess term "%s" uses v%d but the task has only %d inputs.', ...
                term, idx, inputDim);
        end
    end

    % Match the polynomial spelling used by the broad SINDy library for the
    % common direct-variable square seeds.  Composite squares remain explicit.
    term = regexprep(term, 'square\((v[0-9]+)\)', '$1^2');
    term = regexprep(term, 'sqr\((v[0-9]+)\)', '$1^2');
end

function key = canonical_key_local(term)
    key = lower(strrep(strtrim(char(term)), ' ', ''));
    key = strrep(key, '**', '^');
    key = regexprep(key, 'square\((v[0-9]+)\)', '$1^2');
    key = regexprep(key, 'sqr\((v[0-9]+)\)', '$1^2');
    key = strip_outer_parentheses_local(key);
end

function out = strip_outer_parentheses_local(in)
    out = in;
    changed = true;
    while changed && numel(out) >= 2 && out(1) == '(' && out(end) == ')'
        changed = false;
        depth = 0;
        enclosesAll = true;
        for k = 1:numel(out)
            if out(k) == '('
                depth = depth + 1;
            elseif out(k) == ')'
                depth = depth - 1;
                if depth == 0 && k < numel(out)
                    enclosesAll = false;
                    break;
                end
            end
        end
        if enclosesAll && depth == 0
            out = out(2:end-1);
            changed = true;
        end
    end
end

function s = get_nested_struct_local(root, path)
    s = root;
    for k = 1:numel(path)
        if ~isstruct(s) || ~isfield(s, path{k}) || isempty(s.(path{k}))
            s = [];
            return;
        end
        s = s.(path{k});
    end
    if ~isstruct(s)
        s = [];
    end
end

function c = normalize_cellstr_local(x)
    if isempty(x)
        c = {};
    elseif ischar(x)
        c = {x};
    elseif isstring(x)
        c = cellstr(x(:));
    elseif iscell(x)
        c = x(:);
    else
        error('Initial guess expressions must be char, string, or a cell array of strings.');
    end
    out = cell(0,1);
    for k = 1:numel(c)
        value = strtrim(char(string(c{k})));
        if ~isempty(value)
            out{end+1,1} = value; %#ok<AGROW>
        end
    end
    c = out;
end

function val = getfield_default_local(s, name, defaultVal)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = defaultVal;
    end
end
