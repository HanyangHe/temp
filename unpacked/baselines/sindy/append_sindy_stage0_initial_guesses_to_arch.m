function arch = append_sindy_stage0_initial_guesses_to_arch(arch, inputDim, guessTerms)
%APPEND_SINDY_STAGE0_INITIAL_GUESSES_TO_ARCH Add guesses to legacy Phi mode.
%
% The independent/general SINDy route performs this union inside
% make_sindy_general_arch. This helper preserves the same fairness invariant
% when the optional legacy dictionaryMode='phdn_phi' route is selected.

    if nargin < 3 || isempty(guessTerms)
        return;
    end
    if ~isfield(arch, 'caseDictionary') || ~isstruct(arch.caseDictionary)
        arch.caseDictionary = struct();
    end

    existing = explicit_case_dictionary_terms(inputDim, arch, 1, 1);
    out = existing(:);
    keys = cellfun(@canonical_key_local, out, 'UniformOutput', false);

    info = struct('enabled',true,'requestedTerms',{guessTerms(:)}, ...
        'nRequested',numel(guessTerms),'nAdded',0,'nDuplicate',0, ...
        'addedTerms',{{}},'duplicateTerms',{{}}, ...
        'libraryIndices',zeros(numel(guessTerms),1),'libraryTerms',{cell(numel(guessTerms),1)});

    added = cell(0,1);
    for k = 1:numel(guessTerms)
        term = strrep(strtrim(char(guessTerms{k})), ' ', '');
        key = canonical_key_local(term);
        idx = find(strcmp(keys,key),1);
        if isempty(idx)
            added{end+1,1} = term; %#ok<AGROW>
            out{end+1,1} = term; %#ok<AGROW>
            keys{end+1,1} = key; %#ok<AGROW>
            idx = numel(out);
            info.nAdded = info.nAdded + 1;
            info.addedTerms{end+1,1} = term; %#ok<AGROW>
        else
            info.nDuplicate = info.nDuplicate + 1;
            info.duplicateTerms{end+1,1} = term; %#ok<AGROW>
        end
        info.libraryIndices(k) = idx;
        info.libraryTerms{k} = out{idx};
    end

    if ~isempty(added)
        oldGlobal = {};
        if isfield(arch.caseDictionary,'globalTerms') && ~isempty(arch.caseDictionary.globalTerms)
            oldGlobal = arch.caseDictionary.globalTerms;
            if ischar(oldGlobal) || isstring(oldGlobal); oldGlobal = cellstr(oldGlobal); end
        end
        arch.caseDictionary.globalTerms = [oldGlobal(:); added(:)];
        arch.caseDictionary.appendGlobalTerms = true;
    end
    arch.caseDictionary.stage0InitialGuessUnion = info;
    if ~isfield(arch.caseDictionary,'source') || isempty(arch.caseDictionary.source)
        arch.caseDictionary.source = 'legacy PhDN Phi dictionary';
    end
    arch.caseDictionary.source = sprintf('%s + union(Stage0SRInitialGuesses: requested=%d, added=%d, duplicate=%d)', ...
        arch.caseDictionary.source, info.nRequested, info.nAdded, info.nDuplicate);

    if ~isfield(arch,'sindyDictionaryReport') || ~isstruct(arch.sindyDictionaryReport)
        arch.sindyDictionaryReport = struct();
    end
    arch.sindyDictionaryReport.stage0InitialGuessUnion = info;
    arch.sindyDictionaryReport.source = arch.caseDictionary.source;
end

function key = canonical_key_local(term)
    key = lower(strrep(strtrim(char(term)), ' ', ''));
    key = strrep(key, '**', '^');
    key = regexprep(key, 'square\((v[0-9]+)\)', '$1^2');
    key = regexprep(key, 'sqr\((v[0-9]+)\)', '$1^2');
end
