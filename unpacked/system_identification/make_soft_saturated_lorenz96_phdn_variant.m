function variant = make_soft_saturated_lorenz96_phdn_variant( ...
    label,fieldName,stage0Label,stage0FieldName, ...
    runEnabled,displayRecordedReport,initialGuesses,baseOpts,stage0WorkRoot,priorInfo)
%MAKE_SOFT_SATURATED_LORENZ96_PHDN_VARIANT Build one G-level PhDN method.
% G1/G2/G3 inherit common settings and differ only in prior and storage keys.

    if nargin < 10 || isempty(priorInfo)
        priorInfo = struct('label','main','theoreticalLevel',NaN, ...
            'name','user_stage0_initial_guesses','description','', ...
            'includesTransport',NaN,'includesDamping',NaN,'includesForcing',NaN);
    end
    variant = struct();
    variant.id = upper(strtrim(char(priorInfo.label)));
    variant.label = char(label);
    variant.fieldName = char(fieldName);
    variant.stage0Label = char(stage0Label);
    variant.stage0FieldName = char(stage0FieldName);
    variant.runEnabled = logical(runEnabled);
    variant.displayRecordedReport = logical(displayRecordedReport);
    variant.priorInfo = priorInfo;
    initialGuesses = normalize_initial_guesses_local(initialGuesses);
    hasInitialGuesses = ~isempty(initialGuesses);
    variant.initialGuesses = initialGuesses;
    variant.opts = baseOpts;
    variant.opts.stage0.pysr.initialGuessesEnable = hasInitialGuesses;
    variant.opts.stage0.pysr.initialGuesses = initialGuesses;
    % Lorenz public demos supply one cyclic prior expression per original
    % output.  Keep the normal multi-output PySR call shared exactly as before,
    % but record the one-to-one map for any later single-output rescue.
    if hasInitialGuesses
        variant.opts.stage0.pysr.initialGuessOutputMap = 1:numel(initialGuesses);
    else
        variant.opts.stage0.pysr.initialGuessOutputMap = [];
    end
    if ~hasInitialGuesses
        variant.opts.stage0.pysr.fractionReplacedGuesses = 0;
    end
    variant.opts.stage0.pysr.workRoot = fullfile(stage0WorkRoot,lower(variant.id));
    variant.opts.stage0.pysr.priorLevel = variant.id;
    variant.opts.stage0.pysr.priorTheoreticalLevel = priorInfo.theoreticalLevel;
    variant.opts.stage0.pysr.priorName = priorInfo.name;
    variant.opts.stage0.pysr.priorDescription = priorInfo.description;
    variant.opts.stage0.pysr.priorIncludesTransport = priorInfo.includesTransport;
    variant.opts.stage0.pysr.priorIncludesDamping = priorInfo.includesDamping;
    variant.opts.stage0.pysr.priorIncludesForcing = priorInfo.includesForcing;
    variant.opts.stage0.pysr.priorExpression = strjoin(initialGuesses,'; ');
end

function guesses = normalize_initial_guesses_local(value)
    guesses = {};
    if isempty(value); return; end
    if ischar(value); items = {value};
    elseif isstring(value); items = cellstr(value(:));
    elseif iscell(value); items = value(:);
    else; error('Stage0SRInitialGuesses must be char, string, or cellstr.'); end
    for iItem = 1:numel(items)
        item = items{iItem};
        if iscell(item) || (isstring(item) && ~isscalar(item))
            guesses = [guesses,normalize_initial_guesses_local(item)]; %#ok<AGROW>
        else
            term = strtrim(char(string(item)));
            if ~isempty(term) && ~any(strcmp(guesses,term)); guesses{end+1}=term; end %#ok<AGROW>
        end
    end
end
