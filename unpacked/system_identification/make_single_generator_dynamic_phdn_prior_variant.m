function variant = make_single_generator_dynamic_phdn_prior_variant( ...
    id,label,fieldName,stage0Label,stage0FieldName, ...
    runEnabled,displayRecordedReport,initialGuesses,baseOpts,stage0WorkRoot)
%MAKE_SINGLE_GENERATOR_DYNAMIC_PHDN_PRIOR_VARIANT Build one G-level PhDN method.
%
% All settings are inherited from BASEOPTS.  Only the initial-guess library,
% prior metadata, and isolated PySR work root differ across G1/G2/G3.

    variant = struct();
    variant.id = char(id);
    variant.label = char(label);
    variant.fieldName = char(fieldName);
    variant.stage0Label = char(stage0Label);
    variant.stage0FieldName = char(stage0FieldName);
    variant.runEnabled = logical(runEnabled);
    variant.displayRecordedReport = logical(displayRecordedReport);
    variant.initialGuesses = initialGuesses;
    variant.y4Guess = initialGuesses{end};
    variant.opts = baseOpts;
    variant.opts.stage0.pysr.initialGuesses = initialGuesses;
    % The two generator prior expressions correspond to domega_dot (y2) and
    % Efd_dot (y4).  The ordinary multi-output search still receives both as a
    % shared soft library; the mapping is used only if a single-output rescue
    % is triggered, so the rescue can use the physically matched guess.
    if numel(initialGuesses) >= 2
        % Any leading library entries are treated as genuinely shared (map=0);
        % the final two case-local sparse combinations are y2 and y4.
        variant.opts.stage0.pysr.initialGuessOutputMap = ...
            [zeros(1,numel(initialGuesses)-2),2,4];
    else
        variant.opts.stage0.pysr.initialGuessOutputMap = [];
    end
    variant.opts.stage0.pysr.workRoot = fullfile( ...
        stage0WorkRoot,lower(char(id)));
    variant.opts.stage0.pysr.priorLevel = char(id);
    variant.opts.stage0.pysr.priorExpression = variant.y4Guess;
end
