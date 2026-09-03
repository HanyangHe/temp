function result = tag_single_generator_dynamic_phdn_prior_variant(result,variant)
%TAG_SINGLE_GENERATOR_DYNAMIC_PHDN_PRIOR_VARIANT Add auditable G-level metadata.

    result.methodFamily = 'phdn';
    result.phdnPriorLevel = variant.id;
    result.phdnMethodLabel = variant.label;
    result.methodLabel = variant.label;
    result.phdnPriorExpression = variant.y4Guess;
    result.phdnInitialGuesses = variant.initialGuesses;
    result.sindyMatchedPrior = strcmpi(variant.id,'G3');

    % Make an independently replayed report self-identifying.  Do not alter
    % the numerical report body or prepend the header more than once.
    if isfield(result,'recordedConsoleReport') && ...
            ~isempty(result.recordedConsoleReport)
        marker = sprintf('PhDN prior variant: %s',variant.id);
        reportText = char(result.recordedConsoleReport);
        if ~contains(reportText,marker)
            header = sprintf([ ...
                '============================================================\n', ...
                'PhDN prior variant: %s | method=%s\n', ...
                'Injected y4 structural guess: %s\n', ...
                'Ground-truth coefficients supplied: no; outer coefficients ', ...
                'are data-regression initializations.\n', ...
                '============================================================\n'], ...
                variant.id,variant.label,variant.y4Guess);
            result.recordedConsoleReport = [header reportText];
        end
    end
end
