function result = tag_single_generator_dynamic_stage0_prior_variant(result,variant)
%TAG_SINGLE_GENERATOR_DYNAMIC_STAGE0_PRIOR_VARIANT Tag one Stage0-SR ablation.

    result.methodFamily = 'stage0-sr';
    result.phdnPriorLevel = variant.id;
    result.parentPhdnMethodLabel = variant.label;
    result.methodLabel = variant.stage0Label;
    result.phdnPriorExpression = variant.y4Guess;
    result.phdnInitialGuesses = variant.initialGuesses;
    result.sindyMatchedPrior = strcmpi(variant.id,'G3');
end
