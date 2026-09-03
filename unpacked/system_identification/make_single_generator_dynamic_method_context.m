function context = make_single_generator_dynamic_method_context( ...
    task,samplingPlan,roundIndex,phdnOptions,baselineSweepOptions, ...
    outputCaseRoot,trainingSampleList,numRounds,stage0RandomState,baselineSeed)
%MAKE_SINGLE_GENERATOR_DYNAMIC_METHOD_CONTEXT Build a self-contained record context.
%
% Every independently persisted method result receives this context so that
% it can later be replayed, audited, or compared without depending on the
% aggregate result_pack.mat produced by a different run.

    context = struct();
    context.schemaVersion = 'single_generator_dynamic_experiment_context_v1';
    context.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
    context.matlabVersion = version;
    context.task = task;
    context.samplingPlan = samplingPlan;
    context.roundIndex = roundIndex;
    context.nTrain = samplingPlan.nTrain;
    context.phdnOptions = phdnOptions;
    context.baselineSweepOptions = baselineSweepOptions;
    context.outputCaseRoot = outputCaseRoot;
    context.trainingSampleList = trainingSampleList;
    context.numRounds = numRounds;
    context.stage0RandomState = stage0RandomState;
    context.baselineSeed = baselineSeed;

    context.identifiers = struct();
    context.identifiers.caseName = text_field_local(task,'name','SingleGeneratorDynamic_SMIB_AVR');
    context.identifiers.modelVariant = text_field_local(task,'modelVariant','');
    context.identifiers.variableMappingDescription = ...
        text_field_local(task,'variableMappingDescription','');
    context.identifiers.roundKey = sprintf('round_%02d',roundIndex);
    context.identifiers.sampleKey = sprintf('N_%05d',samplingPlan.nTrain);
end

function value = text_field_local(s,name,defaultValue)
    value = defaultValue;
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = char(string(s.(name)));
    end
end
