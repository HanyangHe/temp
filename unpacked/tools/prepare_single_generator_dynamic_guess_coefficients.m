function report = prepare_single_generator_dynamic_guess_coefficients(basisExpressions, outputIndex, userOptions)
%PREPARE_SINGLE_GENERATOR_DYNAMIC_GUESS_COEFFICIENTS Demo-matched guess helper.
%
% This case-specific wrapper reproduces the current
% run_demo_single_generator_dynamic_stage012 sampling configuration and calls
% prepare_stage0_guess_linear_coefficients. Edit the compact settings below
% whenever the corresponding demo's sample-count or Sobol settings change.
%
% Example for the current weak G1/G2 bridge:
%   report = prepare_single_generator_dynamic_guess_coefficients( ...
%       {'1','x4','sqrt(square(x3)+square(x1))'},4);
%
% Example with the correct trigonometric branch type:
%   report = prepare_single_generator_dynamic_guess_coefficients( ...
%       {'1','x4','sqrt(square(x3)+square(sin(x1)))'},4);
%
% The printed expression can be copied into Stage0SRInitialGuesses. The same
% expression will then be mirrored into the matched SINDy baseline dictionary
% by the existing Stage0-initial-guess synchronization mechanism.

    if nargin < 1 || isempty(basisExpressions)
        basisExpressions = {'1','x4','sqrt(square(x3)+square(x1))'};
    end
    if nargin < 2 || isempty(outputIndex); outputIndex = 4; end
    if nargin < 3 || isempty(userOptions); userOptions = struct(); end

    % Keep these settings synchronized with the corresponding case demo.
    caseMode = 'general';
    caseToRun = 'SMIB_AVR';
    nTrain = 250;
    nValidation = 500;
    nTest = 1000;
    maxTrain = nTrain;
    samplingMethod = 'scrambled_sobol';
    sobolScrambleMethod = 'MatousekAffineOwen';
    sobolSkip = 1024;
    trainSeed = 7301;
    validationSeed = 7401;
    testSeed = 7501;
    oodSeed = 8501;

    task = task_single_generator_dynamic(caseToRun,caseMode);
    plan = struct();
    plan.nTrain = nTrain;
    plan.nValidation = nValidation;
    plan.nTest = nTest;
    plan.maxTrain = maxTrain;
    plan.trainSeed = trainSeed;
    plan.validationSeed = validationSeed;
    plan.testSeed = testSeed;
    plan.oodSeed = oodSeed;
    plan.samplingMethod = samplingMethod;
    plan.sobolScrambleMethod = sobolScrambleMethod;
    plan.sobolSkip = sobolSkip;

    task.samplingPlan = plan;
    task.sampleFcn = @(n,domain) sample_single_generator_dynamic_split(n,domain,plan);

    report = prepare_stage0_guess_linear_coefficients( ...
        task,plan,outputIndex,basisExpressions,userOptions);
end
