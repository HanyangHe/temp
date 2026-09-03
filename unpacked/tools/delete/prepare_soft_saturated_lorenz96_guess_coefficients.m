function report = prepare_soft_saturated_lorenz96_guess_coefficients(varargin)
%PREPARE_SOFT_SATURATED_LORENZ96_GUESS_COEFFICIENTS Demo-matched L96 guess helper.
%
% This wrapper intentionally mirrors the flexible/manual workflow used by
% prepare_single_generator_dynamic_guess_coefficients.  The user supplies an
% ARBITRARY set of coefficient-free basis expressions and an explicit target
% output index.  No Lorenz-specific G2/G3 term pattern is assumed by this
% function; it only reproduces the matched Lorenz--96 training samples and
% fits outer linear coefficients for whatever bases the user requests.
%
% Preferred generator-style call:
%   report = prepare_soft_saturated_lorenz96_guess_coefficients( ...
%       {'1','x1','x10*(x2-x9)'},1);
%
% Arbitrary user-defined combinations are allowed, e.g.
%   report = prepare_soft_saturated_lorenz96_guess_coefficients( ...
%       {'1','sin(x3)','x2*x7','sqrt(square(x5)+square(x9))'},4);
%
% A convenience vararg form is also supported.  The LAST non-struct input
% must be the explicit numeric output index:
%   report = prepare_soft_saturated_lorenz96_guess_coefficients( ...
%       '1','sin(x3)','x2*x7',4);
%
% Optional settings are supplied by a trailing struct in either form:
%   report = prepare_soft_saturated_lorenz96_guess_coefficients( ...
%       {'1','x1','x10*(x2-x9)'},1,struct('ridgeLambda',1e-12));
%
% Supported expressions are exactly those accepted by
% prepare_stage0_guess_linear_coefficients.  Thus this helper can be used for
% G2/G3 or any alternative prior structure a user wishes to test.
%
% The target output index is NEVER inferred from the bases.  This is
% intentional: a valid user guess does not need to contain x_i, a damping
% term, a constant, or any other fixed Lorenz template.
%
% Demo-matched default data settings are K=10, F=8, kappa=1, clean N=250,
% and the current Sobol seeds/scramble.  They can be overridden through the
% trailing options struct (K, F, kappa, nTrain, nValidation, nTest, maxTrain,
% trainSeed, validationSeed, testSeed, oodSeed, sobolScrambleMethod,
% sobolSkip), in addition to the fitting options understood by
% prepare_stage0_guess_linear_coefficients.
%
% This is a preparation/diagnostic tool only. It never edits the main or
% derivative-noise robustness demos. Review/round/copy the printed expression
% into Stage0SRInitialGuesses manually, exactly as in the generator workflow.

    [basisExpressions,outputIndex,userOptions] = parse_inputs_local(varargin{:});

    % Keep these defaults synchronized with the corresponding L96 demos.
    K = get_option_local(userOptions,'K',10);
    F = get_option_local(userOptions,'F',8);
    kappa = get_option_local(userOptions,'kappa',1);
    nTrain = get_option_local(userOptions,'nTrain',250);
    nValidation = get_option_local(userOptions,'nValidation',500);
    nTest = get_option_local(userOptions,'nTest',1000);
    maxTrain = get_option_local(userOptions,'maxTrain',2000);
    trainSeed = get_option_local(userOptions,'trainSeed',7301);
    validationSeed = get_option_local(userOptions,'validationSeed',7401);
    testSeed = get_option_local(userOptions,'testSeed',7501);
    oodSeed = get_option_local(userOptions,'oodSeed',8501);
    sobolScrambleMethod = char(get_option_local(userOptions, ...
        'sobolScrambleMethod','MatousekAffineOwen'));
    sobolSkip = get_option_local(userOptions,'sobolSkip',1024);

    validateattributes(K,{'numeric'},{'scalar','integer','>=',4});
    validateattributes(F,{'numeric'},{'scalar','real','finite'});
    validateattributes(kappa,{'numeric'},{'scalar','real','finite','positive'});
    validateattributes(nTrain,{'numeric'},{'scalar','integer','positive'});
    validateattributes(maxTrain,{'numeric'},{'scalar','integer','>=',nTrain});
    validateattributes(outputIndex,{'numeric'}, ...
        {'scalar','integer','positive','<=',K},mfilename,'outputIndex');

    caseMode = 'general';
    caseToRun = sprintf('SS_L96_K%d',K);
    task = task_soft_saturated_lorenz96(caseToRun,caseMode,F,kappa);

    plan = struct();
    plan.nTrain = nTrain;
    plan.nValidation = nValidation;
    plan.nTest = nTest;
    plan.maxTrain = maxTrain;
    plan.trainSeed = trainSeed;
    plan.validationSeed = validationSeed;
    plan.testSeed = testSeed;
    plan.oodSeed = oodSeed;
    plan.samplingMethod = 'scrambled_sobol';
    plan.sobolScrambleMethod = sobolScrambleMethod;
    plan.sobolSkip = sobolSkip;
    task.samplingPlan = plan;
    task.sampleFcn = @(n,domain) sample_soft_saturated_lorenz96_split(n,domain,plan);

    report = prepare_stage0_guess_linear_coefficients( ...
        task,plan,outputIndex,basisExpressions,userOptions);

    % Metadata only; no structure/prior-level interpretation is imposed.
    report.trueSystemForcing = F;
    report.trueSystemKappa = kappa;
    report.inputSemantics = 'arbitrary_user_supplied_basis_set';
end

function [basisExpressions,outputIndex,userOptions] = parse_inputs_local(varargin)
    userOptions = struct();
    args = varargin;

    if ~isempty(args) && isstruct(args{end})
        userOptions = args{end};
        args(end) = [];
    end

    if isempty(args)
        % Mirror the generator helper's convenient default-example behavior.
        basisExpressions = {'1','x1','x10*(x2-x9)'};
        outputIndex = 1;
        return;
    end

    % Preferred generator-style form: (basisList, outputIndex [, options]).
    if numel(args) == 2 && isnumeric(args{2}) && isscalar(args{2})
        basisExpressions = normalize_basis_list_local(args{1});
        outputIndex = round(double(args{2}));
        return;
    end

    % Flexible convenience form: expr1, expr2, ..., outputIndex [, options].
    % The output index is explicit and always last; bases are otherwise
    % completely arbitrary and no term is interpreted as damping/forcing.
    if numel(args) >= 2 && isnumeric(args{end}) && isscalar(args{end})
        outputIndex = round(double(args{end}));
        basisExpressions = normalize_basis_list_local(args(1:end-1));
        return;
    end

    error([ ...
        'Specify an explicit numeric target output index. Use either ', ...
        'prepare_soft_saturated_lorenz96_guess_coefficients({bases},outputIndex) ', ...
        'or prepare_soft_saturated_lorenz96_guess_coefficients(expr1,...,outputIndex).']);
end

function basisList = normalize_basis_list_local(value)
    if ischar(value) || (isstring(value) && isscalar(value))
        basisList = {char(value)};
    elseif isstring(value)
        basisList = cellstr(value(:));
    elseif iscell(value)
        basisList = value(:).';
    else
        error('Basis inputs must be char, string, or a cell array of expressions.');
    end
    for k = 1:numel(basisList)
        basisList{k} = strtrim(char(string(basisList{k})));
        if isempty(basisList{k})
            error('Basis expression %d is empty.',k);
        end
    end
end

function value = get_option_local(options,name,defaultValue)
    if isstruct(options) && isfield(options,name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = defaultValue;
    end
end
