function task = task_symbolic_representability_nested_2d()
%TASK_SYMBOLIC_REPRESENTABILITY_NESTED_2D Deep two-input scalar expression.
%
% Target:
%   y = sqrt(log(1+exp(x1^2+0.5*x2^2))+0.25) ...
%       + sin(exp(0.5*x1*x2))/(1+x1^2+x2^2)
%
% The case uses a deeper composition than the main example while retaining
% only two inputs and one scalar output.

    task = struct();
    task.name = 'SymbolicRepresentability_Nested2D';
    task.caseName = 'symbolic_representability_nested_2d';
    task.casemode = 'strong_prior';
    task.sourceName = 'manuscript_constructive_symbolic_representability';
    task.description = ['Deep two-input scalar DAG with nested exp-log-sqrt and ', ...
        'a multiplicatively coupled sin-exp reciprocal branch.'];

    task.nx = 2;
    task.ny = 1;
    task.variableNames = {'x1','x2'};
    task.outputNames = {'y'};
    task.symbolicExpressions = { ...
        'sqrt(log(1+exp(x1^2+0.5*x2^2))+0.25)+sin(exp(0.5*x1*x2))/(1+x1^2+x2^2)' ...
    };

    task.dataDefaults = struct('nTrain',1500,'nValidation',500, ...
        'nIDTest',500,'nOODSamples',500);

    task.sampling = struct();
    task.sampling.method = 'global_randomized_lhs_with_validity_rejection';
    task.sampling.inverseDenominatorMargin = 5e-2;
    task.sampling.logArgumentMargin = 1e-4;
    task.sampling.sqrtArgumentMargin = 1e-8;
    task.sampling.maximumExpArgument = 50;
    task.sampling.maximumAbsoluteTarget = Inf;
    task.sampling.maxBatches = 100;
    task.sampling.minimumReplacementBatchSize = 32;
    task.sampling.replacementOversampleFactor = 1.25;
    task.sampleValidityFcn = @sample_validity_nested_2d_local;

    task.domain = make_variable_union_domain(task.variableNames, { ...
        [-1.00,1.00], ...
        [-1.00,1.00] ...
    });
    task.domain.source = 'constructive_nested_2d_id';

    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [-1.60,-1.20;1.20,1.60], ...
        [-1.60,-1.20;1.20,1.60] ...
    });
    task.oodDomain.source = 'constructive_nested_2d_disjoint_ood';

    task.rhsFcn = @rhs_nested_2d_local;

    task.arch = struct();
    task.arch.layer = 6;
    task.arch.hiddenDims = [3,3,3,2,2];
    task.arch.nx = task.nx;
    task.arch.ny = task.ny;
    task.arch.operatorMode = 'true';
    task.arch.dictionaryMode = 'manual_constructive_exact_dag';

    task.prior = struct();
    task.prior.priorInterfaceEnabled = true;
    task.prior.level = 4;
    task.prior.levelName = 'strong_prior';
    task.prior.dictionaryMode = 'rowwise_exact_constructive_support';
    task.prior.useAdmissibleMask = true;
    task.prior.maskType = 'rowwise_exact_constructive_support';
end

function Y = rhs_nested_2d_local(X)
    x1 = X(:,1);
    x2 = X(:,2);

    a = x1.^2 + 0.5.*x2.^2;
    b = 0.5.*x1.*x2;
    d = 1 + x1.^2 + x2.^2;
    Y = sqrt(log(1 + exp(a)) + 0.25) + sin(exp(b))./d;
end

function [valid, details] = sample_validity_nested_2d_local(X, options)
    x1 = X(:,1);
    x2 = X(:,2);

    inverseMargin = get_sampling_option_local(options, ...
        'inverseDenominatorMargin',5e-2);
    logMargin = get_sampling_option_local(options,'logArgumentMargin',1e-4);
    sqrtMargin = get_sampling_option_local(options,'sqrtArgumentMargin',1e-8);
    maxExpArgument = get_sampling_option_local(options,'maximumExpArgument',50);

    expArg1 = x1.^2 + 0.5.*x2.^2;
    expArg2 = 0.5.*x1.*x2;
    denominator = 1 + x1.^2 + x2.^2;

    finiteInputs = all(isfinite(X),2);
    expSafe = isfinite(expArg1) & isfinite(expArg2) & ...
        abs(expArg1) <= maxExpArgument & abs(expArg2) <= maxExpArgument;
    denominatorSafe = denominator >= inverseMargin;

    logArgument = nan(size(expArg1));
    logArgument(expSafe) = 1 + exp(expArg1(expSafe));
    logSafe = isfinite(logArgument) & logArgument >= logMargin;

    sqrtArgument = nan(size(expArg1));
    sqrtArgument(logSafe) = log(logArgument(logSafe)) + 0.25;
    sqrtSafe = isfinite(sqrtArgument) & sqrtArgument >= sqrtMargin;

    valid = finiteInputs & expSafe & denominatorSafe & logSafe & sqrtSafe;

    details = struct();
    details.expArgument1 = expArg1;
    details.expArgument2 = expArg2;
    details.denominator = denominator;
    details.logArgument = logArgument;
    details.sqrtArgument = sqrtArgument;
end

function value = get_sampling_option_local(options, fieldName, defaultValue)
    if isstruct(options) && isfield(options,fieldName) && ~isempty(options.(fieldName))
        value = options.(fieldName);
    else
        value = defaultValue;
    end
end
