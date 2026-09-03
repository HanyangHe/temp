function task = task_symbolic_representability_dense_skip_2d()
%TASK_SYMBOLIC_REPRESENTABILITY_DENSE_SKIP_2D Moderate two-input scalar DAG.
%
% Let
%   a = 1+x1^2+x2^2,          b = 0.5*x1*x2,
%   u = sqrt(a)+0.5*sin(x1),   v = exp(b)+0.5*cos(x2),
%   p = log(u)+0.5*b,          q = sin(v)+0.5*a.
% Then
%   y = p+0.5*q+0.5*cos(b)+0.5*sin(x2).
%
% The case retains Coef_M{2,2}, Coef_M{2,3}, Coef_M{3,4}, and
% Coef_M{4,4}, but removes the extra hidden layer and highly redundant
% residual readout used by the v73m structural stress test.

    task = struct();
    task.name = 'SymbolicRepresentability_ModerateSkip2D';
    task.caseName = 'symbolic_representability_moderate_skip_2d';
    task.casemode = 'strong_prior';
    task.sourceName = 'manuscript_constructive_symbolic_representability';
    task.description = ['Two-input scalar DAG with selected middle branches ', ...
        'and reduced parameter redundancy.'];

    task.nx = 2;
    task.ny = 1;
    task.variableNames = {'x1','x2'};
    task.outputNames = {'y'};
    task.symbolicExpressions = { ...
        ['p+0.5*q+0.5*cos(b)+0.5*sin(x2),where ', ...
         'a=1+x1^2+x2^2,b=0.5*x1*x2,', ...
         'u=sqrt(a)+0.5*sin(x1),v=exp(b)+0.5*cos(x2),', ...
         'p=log(u)+0.5*b,q=sin(v)+0.5*a'] ...
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
    task.sampleValidityFcn = @sample_validity_moderate_skip_2d_local;

    task.domain = make_variable_union_domain(task.variableNames, { ...
        [-1.00,1.00], ...
        [-1.00,1.00] ...
    });
    task.domain.source = 'constructive_moderate_skip_2d_id';

    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [-1.60,-1.20;1.20,1.60], ...
        [-1.60,-1.20;1.20,1.60] ...
    });
    task.oodDomain.source = 'constructive_moderate_skip_2d_disjoint_ood';

    task.rhsFcn = @rhs_moderate_skip_2d_local;

    task.arch = struct();
    task.arch.layer = 4;
    task.arch.hiddenDims = [2,2,2];
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

function Y = rhs_moderate_skip_2d_local(X)
    x1 = X(:,1);
    x2 = X(:,2);

    a = 1 + x1.^2 + x2.^2;
    b = 0.5.*x1.*x2;
    u = sqrt(a) + 0.5.*sin(x1);
    v = exp(b) + 0.5.*cos(x2);
    p = log(u) + 0.5.*b;
    q = sin(v) + 0.5.*a;
    Y = p + 0.5.*q + 0.5.*cos(b) + 0.5.*sin(x2);
end

function [valid, details] = sample_validity_moderate_skip_2d_local(X, options)
    x1 = X(:,1);
    x2 = X(:,2);

    logMargin = get_sampling_option_local(options,'logArgumentMargin',1e-4);
    sqrtMargin = get_sampling_option_local(options,'sqrtArgumentMargin',1e-8);
    maxExpArgument = get_sampling_option_local(options,'maximumExpArgument',50);

    a = 1 + x1.^2 + x2.^2;
    b = 0.5.*x1.*x2;
    sqrtSafe = isfinite(a) & a >= sqrtMargin;

    u = nan(size(a));
    u(sqrtSafe) = sqrt(a(sqrtSafe)) + 0.5.*sin(x1(sqrtSafe));
    logSafe = isfinite(u) & u >= logMargin;
    expSafe = isfinite(b) & abs(b) <= maxExpArgument;

    valid = all(isfinite(X),2) & sqrtSafe & logSafe & expSafe;

    details = struct();
    details.radicandA = a;
    details.expArgumentB = b;
    details.logArgumentU = u;
end

function value = get_sampling_option_local(options, fieldName, defaultValue)
    if isstruct(options) && isfield(options,fieldName) && ~isempty(options.(fieldName))
        value = options.(fieldName);
    else
        value = defaultValue;
    end
end
