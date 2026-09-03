function task = task_symbolic_representability_shared_3d()
%TASK_SYMBOLIC_REPRESENTABILITY_SHARED_3D Deep shared-subexpression scalar case.
%
% Let
%   r = sqrt(x1^2+x2^2+0.5*x3^2),
%   q = x1*x2/(1+x3^2).
% The target is
%   y = log(1+exp(r)) + cos(q) + sin(r+q)/(1+x3).
%
% Both r and q are reused by multiple downstream branches, giving a genuine
% DAG rather than a simple expression chain.

    task = struct();
    task.name = 'SymbolicRepresentability_Shared3D';
    task.caseName = 'symbolic_representability_shared_3d';
    task.casemode = 'strong_prior';
    task.sourceName = 'manuscript_constructive_symbolic_representability';
    task.description = ['Three-input scalar DAG that reuses radial and rational ', ...
        'subexpressions in exp-log, cosine, and sine-reciprocal paths.'];

    task.nx = 3;
    task.ny = 1;
    task.variableNames = {'x1','x2','x3'};
    task.outputNames = {'y'};
    task.symbolicExpressions = { ...
        'log(1+exp(sqrt(x1^2+x2^2+0.5*x3^2)))+cos(x1*x2/(1+x3^2))+sin(sqrt(x1^2+x2^2+0.5*x3^2)+x1*x2/(1+x3^2))/(1+x3)' ...
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
    task.sampleValidityFcn = @sample_validity_shared_3d_local;

    task.domain = make_variable_union_domain(task.variableNames, { ...
        [-1.00,1.00], ...
        [-1.00,1.00], ...
        [0.20,1.20] ...
    });
    task.domain.source = 'constructive_shared_3d_id';

    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [-1.50,-1.10;1.10,1.50], ...
        [-1.50,-1.10;1.10,1.50], ...
        [1.40,2.00] ...
    });
    task.oodDomain.source = 'constructive_shared_3d_disjoint_ood';

    task.rhsFcn = @rhs_shared_3d_local;

    task.arch = struct();
    task.arch.layer = 6;
    task.arch.hiddenDims = [4,3,4,4,3];
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

function Y = rhs_shared_3d_local(X)
    x1 = X(:,1);
    x2 = X(:,2);
    x3 = X(:,3);

    r = sqrt(x1.^2 + x2.^2 + 0.5.*x3.^2);
    q = (x1.*x2)./(1+x3.^2);
    Y = log(1+exp(r)) + cos(q) + sin(r+q)./(1+x3);
end

function [valid, details] = sample_validity_shared_3d_local(X, options)
    x1 = X(:,1);
    x2 = X(:,2);
    x3 = X(:,3);

    inverseMargin = get_sampling_option_local(options, ...
        'inverseDenominatorMargin',5e-2);
    logMargin = get_sampling_option_local(options,'logArgumentMargin',1e-4);
    sqrtMargin = get_sampling_option_local(options,'sqrtArgumentMargin',1e-8);
    maxExpArgument = get_sampling_option_local(options,'maximumExpArgument',50);

    radicand = x1.^2 + x2.^2 + 0.5.*x3.^2;
    denominatorQ = 1 + x3.^2;
    denominatorOut = 1 + x3;

    finiteInputs = all(isfinite(X),2);
    sqrtSafe = isfinite(radicand) & radicand >= sqrtMargin;
    denominatorSafe = isfinite(denominatorQ) & isfinite(denominatorOut) & ...
        denominatorQ >= inverseMargin & denominatorOut >= inverseMargin;

    r = nan(size(radicand));
    r(sqrtSafe) = sqrt(radicand(sqrtSafe));
    expSafe = isfinite(r) & abs(r) <= maxExpArgument;

    logArgument = nan(size(r));
    logArgument(expSafe) = 1 + exp(r(expSafe));
    logSafe = isfinite(logArgument) & logArgument >= logMargin;

    valid = finiteInputs & sqrtSafe & denominatorSafe & expSafe & logSafe;

    details = struct();
    details.radicand = radicand;
    details.denominatorQ = denominatorQ;
    details.denominatorOutput = denominatorOut;
    details.expArgument = r;
    details.logArgument = logArgument;
end

function value = get_sampling_option_local(options, fieldName, defaultValue)
    if isstruct(options) && isfield(options,fieldName) && ~isempty(options.(fieldName))
        value = options.(fieldName);
    else
        value = defaultValue;
    end
end
