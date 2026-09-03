function task = task_symbolic_representability_dense_skip_3d()
%TASK_SYMBOLIC_REPRESENTABILITY_DENSE_SKIP_3D Moderate 3-D two-output DAG.
%
% Let
%   a = 1+x1^2+x2^2,            b = 0.5*x1*x2,
%   c = 1+x3^2,
%   u = sqrt(a)+0.5*sin(x1),     v = exp(b)+0.5*cos(x2),
%   w = 1/c+0.5*x1*x3,
%   p = log(u)+0.5*b+0.5*cos(x3),
%   q = sin(v)+0.5*w+0.5*x2^2.
% Then
%   y1 = p+0.5*q+0.5*cos(b)+0.5*sin(x3),
%   y2 = q+0.5*p+0.5*u+0.5*cos(x1*x2).
%
% The case retains representative middle branches and a multi-output
% fan-in/fan-out structure while reducing depth and active coefficients.

    task = struct();
    task.name = 'SymbolicRepresentability_ModerateSkip3D';
    task.caseName = 'symbolic_representability_moderate_skip_3d';
    task.casemode = 'strong_prior';
    task.sourceName = 'manuscript_constructive_symbolic_representability';
    task.description = ['Three-input two-output moderate fan-in DAG with ', ...
        'representative middle branches and reduced compensation freedom.'];

    task.nx = 3;
    task.ny = 2;
    task.variableNames = {'x1','x2','x3'};
    task.outputNames = {'y1','y2'};
    commonDefinition = [ ...
        'where a=1+x1^2+x2^2,b=0.5*x1*x2,c=1+x3^2,', ...
        'u=sqrt(a)+0.5*sin(x1),v=exp(b)+0.5*cos(x2),', ...
        'w=1/c+0.5*x1*x3,p=log(u)+0.5*b+0.5*cos(x3),', ...
        'q=sin(v)+0.5*w+0.5*x2^2'];
    task.symbolicExpressions = { ...
        ['p+0.5*q+0.5*cos(b)+0.5*sin(x3),' commonDefinition], ...
        ['q+0.5*p+0.5*u+0.5*cos(x1*x2),' commonDefinition] ...
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
    task.sampleValidityFcn = @sample_validity_moderate_skip_3d_local;

    task.domain = make_variable_union_domain(task.variableNames, { ...
        [-1.00,1.00], ...
        [-1.00,1.00], ...
        [0.20,1.20] ...
    });
    task.domain.source = 'constructive_moderate_skip_3d_id';

    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [-1.50,-1.10;1.10,1.50], ...
        [-1.50,-1.10;1.10,1.50], ...
        [1.40,2.00] ...
    });
    task.oodDomain.source = 'constructive_moderate_skip_3d_disjoint_ood';

    task.rhsFcn = @rhs_moderate_skip_3d_local;

    task.arch = struct();
    task.arch.layer = 4;
    task.arch.hiddenDims = [3,3,2];
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

function Y = rhs_moderate_skip_3d_local(X)
    x1 = X(:,1);
    x2 = X(:,2);
    x3 = X(:,3);

    a = 1 + x1.^2 + x2.^2;
    b = 0.5.*x1.*x2;
    c = 1 + x3.^2;
    u = sqrt(a) + 0.5.*sin(x1);
    v = exp(b) + 0.5.*cos(x2);
    w = 1./c + 0.5.*x1.*x3;
    p = log(u) + 0.5.*b + 0.5.*cos(x3);
    q = sin(v) + 0.5.*w + 0.5.*x2.^2;

    y1 = p + 0.5.*q + 0.5.*cos(b) + 0.5.*sin(x3);
    y2 = q + 0.5.*p + 0.5.*u + 0.5.*cos(x1.*x2);
    Y = [y1,y2];
end

function [valid, details] = sample_validity_moderate_skip_3d_local(X, options)
    x1 = X(:,1);
    x2 = X(:,2);
    x3 = X(:,3);

    inverseMargin = get_sampling_option_local(options, ...
        'inverseDenominatorMargin',5e-2);
    logMargin = get_sampling_option_local(options,'logArgumentMargin',1e-4);
    sqrtMargin = get_sampling_option_local(options,'sqrtArgumentMargin',1e-8);
    maxExpArgument = get_sampling_option_local(options,'maximumExpArgument',50);

    a = 1 + x1.^2 + x2.^2;
    b = 0.5.*x1.*x2;
    c = 1 + x3.^2;
    sqrtSafe = isfinite(a) & a >= sqrtMargin;
    inverseSafe = isfinite(c) & c >= inverseMargin;

    u = nan(size(a));
    u(sqrtSafe) = sqrt(a(sqrtSafe)) + 0.5.*sin(x1(sqrtSafe));
    logSafe = isfinite(u) & u >= logMargin;
    expSafe = isfinite(b) & abs(b) <= maxExpArgument;

    valid = all(isfinite(X),2) & sqrtSafe & inverseSafe & logSafe & expSafe;

    details = struct();
    details.radicandA = a;
    details.denominatorC = c;
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
