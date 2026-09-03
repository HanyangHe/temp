function task = task_symbolic_representability_3d()
%TASK_SYMBOLIC_REPRESENTABILITY_3D Three-input/three-output constructive PhDN case.
%
% The case is designed for the Numerical Experiments subsection
% "Symbolic Representability". It contains the elementary operator family
% used by the current PhDN implementation: constant/identity, polynomial,
% multiplication, reciprocal, square root, exponential, logarithm, sine,
% and cosine.
%
% Target map:
%   y1 = 3*cos(x2) + 0.5*x1^2 + sqrt(x3)
%   y2 = log(0.25*exp(x1*x2) + 1/(x1^2+x2+1)) + sin(x2)
%   y3 = log(0.25*exp(x1*x2) + 1/(x1^2+x3+2*x2*x3)) ...
%        + 3*x1/(8*x3)
%
% The ID/OOD domains are chosen so x3>0 and both reciprocal/logarithm
% arguments remain strictly positive. OOD samples are disjoint from the ID
% box and extrapolate in all three input coordinates.

	task = struct();
	task.name = 'SymbolicRepresentability_3D';
	task.caseName = 'symbolic_representability_3d';
	task.casemode = 'strong_prior';
	task.sourceName = 'manuscript_constructive_symbolic_representability';
	task.description = ['Constructive three-output PhDN DAG containing polynomial, ', ...
		'product, inv, sqrt, exp, log, sin, and cos operators.'];

	task.nx = 3;
	task.ny = 3;
	task.variableNames = {'x1','x2','x3'};
	task.outputNames = {'y1','y2','y3'};
	task.symbolicExpressions = { ...
		'3*cos(x2)+0.5*x1^2+sqrt(x3)', ...
		'log(0.25*exp(x1*x2)+1/(x1^2+x2+1))+sin(x2)', ...
		'log(0.25*exp(x1*x2)+1/(x1^2+x3+2*x2*x3))+3*x1/(8*x3)' ...
	};

	% Four independently generated global space-filling datasets.
	task.dataDefaults = struct('nTrain', 1500, 'nValidation', 500, ...
		'nIDTest', 500, 'nOODSamples', 500);

	% Sampling remains global over each complete 3-D domain. No hand-selected
	% subregion mixture or expression-dependent weighting is used. Candidate
	% points are rejected only when they violate a real-valued operator domain
	% or approach a declared reciprocal singularity.
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
	task.sampleValidityFcn = @sample_validity_symbolic_representability_3d_local;

	% Safe interpolation domain.
	task.domain = make_variable_union_domain(task.variableNames, { ...
		[-1.00, 1.00], ...
		[-0.35, 0.75], ...
		[ 0.60, 1.60] ...
	});
	task.domain.source = 'constructive_symbolic_representability_id';

	% Disjoint extrapolation domain. The sampler still evaluates the same
	% validity conditions pointwise instead of relying only on box metadata.
	task.oodDomain = make_variable_union_domain(task.variableNames, { ...
		[-1.50, -1.10; 1.10, 1.50], ...
		[-0.55, -0.45; 0.85, 1.05], ...
		[ 1.80,  2.40] ...
	});
	task.oodDomain.source = 'constructive_symbolic_representability_disjoint_ood';
	task.oodDomain.minimumX3 = 1.80;
	task.oodDomain.minimumDenominator2 = 1.66;
	task.oodDomain.minimumDenominator3 = 0.97;

	task.rhsFcn = @rhs_symbolic_representability_3d_local;

	% The constructive compiler replaces this base architecture with the same
	% explicit dimensions plus its branch dictionaries and exact coefficients.
	task.arch = struct();
	task.arch.layer = 5;
	task.arch.hiddenDims = [3, 3, 2, 2];
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

function Y = rhs_symbolic_representability_3d_local(X)
	x1 = X(:,1);
	x2 = X(:,2);
	x3 = X(:,3);

	den2 = x1.^2 + x2 + 1;
	den3 = x1.^2 + x3 + 2.*x2.*x3;
	sharedExp = 0.25 .* exp(x1.*x2);

	y1 = 3.*cos(x2) + 0.5.*x1.^2 + sqrt(x3);
	y2 = log(sharedExp + 1./den2) + sin(x2);
	y3 = log(sharedExp + 1./den3) + 3.*x1./(8.*x3);
	Y = [y1, y2, y3];
end

function [valid, details] = sample_validity_symbolic_representability_3d_local(X, options)
% Reject only non-real or near-singular points; do not weight subregions.
	x1 = X(:,1);
	x2 = X(:,2);
	x3 = X(:,3);

	inverseMargin = get_sampling_option_local(options, 'inverseDenominatorMargin', 5e-2);
	logMargin = get_sampling_option_local(options, 'logArgumentMargin', 1e-4);
	sqrtMargin = get_sampling_option_local(options, 'sqrtArgumentMargin', 1e-8);
	maxExpArgument = get_sampling_option_local(options, 'maximumExpArgument', 50);

	den2 = x1.^2 + x2 + 1;
	den3 = x1.^2 + x3 + 2.*x2.*x3;
	expArgument = x1.*x2;

	finiteInputs = all(isfinite(X),2);
	expSafe = isfinite(expArgument) & abs(expArgument) <= maxExpArgument;
	x3Safe = x3 >= sqrtMargin & abs(x3) >= inverseMargin;
	den2Safe = den2 >= inverseMargin;
	den3Safe = den3 >= inverseMargin;

	sharedExp = nan(size(expArgument));
	sharedExp(expSafe) = 0.25 .* exp(expArgument(expSafe));
	logArg2 = nan(size(den2));
	logArg3 = nan(size(den3));
	validForLog2 = expSafe & den2Safe;
	validForLog3 = expSafe & den3Safe;
	logArg2(validForLog2) = sharedExp(validForLog2) + 1./den2(validForLog2);
	logArg3(validForLog3) = sharedExp(validForLog3) + 1./den3(validForLog3);
	logSafe = isfinite(logArg2) & isfinite(logArg3) & ...
		logArg2 >= logMargin & logArg3 >= logMargin;

	valid = finiteInputs & expSafe & x3Safe & den2Safe & den3Safe & logSafe;

	details = struct();
	details.denominator2 = den2;
	details.denominator3 = den3;
	details.logArgument2 = logArg2;
	details.logArgument3 = logArg3;
	details.expArgument = expArgument;
end

function value = get_sampling_option_local(options, fieldName, defaultValue)
	if isstruct(options) && isfield(options, fieldName) && ~isempty(options.(fieldName))
		value = options.(fieldName);
	else
		value = defaultValue;
	end
end

