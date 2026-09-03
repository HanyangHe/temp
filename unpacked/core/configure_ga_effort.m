function opts = configure_ga_effort(opts, nTheta, effort)
%CONFIGURE_GA_EFFORT Configure GA budget from active parameter count.
%
% This function only configures GA-related budgets.  It does not change the
% random-LSQ seed candidate count or the LSQ multi-start count.
%
% The GA budget is defined by the evaluation-times-per-dimension metric:
%
%   ETPD = populationSize * maxGenerations / nTheta.
%
% Effort levels:
%   quick   : targetETPD = 50
%   default : targetETPD = 100
%   robust  : targetETPD = 200
%
% By default, legacy auto caps are NOT applied because they can make
% quick/default/robust collapse to the same pop/gen budget. To enforce caps,
% set opts.init.ga.respectAutoCaps = true.

	if nargin < 3 || isempty(effort)
		effort = 'default';
	end
	if nargin < 2 || isempty(nTheta) || ~isfinite(nTheta)
		nTheta = 1;
	end

	nTheta = max(1, round(nTheta));
	effort = lower(strtrim(char(effort)));

	if ~isfield(opts, 'init') || ~isstruct(opts.init)
		opts.init = struct();
	end
	if ~isfield(opts.init, 'ga') || ~isstruct(opts.init.ga)
		opts.init.ga = struct();
	end

	% ---------------------------------------------------------------------
	% Effort-level ETPD targets.
	% ---------------------------------------------------------------------
	switch effort
		case 'quick'
			targetETPD = 50;
			stallFrac = 0.40;
			minPop = 40;
			minGen = 10;

		case 'default'
			targetETPD = 100;
			stallFrac = 0.35;
			minPop = 60;
			minGen = 15;

		case 'robust'
			targetETPD = 200;
			stallFrac = 0.30;
			minPop = 80;
			minGen = 20;

		otherwise
			error('Unknown GA effort level: %s', effort);
	end

	% Optional direct override for target ETPD.
	targetETPD = getfield_default_local(opts.init.ga, 'targetETPD', targetETPD);
	targetETPD = max(1, targetETPD);

	% Generation/population ratio.  Default: gen ~= pop/3.
	genPopRatio = getfield_default_local(opts.init.ga, 'generationPopulationRatio', 1/3);
	if ~isfinite(genPopRatio) || genPopRatio <= 0
		genPopRatio = 1/3;
	end

	% ETPD budget:
	%   evalBudget = targetETPD * nTheta.
	% With gen ~= ratio * pop:
	%   ratio * pop^2 ~= evalBudget.
	targetEvalBudget = ceil(targetETPD * nTheta);

	pop = round(sqrt(targetEvalBudget / genPopRatio));
	pop = max(minPop, pop);

	gen = round(genPopRatio * pop);
	gen = max(minGen, gen);

	% ---------------------------------------------------------------------
	% Optional legacy caps.
	% By default these caps are disabled, because they previously forced
	% quick/default/robust to the same budget, e.g. 120*40.
	% ---------------------------------------------------------------------
	respectAutoCaps = getfield_default_local(opts.init.ga, 'respectAutoCaps', false);
	capLimited = false;

	if respectAutoCaps
		maxPop = getfield_default_local(opts.init.ga, 'maxPopulationSizeAuto', Inf);
		maxGen = getfield_default_local(opts.init.ga, 'maxGenerationsAuto', Inf);
		maxBudget = getfield_default_local(opts.init.ga, 'maxEvalBudgetAuto', Inf);

		if isfinite(maxPop) && maxPop > 0 && pop > maxPop
			pop = maxPop;
			capLimited = true;
		end

		if isfinite(maxGen) && maxGen > 0 && gen > maxGen
			gen = maxGen;
			capLimited = true;
		end

		if isfinite(maxBudget) && maxBudget > 0 && pop * gen > maxBudget
			gen = max(1, floor(maxBudget / max(1, pop)));
			capLimited = true;
		end
	end

	pop = max(1, round(pop));
	gen = max(1, round(gen));

	opts.init.ga.effort = effort;
	opts.init.ga.populationSize = pop;
	opts.init.ga.maxGenerations = gen;
	opts.init.ga.maxStallGenerations = max(5, ceil(stallFrac * gen));

	opts.init.ga.autoConfiguredNTheta = nTheta;
	opts.init.ga.targetETPD = targetETPD;
	opts.init.ga.generationPopulationRatio = genPopRatio;
	opts.init.ga.autoConfiguredEvalBudget = pop * gen;
	opts.init.ga.effectiveETPD = (pop * gen) / nTheta;
	opts.init.ga.capLimited = capLimited;

	if capLimited
		warning(['GA auto budget was limited by maxPopulationSizeAuto/maxGenerationsAuto/maxEvalBudgetAuto. ', ...
			'Effective ETPD = %.2f, target ETPD = %.2f.'], ...
			opts.init.ga.effectiveETPD, opts.init.ga.targetETPD);
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
