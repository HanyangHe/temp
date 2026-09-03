function oodDomain = make_ood_domain(task, opts)
%MAKE_OOD_DOMAIN Build an out-of-distribution interval-union domain.
%
% Default behavior:
%   OOD intervals are generated from the outer envelope of task.domain.
%   Unlike the old box-only version, the returned OOD domain also uses the
%   per-variable interval-union format.
%
% Options:
%   opts.ood.autoMode       = 'upper' | 'lower' | 'both'
%   opts.ood.autoGapRatio   = scalar, default 0.05
%   opts.ood.autoWidthRatio = scalar, default 0.25
%
% Manual override:
%   Set opts.ood.useTaskOodDomain = true to use task.oodDomain if present.

	if nargin < 2 || isempty(opts)
		opts = struct();
	end
	if ~isfield(opts, 'ood') || isempty(opts.ood)
		opts.ood = struct();
	end

	if isfield(task, 'variableNames')
		varNames = task.variableNames;
	else
		varNames = arrayfun(@(k) sprintf('x%d', k), 1:task.nx, 'UniformOutput', false);
	end

	useTaskOodDomain = false;
	if isfield(opts.ood, 'useTaskOodDomain') && ~isempty(opts.ood.useTaskOodDomain)
		useTaskOodDomain = logical(opts.ood.useTaskOodDomain);
	end

	if useTaskOodDomain && isfield(task, 'oodDomain') && ~isempty(task.oodDomain)
		oodDomain = normalize_task_domain(task.oodDomain, task.nx, varNames);
		return;
	end

	if ~isfield(task, 'domain') || isempty(task.domain)
		error('task.domain is required to generate OOD data.');
	end

	domain = normalize_task_domain(task.domain, task.nx, varNames);
	autoMode = get_ood_option(opts.ood, 'autoMode', 'upper');
	if isstring(autoMode)
		autoMode = char(autoMode);
	end
	gapRatio = get_ood_option(opts.ood, 'autoGapRatio', 0.05);
	widthRatio = get_ood_option(opts.ood, 'autoWidthRatio', 0.25);

	lb = reshape(domain.lb, 1, []);
	ub = reshape(domain.ub, 1, []);
	width = ub - lb;

	badWidth = (~isfinite(width)) | (width <= 0);
	width(badWidth) = 1;

	gap = gapRatio .* width;
	oodWidth = widthRatio .* width;

	intervals = cell(1, task.nx);
	switch lower(strtrim(autoMode))
		case 'upper'
			for k = 1:task.nx
				intervals{k} = [ub(k) + gap(k), ub(k) + gap(k) + oodWidth(k)];
			end

		case 'lower'
			for k = 1:task.nx
				intervals{k} = [lb(k) - gap(k) - oodWidth(k), lb(k) - gap(k)];
			end

		case 'both'
			for k = 1:task.nx
				intervals{k} = [ ...
					lb(k) - gap(k) - oodWidth(k), lb(k) - gap(k); ...
					ub(k) + gap(k), ub(k) + gap(k) + oodWidth(k)];
			end

		otherwise
			error('Unknown opts.ood.autoMode: %s. Use upper, lower, or both.', autoMode);
	end

	oodDomain = make_variable_union_domain(varNames, intervals);
	oodDomain.source = 'auto_envelope_shift';
	oodDomain.autoMode = autoMode;
	oodDomain.autoGapRatio = gapRatio;
	oodDomain.autoWidthRatio = widthRatio;
end

function val = get_ood_option(ood, name, defaultVal)
	if isfield(ood, name) && ~isempty(ood.(name))
		val = ood.(name);
	else
		val = defaultVal;
	end
end
