function domain = normalize_task_domain(domainIn, nx, varNames)
%NORMALIZE_TASK_DOMAIN Normalize legacy lb/ub domains and interval-union domains.
%
% New preferred domain format:
%   domain.intervals{i} = [lb1, ub1; lb2, ub2; ...]
%
% Legacy compatibility:
%   domain.lb and domain.ub are converted into one interval per variable.

	if nargin < 2 || isempty(nx)
		nx = [];
	end
	if nargin < 3 || isempty(varNames)
		varNames = {};
	end
	if isstring(varNames)
		varNames = cellstr(varNames);
	end
	if ischar(varNames)
		varNames = {varNames};
	end

	if iscell(domainIn) || isnumeric(domainIn)
		domain = make_variable_union_domain(varNames, domainIn);
		return;
	end

	if ~isstruct(domainIn)
		error('Domain must be a struct, a cell interval list, or an nx-by-2 numeric interval array.');
	end

	if isempty(varNames) && isfield(domainIn, 'variableNames') && ~isempty(domainIn.variableNames)
		varNames = domainIn.variableNames;
		if isstring(varNames)
			varNames = cellstr(varNames);
		end
	end

	if isempty(nx)
		if ~isempty(varNames)
			nx = numel(varNames);
		elseif isfield(domainIn, 'intervals') && ~isempty(domainIn.intervals)
			nx = numel(domainIn.intervals);
		elseif isfield(domainIn, 'variableIntervals') && ~isempty(domainIn.variableIntervals)
			nx = numel(domainIn.variableIntervals);
		elseif isfield(domainIn, 'lb') && ~isempty(domainIn.lb)
			nx = numel(domainIn.lb);
		else
			error('Cannot infer the number of variables in the domain.');
		end
	end

	if isempty(varNames)
		varNames = arrayfun(@(k) sprintf('x%d', k), 1:nx, 'UniformOutput', false);
	end

	if isfield(domainIn, 'intervals') && ~isempty(domainIn.intervals)
		intervalSpec = domainIn.intervals;
	elseif isfield(domainIn, 'variableIntervals') && ~isempty(domainIn.variableIntervals)
		intervalSpec = domainIn.variableIntervals;
	elseif isfield(domainIn, 'lb') && isfield(domainIn, 'ub') && ~isempty(domainIn.lb) && ~isempty(domainIn.ub)
		lb = reshape(domainIn.lb, 1, []);
		ub = reshape(domainIn.ub, 1, []);
		if numel(lb) ~= nx || numel(ub) ~= nx
			error('domain.lb/domain.ub dimensions do not match the number of variables.');
		end
		intervalSpec = cell(1, nx);
		for k = 1:nx
			intervalSpec{k} = [lb(k), ub(k)];
		end
	else
		error('Domain must provide either intervals/variableIntervals or legacy lb/ub fields.');
	end

	domain = make_variable_union_domain(varNames, intervalSpec);

	% Preserve optional metadata that may be useful outside the sampler.
	fn = fieldnames(domainIn);
	for k = 1:numel(fn)
		name = fn{k};
		if ~isfield(domain, name)
			domain.(name) = domainIn.(name);
		end
	end
end
