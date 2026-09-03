function domain = make_variable_union_domain(varNames, intervalSpec)
%MAKE_VARIABLE_UNION_DOMAIN Build a per-variable union-of-intervals domain.
%
% Each input variable owns its own interval list. The full sampling domain is
% the Cartesian product of these one-dimensional interval unions.
%
% Example:
%   domain = make_variable_union_domain({'x1','x2'}, {[-2, 2], [-2, -0.2; 0.2, 2]});
%
% The legacy fields domain.lb and domain.ub are still stored as the outer
% envelope only, so old diagnostic or plotting code can keep working. New
% sampling code should use domain.intervals instead of domain.lb/domain.ub.

	if nargin < 1 || isempty(varNames)
		varNames = {};
	end
	if nargin < 2
		intervalSpec = {};
	end

	if isstring(varNames)
		varNames = cellstr(varNames);
	end
	if ischar(varNames)
		varNames = {varNames};
	end

	if isstruct(intervalSpec)
		domain = normalize_task_domain(intervalSpec, numel(varNames), varNames);
		return;
	end

	nx = numel(varNames);
	if nx == 0
		if iscell(intervalSpec)
			nx = numel(intervalSpec);
		elseif isnumeric(intervalSpec) && size(intervalSpec, 2) == 2
			nx = size(intervalSpec, 1);
		else
			error('Cannot infer the number of variables from intervalSpec.');
		end
		varNames = arrayfun(@(k) sprintf('x%d', k), 1:nx, 'UniformOutput', false);
	end

	intervals = normalize_interval_spec_local(intervalSpec, nx);

	domain = struct();
	domain.type = 'per_variable_interval_union';
	domain.variableNames = reshape(varNames, 1, []);
	domain.intervals = intervals;
	domain.variableIntervals = intervals;
	domain.lb = cellfun(@(I) min(I(:, 1)), intervals);
	domain.ub = cellfun(@(I) max(I(:, 2)), intervals);
	domain.nIntervals = cellfun(@(I) size(I, 1), intervals);
end

function intervals = normalize_interval_spec_local(intervalSpec, nx)
	if iscell(intervalSpec)
		if numel(intervalSpec) ~= nx
			error('The interval cell array must contain one entry per variable.');
		end
		intervals = cell(1, nx);
		for k = 1:nx
			intervals{k} = normalize_one_variable_intervals_local(intervalSpec{k}, k);
		end
		return;
	end

	if isnumeric(intervalSpec)
		if size(intervalSpec, 2) ~= 2 || size(intervalSpec, 1) ~= nx
			error('A numeric intervalSpec must be an nx-by-2 array [lb, ub].');
		end
		intervals = cell(1, nx);
		for k = 1:nx
			intervals{k} = normalize_one_variable_intervals_local(intervalSpec(k, :), k);
		end
		return;
	end

	error('intervalSpec must be a cell array or an nx-by-2 numeric array.');
end

function I = normalize_one_variable_intervals_local(I, varIndex)
	if isempty(I)
		error('The interval list for variable %d is empty.', varIndex);
	end
	if ~isnumeric(I)
		error('The interval list for variable %d must be numeric.', varIndex);
	end
	if isvector(I)
		if numel(I) ~= 2
			error('A vector interval for variable %d must have exactly two entries [lb, ub].', varIndex);
		end
		I = reshape(I, 1, 2);
	end
	if size(I, 2) ~= 2
		error('Intervals for variable %d must be stored as an nInterval-by-2 array.', varIndex);
	end
	if any(~isfinite(I(:)))
		error('Intervals for variable %d contain nonfinite values.', varIndex);
	end

	% If a lower bound is accidentally larger than the upper bound, clamp the
	% lower bound to the upper bound without changing the upper bound.
	I(:, 1) = min(I(:, 1), I(:, 2));

	I = sortrows(I, 1);
	I = merge_overlapping_intervals_local(I);
end

function Iout = merge_overlapping_intervals_local(I)
	Iout = I(1, :);
	for k = 2:size(I, 1)
		if I(k, 1) <= Iout(end, 2)
			Iout(end, 2) = max(Iout(end, 2), I(k, 2));
		else
			Iout(end + 1, :) = I(k, :); %#ok<AGROW>
		end
	end
end
