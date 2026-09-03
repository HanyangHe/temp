function [X, Y] = sample_task_data(task, nSamples, domain)
%SAMPLE_TASK_DATA Generate X and target Y from a task module.
%
% Usage:
%   [X, Y] = sample_task_data(task, nSamples)
%   [X, Y] = sample_task_data(task, nSamples, domain)
%
% The preferred domain format is a per-variable union of intervals:
%   domain.intervals{i} = [lb1, ub1; lb2, ub2; ...]
%
% Legacy domain.lb/domain.ub fields are still accepted and are converted into
% one interval per variable.

	if nargin < 3 || isempty(domain)
		domain = task.domain;
	end

	if isfield(task, 'variableNames')
		varNames = task.variableNames;
	else
		varNames = arrayfun(@(k) sprintf('x%d', k), 1:task.nx, 'UniformOutput', false);
	end
	domain = normalize_task_domain(domain, task.nx, varNames);

	if isfield(task, 'sampleFcn') && ~isempty(task.sampleFcn)
		% Preferred custom sampler signature: sampleFcn(nSamples, domain).
		% Legacy signature sampleFcn(nSamples, lb, ub) is kept as a fallback.
		try
			X = task.sampleFcn(nSamples, domain);
		catch ME_domain
			try
				X = task.sampleFcn(nSamples, domain.lb, domain.ub);
			catch
				rethrow(ME_domain);
			end
		end
	else
		X = sample_from_variable_domain(nSamples, domain);
	end

	Y = task.rhsFcn(X);
end
