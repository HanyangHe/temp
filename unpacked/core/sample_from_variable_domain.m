function X = sample_from_variable_domain(nSamples, domain)
%SAMPLE_FROM_VARIABLE_DOMAIN Sample uniformly from a per-variable interval union.
%
% For each variable with multiple intervals, an interval is selected with
% probability proportional to interval length, and the sample is then drawn
% uniformly inside that interval. For a single-interval variable, no interval
% selection random numbers are consumed. Therefore, when all variables have one
% interval, the sampler is random-stream compatible with the legacy box sampler:
%
%   X = lb + rand(nSamples, nx) .* (ub - lb);

	domain = normalize_task_domain(domain);
	nx = numel(domain.intervals);
	nInterval = cellfun(@(I) size(I, 1), domain.intervals);

	% Exact legacy compatibility path. This preserves not only the same
	% distribution, but also the same sample matrix under the same RNG seed.
	if all(nInterval == 1)
		lb = reshape(domain.lb, 1, []);
		ub = reshape(domain.ub, 1, []);
		X = lb + rand(nSamples, nx) .* (ub - lb);
		return;
	end

	X = zeros(nSamples, nx);

	for j = 1:nx
		I = domain.intervals{j};

		if size(I, 1) == 1
			lo = I(1, 1);
			hi = I(1, 2);
			X(:, j) = lo + rand(nSamples, 1) .* (hi - lo);
			continue;
		end

		width = I(:, 2) - I(:, 1);
		if all(width <= 0)
			weights = ones(size(width)) ./ numel(width);
		else
			weights = max(width, 0);
			weights = weights ./ sum(weights);
		end

		cumw = cumsum(weights(:)).';
		cumw(end) = 1;
		r = rand(nSamples, 1);
		idx = sum(bsxfun(@gt, r, cumw), 2) + 1;

		lo = I(idx, 1);
		hi = I(idx, 2);
		X(:, j) = lo + rand(nSamples, 1) .* (hi - lo);
	end
end
