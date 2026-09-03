function theta0Mat = make_ga_seed_points(nVars, lb, ub, opts)
%MAKE_GA_SEED_POINTS Generate generic uniform-box seed candidates.
%
% The public control is opts.numCandidates.  These are seed candidates, not
% necessarily LSQ starts.  The actual number of LSQ starts is controlled by
% opts.init.lsq.numStarts in the main initializer.

	if nargin < 4
		opts = struct();
	end

	seed = getfield_default_local(opts, 'rngSeed', getfield_default_local(opts, 'seed', 1));
	rng(seed);

	nSeeds = getfield_default_local(opts, 'numCandidates', 40);
	nSeeds = max(0, round(nSeeds));

	lb = lb(:).';
	ub = ub(:).';

	if nVars == 0 || nSeeds == 0
		theta0Mat = zeros(0, nVars);
		return;
	end

	theta0Mat = lb + rand(nSeeds, nVars) .* (ub - lb);
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
