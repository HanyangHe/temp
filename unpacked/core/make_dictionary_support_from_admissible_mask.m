function supportA = make_dictionary_support_from_admissible_mask(admissibleA, arch, granularity)
%MAKE_DICTIONARY_SUPPORT_FROM_ADMISSIBLE_MASK Convert row-wise support to dictionary support.
%
% supportA = make_dictionary_support_from_admissible_mask(admissibleA, arch, granularity)
%
% granularity:
%   'blockwise' : each branch/block A{src,ell} keeps only the basis columns
%                 active in that same strong-prior block, broadcast to all rows.
%   'layerwise' : each target layer ell first collects the union of active
%                 term names from all incoming strong-prior blocks, then every
%                 compatible incoming branch in that layer uses that same
%                 layer-level local dictionary.
%
% This mask is for weak_strong_prior only. It is not the exact row-wise
% admissible mask used by strong_prior.

	if nargin < 3 || isempty(granularity)
		granularity = 'blockwise';
	end
	granularity = lower(strtrim(granularity));

	switch granularity
		case 'blockwise'
			supportA = make_blockwise_support_local(admissibleA);
		case 'layerwise'
			if nargin < 2 || isempty(arch)
				error('Layerwise dictionary support requires arch to map mask columns to dictionary term names.');
			end
			supportA = make_layerwise_support_local(admissibleA, arch);
		otherwise
			error('Unknown dictionary support granularity: %s. Use ''layerwise'' or ''blockwise''.', granularity);
	end
end

function supportA = make_blockwise_support_local(admissibleA)
	supportA = admissibleA;
	for ell = 1:size(admissibleA, 2)
		for src = 1:ell
			M = admissibleA{src, ell};
			if isempty(M)
				continue;
			end
			colActive = any(logical(M), 1);
			supportA{src, ell} = repmat(colActive, size(M, 1), 1);
		end
	end
end

function supportA = make_layerwise_support_local(admissibleA, arch)
	dims = get_arch_dims(arch);
	supportA = admissibleA;
	for ell = 1:size(admissibleA, 2)
		layerTerms = {};

		% 1) Collect the strong-prior active term names from all incoming blocks.
		for src = 1:ell
			M = admissibleA{src, ell};
			if isempty(M)
				continue;
			end
			k = ell - src + 1;
			inputDim = dims(k);
			termNames = explicit_case_dictionary_terms(inputDim, arch, ell, src);
			colActive = any(logical(M), 1);
			idx = find(colActive(:).');
			idx = idx(idx <= numel(termNames));
			for ii = 1:numel(idx)
				layerTerms{end+1,1} = char(termNames{idx(ii)}); %#ok<AGROW>
			end
		end
		layerTerms = unique_stable_local(layerTerms);

		% 2) Broadcast this layer-level local dictionary to every compatible block.
		for src = 1:ell
			M = admissibleA{src, ell};
			if isempty(M)
				continue;
			end
			k = ell - src + 1;
			inputDim = dims(k);
			termNames = explicit_case_dictionary_terms(inputDim, arch, ell, src);
			colActive = false(1, numel(termNames));
			for q = 1:numel(termNames)
				colActive(q) = any(strcmp(layerTerms, char(termNames{q})));
			end
			supportA{src, ell} = repmat(colActive, size(M, 1), 1);
		end
	end
end

function out = unique_stable_local(in)
	out = {};
	for i = 1:numel(in)
		v = char(in{i});
		if ~any(strcmp(out, v))
			out{end+1,1} = v; %#ok<AGROW>
		end
	end
end
