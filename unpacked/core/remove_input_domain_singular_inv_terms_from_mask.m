function [maskOut, report] = remove_input_domain_singular_inv_terms_from_mask(maskIn, arch, opts)
%REMOVE_INPUT_DOMAIN_SINGULAR_INV_TERMS_FROM_MASK Remove inv terms singular in ID input domain.
%
% This filter uses only the declared in-distribution input domain.  For raw
% input-source blocks, it removes inverse terms whose denominator monomial can
% be zero somewhere inside the domain, even if the finite training samples do
% not hit the singular point exactly.
%
% Example for I_6_2 with theta in [-10,10] and sigma in [-10,-0.5] U [0.5,10]:
%   remove: inv(theta), inv(theta^2), inv(theta*sigma)
%   keep  : inv(sigma), inv(sigma^2), theta^2*inv(sigma^2)

	if nargin < 3 || isempty(opts)
		opts = struct();
	end

	[enable, verbose, domain] = parse_options_local(opts);
	maskOut = maskIn;
	report = make_empty_report_local();
	report.enabled = enable;

	if ~enable || isempty(maskIn) || ~iscell(maskIn) || isempty(domain)
		return;
	end

	try
		dims = get_arch_dims(arch);
		domain = normalize_task_domain(domain);
	catch
		return;
	end

	zeroPossible = variable_zero_possible_local(domain);

	for ell = 1:size(maskIn, 2)
		for src = 1:ell
			if src > size(maskIn, 1) || isempty(maskOut{src, ell})
				continue;
			end

			% Only source state 1 corresponds to the raw input variables x.
			inputState = ell - src + 1;
			if inputState ~= 1
				continue;
			end

			M = logical(maskOut{src, ell});
			if isempty(M)
				continue;
			end

			if inputState <= numel(dims)
				inputDim = dims(inputState);
			else
				inputDim = numel(zeroPossible);
			end
			terms = branch_dictionary_terms(inputDim, arch, 'x', ell);
			nCols = min(numel(terms), size(M, 2));
			if nCols == 0
				continue;
			end

			removeCols = false(1, size(M, 2));
			for col = 1:nCols
				[isInv, expArg] = get_inv_argument_exponent_local(terms(col));
				if isInv && monomial_can_be_zero_local(expArg, zeroPossible)
					removeCols(col) = true;
				end
			end

			if ~any(removeCols)
				continue;
			end

			removeMat = M & repmat(removeCols, size(M, 1), 1);
			nRemoved = nnz(removeMat);
			if nRemoved == 0
				continue;
			end

			M(removeMat) = false;
			maskOut{src, ell} = M;

			report.nRemovedCoefficients = report.nRemovedCoefficients + nRemoved;
			report.nAffectedBlocks = report.nAffectedBlocks + 1;
			report.blocks(end + 1, 1) = make_block_report_local(src, ell, removeCols, removeMat, terms); %#ok<AGROW>
		end
	end

	report.originalActive = count_active_mask(maskIn);
	report.filteredActive = count_active_mask(maskOut);

	if verbose && report.nRemovedCoefficients > 0
		fprintf('Input-domain inverse-feasibility filter: removed %d singular trainable coefficients (%d -> %d active).\n', ...
			report.nRemovedCoefficients, report.originalActive, report.filteredActive);
	elseif verbose
		fprintf('Input-domain inverse-feasibility filter: no singular input-domain inverse terms were active.\n');
	end
end

function [enable, verbose, domain] = parse_options_local(opts)
	enable = true;
	verbose = false;
	domain = [];
	if isfield(opts, 'init') && isfield(opts.init, 'domainFilter')
		s = opts.init.domainFilter;
		if isfield(s, 'removeInputDomainSingularInvTerms') && ~isempty(s.removeInputDomainSingularInvTerms)
			enable = logical(s.removeInputDomainSingularInvTerms);
		end
		if isfield(s, 'inputDomainSingularInvVerbose') && ~isempty(s.inputDomainSingularInvVerbose)
			verbose = logical(s.inputDomainSingularInvVerbose);
		end
		if isfield(s, 'inputDomain') && ~isempty(s.inputDomain)
			domain = s.inputDomain;
		end
	end
	if isempty(domain) && isfield(opts, 'training') && isfield(opts.training, 'inputDomain') && ~isempty(opts.training.inputDomain)
		domain = opts.training.inputDomain;
	end
end

function z = variable_zero_possible_local(domain)
	nx = numel(domain.intervals);
	z = false(1, nx);
	for j = 1:nx
		I = domain.intervals{j};
		z(j) = any(I(:, 1) <= 0 & I(:, 2) >= 0);
	end
end

function tf = monomial_can_be_zero_local(expArg, zeroPossible)
	tf = false;
	if isempty(expArg) || ~isnumeric(expArg)
		return;
	end
	n = min(numel(expArg), numel(zeroPossible));
	if n == 0
		return;
	end
	tf = any(expArg(1:n) > 0 & zeroPossible(1:n));
end

function [isInv, expArg] = get_inv_argument_exponent_local(term)
	isInv = false;
	expArg = [];
	if ~isstruct(term) || ~isfield(term, 'opName') || ~strcmpi(term.opName, 'inv')
		return;
	end
	isInv = true;
	if strcmpi(term.type, 'cross') && isfield(term, 'exponent') && isstruct(term.exponent) && isfield(term.exponent, 'opArg')
		expArg = term.exponent.opArg;
	elseif isfield(term, 'exponent') && isnumeric(term.exponent)
		expArg = term.exponent;
	end
end

function report = make_empty_report_local()
	report = struct();
	report.enabled = true;
	report.originalActive = NaN;
	report.filteredActive = NaN;
	report.nRemovedCoefficients = 0;
	report.nAffectedBlocks = 0;
	report.blocks = repmat(make_block_report_local(NaN, NaN, false(1, 0), false(0, 0), struct([])), 0, 1);
end

function b = make_block_report_local(src, ell, removeCols, removeMat, terms)
	b = struct();
	b.src = src;
	b.ell = ell;
	b.block = sprintf('A{%d,%d}', src, ell);
	b.removedCols = find(removeCols(:).');
	b.nRemovedCols = numel(b.removedCols);
	b.nRemovedCoefficients = nnz(removeMat);
	if isempty(b.removedCols)
		b.removedTerms = {};
	else
		idx = b.removedCols(b.removedCols <= numel(terms));
		b.removedTerms = {terms(idx).name};
	end
end
