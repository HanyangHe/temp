function [maskOut, report] = remove_redundant_identity_terms_from_mask(maskIn, arch, opts)
%REMOVE_REDUNDANT_IDENTITY_TERMS_FROM_MASK Remove algebraic 1-like cross terms.
%
% This filter removes dictionary columns of the form
%
%   u(h) * inv(u(h))
%
% such as x1*inv(x1), x2^2*inv(x2^2), h1*h2*inv(h1*h2).
% These terms are algebraically equal to 1 away from singularities and tend to
% act as redundant bias channels.  They can act as redundant bias channels and
% steal active capacity from physically meaningful ratio terms.
%
% Important: the filter does NOT remove non-canceling ratios such as
%
%   x1^2*inv(x2^2)
%
% because the numerator and denominator polynomial exponents are different.

	if nargin < 3 || isempty(opts)
		opts = struct();
	end

	enable = true;
	verbose = false;
	if isfield(opts, 'training') && isfield(opts.training, 'removeIdentityCancellationTerms') && ...
			~isempty(opts.training.removeIdentityCancellationTerms)
		enable = logical(opts.training.removeIdentityCancellationTerms);
	end
	if isfield(opts, 'training') && isfield(opts.training, 'identityCancellationVerbose') && ...
			~isempty(opts.training.identityCancellationVerbose)
		verbose = logical(opts.training.identityCancellationVerbose);
	end

	maskOut = maskIn;
	report = make_empty_report_local();
	report.enabled = enable;

	if ~enable || isempty(maskIn) || ~iscell(maskIn)
		return;
	end

	dims = get_arch_dims(arch);

	for ell = 1:size(maskIn, 2)
		for src = 1:ell
			if src > size(maskIn, 1) || isempty(maskIn{src, ell})
				continue;
			end

			M = logical(maskOut{src, ell});
			if isempty(M)
				continue;
			end

			k = ell - src + 1;
			if k <= numel(dims)
				inputDim = dims(k);
			else
				inputDim = size(M, 2);
			end

			terms = branch_dictionary_terms(inputDim, arch, 'h', ell);
			nCols = min(numel(terms), size(M, 2));
			if nCols == 0
				continue;
			end

			identityCols = false(1, size(M, 2));
			for col = 1:nCols
				identityCols(col) = is_identity_cancellation_term_local(terms(col));
			end

			if ~any(identityCols)
				continue;
			end

			removeMat = M & repmat(identityCols, size(M, 1), 1);
			nRemoved = nnz(removeMat);
			if nRemoved == 0
				continue;
			end

			M(removeMat) = false;
			maskOut{src, ell} = M;

			report.nRemovedCoefficients = report.nRemovedCoefficients + nRemoved;
			report.nAffectedBlocks = report.nAffectedBlocks + 1;
			report.blocks(end + 1, 1) = make_block_report_local(src, ell, identityCols, removeMat, terms); %#ok<AGROW>
		end
	end

	report.originalActive = count_active_mask(maskIn);
	report.filteredActive = count_active_mask(maskOut);

	if verbose && report.nRemovedCoefficients > 0
		fprintf('Identity-cancellation filter: removed %d redundant trainable coefficients (%d -> %d active).\n', ...
			report.nRemovedCoefficients, report.originalActive, report.filteredActive);
	elseif verbose
		fprintf('Identity-cancellation filter: no redundant trainable coefficients were active.\n');
	end
end

function tf = is_identity_cancellation_term_local(term)
	tf = false;
	if ~isstruct(term) || ~isfield(term, 'type') || ~strcmpi(term.type, 'cross')
		return;
	end
	if ~isfield(term, 'opName') || ~strcmpi(term.opName, 'inv')
		return;
	end
	if ~isfield(term, 'exponent') || ~isstruct(term.exponent) || ...
			~isfield(term.exponent, 'left') || ~isfield(term.exponent, 'opArg')
		return;
	end

	leftExp = term.exponent.left;
	opArgExp = term.exponent.opArg;
	if isempty(leftExp) || isempty(opArgExp) || ~isnumeric(leftExp) || ~isnumeric(opArgExp)
		return;
	end

	tf = isequal(size(leftExp), size(opArgExp)) && isequal(leftExp, opArgExp) && any(leftExp ~= 0);
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

function b = make_block_report_local(src, ell, identityCols, removeMat, terms)
	b = struct();
	b.src = src;
	b.ell = ell;
	b.block = sprintf('A{%d,%d}', src, ell);
	b.identityCols = find(identityCols(:).');
	b.nIdentityCols = numel(b.identityCols);
	b.nRemovedCoefficients = nnz(removeMat);
	if isempty(b.identityCols)
		b.identityTerms = {};
	else
		idx = b.identityCols(b.identityCols <= numel(terms));
		b.identityTerms = {terms(idx).name};
	end
end
