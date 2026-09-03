function [Coef_M, trainMSE, Ypred, readoutInfo] = solve_final_layer_readout(Coef_M, X, Y, arch, normOpt, lambdaOut, admissibleA)
%SOLVE_FINAL_LAYER_READOUT Solve the final-layer readout by ridge regression.
%
% Updated for layer-wise hidden dimensions, optional row-wise hard
% admissibility masks, and invalid-row admissibility filtering.
%
% The final layer ell = L has multiple source branches:
%   y = sum_{src=1}^{L} Xi^{(src,L)} Phi(h^{(L-src+1)}).
%
% With inner layers fixed, concatenate all final-layer branch dictionaries:
%   Zout = [Phi(h^L); Phi(h^{L-1}); ...; Phi(h^1)]
% and solve:
%   Wout = Y * Zout' / (Zout*Zout' + lambda I).
%
% Invalid-row filtering:
%   If a dictionary row generates an invalid or singular value for any
%   training sample, build_branch_cache marks that row as inadmissible.  This
%   routine removes the row from the analytical solve for all output rows and
%   scatters its coefficient back as zero in the full Wout matrix.  This keeps
%   the matrix-form ridge solve well defined and avoids irregular per-sample
%   feature deletion.
%
% If admissibleA is provided, admissibleA{src,L}(r,k)=false additionally
% forbids the k-th basis term in branch src for output row r. No additional
% prior penalty is added; inadmissible coefficients are removed from the
% hypothesis space.

	if nargin < 6 || isempty(lambdaOut)
		lambdaOut = 1e-8;
	end
	if nargin < 5 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	if nargin < 7
		admissibleA = [];
	end

	L = arch.layer;
	h = forward_inner_states(X, Coef_M, arch, normOpt);

	Zblocks = cell(1, L);
	validBlocks = cell(1, L);
	nBlock = zeros(1, L);

	for src = 1:L
		k = L - src + 1;
		branch = build_branch_cache(h{k}, arch, L, h, src);
		Zblocks{src} = branch.Phi;
		validBlocks{src} = get_branch_valid_rows_local(branch);
		nBlock(src) = size(branch.Phi, 1);
	end

	Zout = vertcat(Zblocks{:});
	validFeatureRows = vertcat(validBlocks{:}).';
	invalidFinalFeatureRowsByBlock = cell(1, L);
	nValidFinalFeaturesByBlock = zeros(1, L);
	nInvalidFinalFeaturesByBlock = zeros(1, L);
	for src = 1:L
		validBlock = logical(validBlocks{src}(:));
		invalidFinalFeatureRowsByBlock{src} = find(~validBlock);
		nValidFinalFeaturesByBlock(src) = nnz(validBlock);
		nInvalidFinalFeaturesByBlock(src) = nnz(~validBlock);
	end
	Ymat = Y.';

	if size(Ymat, 1) ~= arch.ny
		error('Output dimension mismatch: Y has %d columns, but arch.ny = %d.', size(Y, 2), arch.ny);
	end

	allowedOut = build_final_readout_allowed_mask_local(admissibleA, L, nBlock, arch.ny);
	if isempty(allowedOut)
		allowedOut = true(arch.ny, size(Zout, 1));
	end
	allowedOut = allowedOut & repmat(validFeatureRows, arch.ny, 1);
	useMaskedReadout = ~all(allowedOut(:));

	Wout = zeros(arch.ny, size(Zout, 1));
	if useMaskedReadout
		for r = 1:arch.ny
			idxAllowed = find(allowedOut(r, :));
			if isempty(idxAllowed)
				continue;
			end
			Zr = Zout(idxAllowed, :);
			G = Zr * Zr.' + lambdaOut * eye(size(Zr, 1));
			Wout(r, idxAllowed) = (Ymat(r, :) * Zr.') / G;
		end
	else
		G = Zout * Zout.' + lambdaOut * eye(size(Zout, 1));
		Wout = (Ymat * Zout.') / G;
	end

	pos = 0;
	for src = 1:L
		idx = pos + (1:nBlock(src));
		Coef_M{src, L} = Wout(:, idx);
		pos = pos + nBlock(src);
	end

	if ~isempty(admissibleA)
		Coef_M = apply_admissible_mask_to_Coef(Coef_M, admissibleA);
	end

	% Use a safe prediction bank as an additional guard. Even when an invalid
	% feature coefficient is zero, MATLAB would still propagate NaN through
	% 0*NaN if the raw feature matrix contained non-finite entries.
	Zpred = Zout;
	Zpred(~validFeatureRows, :) = 0;
	Zpred(~isfinite(Zpred)) = 0;
	Ypred = (Wout * Zpred).';
	trainMSE = mean((Ypred(:) - Y(:)).^2);

	readoutInfo = struct();
	readoutInfo.lambdaOut = lambdaOut;
	readoutInfo.nFinalFeatures = size(Zout, 1);
	readoutInfo.trainMSE = trainMSE;
	readoutInfo.nBlock = nBlock;
	readoutInfo.useAdmissibleMask = ~isempty(admissibleA) || useMaskedReadout;
	readoutInfo.useInvalidRowFiltering = true;
	readoutInfo.nValidFinalFeatures = sum(validFeatureRows);
	readoutInfo.nInvalidFinalFeatures = sum(~validFeatureRows);
	readoutInfo.nValidFinalFeaturesByBlock = nValidFinalFeaturesByBlock;
	readoutInfo.nInvalidFinalFeaturesByBlock = nInvalidFinalFeaturesByBlock;
	readoutInfo.invalidFinalFeatureRowsByBlock = invalidFinalFeatureRowsByBlock;
	readoutInfo.validFinalFeatureRows = validFeatureRows;
	readoutInfo.nAllowedFinalFeaturesByRow = sum(allowedOut, 2);
	readoutInfo.nRemovedFinalFeaturesByRow = size(Zout, 1) - sum(allowedOut, 2);
	readoutInfo.invalidFilteringSummary = sprintf( ...
		'final readout: removed %d/%d feature rows before ridge solve', ...
		readoutInfo.nInvalidFinalFeatures, readoutInfo.nFinalFeatures);
end

function validRows = get_branch_valid_rows_local(branch)
	if isfield(branch, 'PhiValidRows') && ~isempty(branch.PhiValidRows)
		validRows = logical(branch.PhiValidRows(:));
	elseif isfield(branch, 'PhiInvalidRows') && ~isempty(branch.PhiInvalidRows)
		validRows = ~logical(branch.PhiInvalidRows(:));
	else
		validRows = true(size(branch.Phi, 1), 1);
	end
	if numel(validRows) ~= size(branch.Phi, 1)
		error('Branch validity vector length %d does not match Phi row number %d.', ...
			numel(validRows), size(branch.Phi, 1));
	end
	validRows = validRows & all(isfinite(branch.Phi), 2);
end

function allowedOut = build_final_readout_allowed_mask_local(admissibleA, L, nBlock, ny)
	if nargin < 1 || isempty(admissibleA)
		allowedOut = [];
		return;
	end

	nTotal = sum(nBlock);
	allowedOut = true(ny, nTotal);
	pos = 0;
	for src = 1:L
		idx = pos + (1:nBlock(src));
		pos = pos + nBlock(src);
		if src > size(admissibleA, 1) || L > size(admissibleA, 2) || isempty(admissibleA{src, L})
			continue;
		end
		Ablock = logical(admissibleA{src, L});
		if ~isequal(size(Ablock), [ny, nBlock(src)])
			error('admissibleA{%d,%d} has size [%s], expected [%d %d].', ...
				src, L, num2str(size(Ablock)), ny, nBlock(src));
		end
		allowedOut(:, idx) = Ablock;
	end
end
