function [dH, dContext] = backprop_branch_local(branch, A, deltaNext, arch)
%BACKPROP_BRANCH_LOCAL Local backward contribution through one branch.
%
% For within-branch cross features z_ab = u_a * g_b:
%   dz_ab/dh = g_b * du_a/dh + u_a * dg_b/dh.
%
% Hidden-hidden mutual products between different branches are not used in
% the previous-version cross-only dictionary.


	% previous-version explicit case dictionary: use direct dPhi/dH cache.
	if isfield(branch, 'Jphi') && ~isempty(branch.Jphi)
		gPhi = A.' * deltaNext;
		nVar = size(branch.H, 1);
		Nsp = size(branch.H, 2);
		dH = zeros(nVar, Nsp);
		for ss = 1:Nsp
			Jss = branch.Jphi(:, :, ss);
			dH(:, ss) = Jss.' * gPhi(:, ss);
		end
		dContext = cell(1, arch.layer + 1);
		for kk = 1:numel(dContext); dContext{kk} = []; end
		dH(~isfinite(dH)) = 0;
		return;
	end

	gPhi = A.' * deltaNext;

	nVar = size(branch.H, 1);
	Nsp = size(branch.H, 2);

	gU = zeros(size(branch.U, 1), Nsp);
	gQ = zeros(size(branch.Q, 1), Nsp);
	gG = zeros(size(branch.G, 1), Nsp);
	gUleft = zeros(size(branch.Uleft, 1), Nsp);

	if isfield(branch.idx, 'baseU') && ~isempty(branch.idx.baseU)
		gU = gU + gPhi(branch.idx.baseU, :);
	end

	if isfield(branch.idx, 'baseG') && ~isempty(branch.idx.baseG)
		gG = gG + gPhi(branch.idx.baseG, :);
	end

	if isfield(branch, 'cross') && isfield(branch.cross, 'leftIndex')
		leftIndex = branch.cross.leftIndex;
		rightIndex = branch.cross.rightIndex;
		crossRows = branch.cross.rows;

		for r = 1:numel(leftIndex)
			row = crossRows(r);
			a = leftIndex(r);
			b = rightIndex(r);

			grad = gPhi(row, :);
			gUleft(a, :) = gUleft(a, :) + branch.G(b, :) .* grad;
			gG(b, :) = gG(b, :) + branch.Uleft(a, :) .* grad;
		end
	end

	nQ = size(branch.Q, 1);
	nOp = numel(branch.cfg.opNames);

	for op = 1:nOp
		idxG = (op - 1) * nQ + (1:nQ);
		if ~isempty(idxG)
			gQ = gQ + branch.dOp{op} .* gG(idxG, :);
		end
	end

	dH = zeros(nVar, Nsp);

	for s = 1:Nsp
		if ~isempty(gU)
			Ju_s = branch.Ju(:, :, s);
			dH(:, s) = dH(:, s) + Ju_s.' * gU(:, s);
		end

		if ~isempty(gUleft)
			Jleft_s = branch.Jleft(:, :, s);
			dH(:, s) = dH(:, s) + Jleft_s.' * gUleft(:, s);
		end

		if ~isempty(gQ)
			Jq_s = branch.Jq(:, :, s);
			dH(:, s) = dH(:, s) + Jq_s.' * gQ(:, s);
		end
	end

	dContext = cell(1, arch.layer + 1);
	for k = 1:numel(dContext)
		dContext{k} = [];
	end

	dH(~isfinite(dH)) = 0;
end
