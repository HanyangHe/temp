function dims = get_arch_dims(arch)
%GET_ARCH_DIMS Return PhDN state dimensions with flexible hidden widths.
%
% Convention:
%   h^{1} = x has dimension nx.
%   h^{2},...,h^{L} are hidden states with user-defined dimensions.
%   h^{L+1} = y has dimension ny.
%
% If no hidden width is specified, every hidden state defaults to dimension ny.
%
% Supported specifications:
%   arch.dims       : complete vector [nx, d2, ..., dL, ny], length L+1.
%   arch.hiddenDims : hidden dimensions [d2, ..., dL], length L-1.
%   arch.hiddenWidth: scalar hidden width used for all hidden layers.

	if ~isfield(arch, 'layer') || isempty(arch.layer)
		error('arch.layer is required.');
	end
	if ~isfield(arch, 'nx') || isempty(arch.nx)
		error('arch.nx is required.');
	end
	if ~isfield(arch, 'ny') || isempty(arch.ny)
		error('arch.ny is required.');
	end

	L = round(arch.layer);
	if L < 1
		error('arch.layer must be at least 1.');
	end

	nx = round(arch.nx);
	ny = round(arch.ny);

	if isfield(arch, 'dims') && ~isempty(arch.dims)
		dims = round(reshape(arch.dims, 1, []));
		if numel(dims) ~= L + 1
			error('arch.dims must have length arch.layer+1 = %d.', L + 1);
		end
		if dims(1) ~= nx
			error('arch.dims(1) must equal arch.nx = %d.', nx);
		end
		if dims(end) ~= ny
			error('arch.dims(end) must equal arch.ny = %d.', ny);
		end
	else
		nHidden = max(L - 1, 0);

		if isfield(arch, 'hiddenDims') && ~isempty(arch.hiddenDims)
			hiddenDims = round(reshape(arch.hiddenDims, 1, []));
			if numel(hiddenDims) == 1 && nHidden > 1
				hiddenDims = repmat(hiddenDims, 1, nHidden);
			end
			if numel(hiddenDims) ~= nHidden
				error('arch.hiddenDims must have length arch.layer-1 = %d.', nHidden);
			end
		elseif isfield(arch, 'hiddenWidth') && ~isempty(arch.hiddenWidth)
			hiddenDims = repmat(round(arch.hiddenWidth), 1, nHidden);
		else
			hiddenDims = repmat(ny, 1, nHidden);
		end

		dims = [nx, hiddenDims, ny];
	end

	if any(~isfinite(dims)) || any(dims < 1) || any(abs(dims - round(dims)) > 0)
		error('All architecture dimensions must be positive finite integers.');
	end
end
