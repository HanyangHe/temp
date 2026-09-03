function Coef_template = create_coef_template(arch)
%CREATE_COEF_TEMPLATE Create a zero coefficient-cell template for PhDN.
%
% Dimension convention:
%   dims = [nx, d2, ..., dL, ny].
%   For branch (src, ell):
%     input state index: k = ell - src + 1
%     input dimension  : dims(k)
%     output dimension : dims(ell+1)

	dims = get_arch_dims(arch);
	layer = arch.layer;

	Coef_template = cell(layer, layer);

	for ell = 1:layer
		rowDim = dims(ell + 1);

		for src = 1:ell
			k = ell - src + 1;
			inputDim = dims(k);
			nPhi = branch_dictionary_size(inputDim, arch, ell, src);
			Coef_template{src, ell} = zeros(rowDim, nPhi);
		end
	end
end
