function [J, gradActive] = obj_general_masked_withgrad(thetaActive, thetaAnchor, betaAnchor, mask, Coef_zero, X, Y, arch, lambda1, lambda2, epsSmoothL1, normOpt)
%OBJ_GENERAL_MASKED_WITHGRAD Objective and gradient for active masked parameters.
%
% Updated for layer-wise hidden dimensions.

	if nargin < 12 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	if nargin < 3 || isempty(betaAnchor)
		betaAnchor = 0;
	end

	Coef_M = unpack_Coef_M_by_mask(thetaActive, mask, Coef_zero);
	[J, gradCoef] = obj_phdn_core_withgrad(Coef_M, X, Y, arch, lambda1, lambda2, epsSmoothL1, normOpt);

	if betaAnchor > 0
		diff = thetaActive - thetaAnchor;
		J = J + betaAnchor * sum(diff.^2);
		gradAnchor = 2 * betaAnchor * diff;
	else
		gradAnchor = zeros(size(thetaActive));
	end

	gradActive = pack_Coef_M_by_mask(gradCoef, mask) + gradAnchor;
end
