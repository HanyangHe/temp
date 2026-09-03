function Coef_disp = clean_Coef_M_for_display(Coef_M, nDigits, zeroTol, intTol)
%CLEAN_COEF_M_FOR_DISPLAY Clean coefficients for symbolic/display purposes.

	if nargin < 2
		nDigits = 6;
	end
	if nargin < 3
		zeroTol = 1e-10;
	end
	if nargin < 4
		intTol = 1e-8;
	end

	Coef_disp = Coef_M;

	for ell = 1:size(Coef_M, 1)
		for src = 1:size(Coef_M, 2)
			if isempty(Coef_M{src, ell})
				continue;
			end
			A = Coef_M{src, ell};
			A(abs(A) <= zeroTol) = 0;
			A_round = round(A);
			maskInt = abs(A - A_round) <= intTol;
			A(maskInt) = A_round(maskInt);
			A = round(A, nDigits);
			A(abs(A) <= zeroTol) = 0;
			Coef_disp{src, ell} = A;
		end
	end
end
