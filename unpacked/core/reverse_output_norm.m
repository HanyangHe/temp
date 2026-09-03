function Y = reverse_output_norm(Yn, normOpt)
%REVERSE_OUTPUT_NORM Map normalized network outputs back to original scale.
	if nargin < 2 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	Y = Yn;
	if ~isfield(normOpt, 'useOutputNorm') || ~normOpt.useOutputNorm
		return;
	end
	style = lower(strtrim(getfield_default_norm_apply_local(normOpt, 'style', 'mapminmax')));
	switch style
		case {'none','off'}
			return;
		case {'mapminmax','minmax'}
			Y = (Yn - normOpt.ymin) ./ normOpt.yGain + normOpt.yOffset;
			if isfield(normOpt, 'yIsConstant') && any(normOpt.yIsConstant)
				Y(:, normOpt.yIsConstant) = repmat(normOpt.yOffset(normOpt.yIsConstant), size(Yn,1), 1);
			end
		case {'zscore','mapstd','standard'}
			Y = Yn .* normOpt.sigY + normOpt.muY;
		otherwise
			error('Unknown normalization style: %s.', style);
	end
	Y(~isfinite(Y)) = 0;
end

function val = getfield_default_norm_apply_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
