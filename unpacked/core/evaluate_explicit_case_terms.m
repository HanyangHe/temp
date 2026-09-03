function [Phi, Jphi, invalidRows, terms] = evaluate_explicit_case_terms(H, termNames, safety, arch, rawPySRSemanticsRows)
%EVALUATE_EXPLICIT_CASE_TERMS Evaluate compact case-specific terms and Jacobians.
%
% H is d-by-N. Jphi is nTerms-by-d-by-N.  Only a small canonical grammar is
% supported by previous-version compact dictionaries:
%   1, vi, vi^p, products a*b, inv(expr), sqrt(expr), exp(expr), sin(expr), cos(expr), tanh(expr), asin(expr), log(expr)
% Terms are explicit; generated general-mode dictionaries may also use Tn(expr) Chebyshev terms.
% rawPySRSemanticsRows marks compiled Stage-0 structural terms whose forward
% operators must exactly follow official PySR: ordinary / and inv, sqrt(abs)
% where exported, unclipped exp, real-domain asin, and positive-domain log.

	if nargin < 3 || isempty(safety)
		safety = struct();
	end
	if nargin < 4
		arch = struct();
	end
	if nargin < 5 || isempty(rawPySRSemanticsRows)
		rawPySRSemanticsRows = false(numel(termNames), 1);
	end
	rawPySRSemanticsRows = logical(rawPySRSemanticsRows(:));
	if numel(rawPySRSemanticsRows) ~= numel(termNames)
		error('rawPySRSemanticsRows must contain one flag per explicit term.');
	end
	if ~isfield(safety, 'eps') || isempty(safety.eps)
		safety.eps = 1e-8;
	end

	d = size(H, 1);
	N = size(H, 2);
	terms = termNames(:);
	nT = numel(terms);
	Phi = zeros(nT, N);
	Jphi = zeros(nT, d, N);
	invalidRows = false(nT, 1);
	for t = 1:nT
		name = strtrim(char(terms{t}));
		try
			[val, grad] = eval_expr_local(name, H, safety, arch, rawPySRSemanticsRows(t));
			bad = ~isfinite(val) | any(~isfinite(grad), 1);
			invalidRows(t) = any(bad);
			val(~isfinite(val)) = 0;
			grad(~isfinite(grad)) = 0;
			Phi(t, :) = val;
			for n = 1:N
				Jphi(t, :, n) = grad(:, n).';
			end
		catch ME
			warning('previous-version explicit dictionary term "%s" failed: %s. Row is set invalid.', name, ME.message);
			invalidRows(t) = true;
		end
	end
end

function [val, grad] = eval_expr_local(expr, H, safety, arch, useRawPySRSemantics)
	expr = strip_outer_parentheses_local(strrep(strtrim(expr), ' ', ''));
	d = size(H, 1);
	N = size(H, 2);
	if strcmp(expr, '1')
		val = ones(1, N);
		grad = zeros(d, N);
		return;
	end

	% Top-level addition/subtraction.  This must precede unary-sign handling:
	% -v1+v2 means (-v1)+v2, not -(v1+v2).
	% terms such as (v1-v2)^2 or v1+v2.
	[sumParts, sumSigns] = split_top_level_add_sub_local(expr);
	if numel(sumParts) > 1
		val = zeros(1, N);
		grad = zeros(d, N);
		for i = 1:numel(sumParts)
			[vi, gi] = eval_expr_local(sumParts{i}, H, safety, arch, useRawPySRSemantics);
			val = val + sumSigns(i) .* vi;
			grad = grad + sumSigns(i) .* gi;
		end
		return;
	end

	% Unary leading sign for symbolic custom terms after top-level sums have
	% been separated.
	if startsWith(expr, '+') && numel(expr) > 1
		[val, grad] = eval_expr_local(expr(2:end), H, safety, arch, useRawPySRSemantics);
		return;
	elseif startsWith(expr, '-') && numel(expr) > 1 && ~isfinite(str2double(expr))
		[val, grad] = eval_expr_local(expr(2:end), H, safety, arch, useRawPySRSemantics);
		val = -val;
		grad = -grad;
		return;
	end

	% Top-level multiplication/division.  PySR expressions may contain
	% explicit / operators, so the parser supports both * and / at the same
	% precedence level. Stage-0 structural terms use ordinary PySR division;
	% other dictionary terms retain the protected PhDN denominator policy.
	[mdParts, mdOps] = split_top_level_mul_div_local(expr);
	if numel(mdParts) > 1
		[val, grad] = eval_expr_local(mdParts{1}, H, safety, arch, useRawPySRSemantics);
		for i = 2:numel(mdParts)
			[vi, gi] = eval_expr_local(mdParts{i}, H, safety, arch, useRawPySRSemantics);
			if mdOps(i-1) == '*'
				grad = grad .* vi + gi .* val;
				val = val .* vi;
			else
				if useRawPySRSemantics
					% Official PySR/Julia division: no denominator clipping.
					den = vi;
				else
					sgn = sign(vi); sgn(sgn == 0) = 1;
					den = sgn .* max(abs(vi), safety.eps);
				end
				grad = grad ./ den - (val .* gi) ./ (den.^2);
				val = val ./ den;
			end
		end
		return;
	end

	% Power: only top-level ^ with numeric exponent.
	[pbase, pexp, hasPow] = split_power_local(expr);
	if hasPow
		[v, g] = eval_expr_local(pbase, H, safety, arch, useRawPySRSemantics);
		val = v .^ pexp;
		grad = g .* (pexp .* (v .^ (pexp - 1)));
		return;
	end



	% Unary function call.
	[fname, arg, isFun] = parse_function_local(expr);
	if isFun
		[u, du] = eval_expr_local(arg, H, safety, arch, useRawPySRSemantics);
		bsplineIndex = parse_bspline_index_local(fname);
		if ~isempty(bsplineIndex)
			[val, dval] = bspline_basis_value_derivative_local(u, bsplineIndex, arch);
			grad = du .* dval;
			return;
		end

		chebOrder = parse_chebyshev_order_local(fname);
		if ~isempty(chebOrder)
			[val, dval] = chebyshev_value_derivative_local(u, chebOrder);
			grad = du .* dval;
			return;
		end
		switch lower(fname)
			case {'re','real','conj','conjugate'}
				% The PhDN/PySR bridge is explicitly real-valued.  SymPy may
				% introduce re(vK) or conjugate(vK) while simplifying an
				% expression whose symbols were created without real assumptions.
				% These wrappers are exact identities on the real feature matrix.
				val = u;
				grad = du;
			case {'im','imag'}
				% The imaginary part of every real-valued compiled subexpression
				% is identically zero.  Keep this defensive branch so an exporter
				% bookkeeping wrapper can never invalidate an otherwise supported
				% Stage-0 structure.
				val = zeros(size(u));
				grad = zeros(size(du));
			case 'inv'
				if useRawPySRSemantics
					den = u;
				else
					sgn = sign(u); sgn(sgn == 0) = 1;
					den = sgn .* max(abs(u), safety.eps);
				end
				val = 1 ./ den;
				grad = du .* (-1 ./ (den.^2));
			case {'square','sqr'}
				val = u.^2;
				grad = du .* (2 .* u);
			case 'cube'
				val = u.^3;
				grad = du .* (3 .* u.^2);
			case {'abs','Abs'}
				val = abs(u);
				grad = du .* sign(u);
			case {'sqrt','sqrt_abs'}
				if useRawPySRSemantics
					if strcmpi(fname, 'sqrt_abs')
						z = abs(u);
						val = sqrt(z);
						dval = zeros(size(u));
						nz = z > 0;
						dval(nz) = 0.5 .* sign(u(nz)) ./ sqrt(z(nz));
						grad = du .* dval;
					else
						valid = u >= 0;
						val = nan(size(u));
						val(valid) = sqrt(u(valid));
						dval = nan(size(u));
						positive = u > 0;
						dval(positive) = 0.5 ./ sqrt(u(positive));
						dval(u == 0) = 0;
						grad = du .* dval;
					end
				elseif strcmpi(fname, 'sqrt_abs')
					z = max(abs(u), safety.eps);
					val = sqrt(z);
					grad = du .* (0.5 .* sign(u) ./ sqrt(z));
				else
					z = max(u, safety.eps);
					val = sqrt(z);
					grad = du .* (0.5 ./ sqrt(z));
				end
			case 'exp'
				if useRawPySRSemantics
					val = exp(u);
				else
					z = min(max(u, -50), 50);
					val = exp(z);
				end
				grad = du .* val;
			case 'sin'
				val = sin(u);
				grad = du .* cos(u);
			case 'cos'
				val = cos(u);
				grad = du .* (-sin(u));
			case 'tanh'
				val = tanh(u);
				grad = du .* (1 - val.^2);
			case 'asin'
				if useRawPySRSemantics
					valid = abs(u) <= 1;
					val = nan(size(u));
					val(valid) = asin(u(valid));
					dval = nan(size(u));
					interior = abs(u) < 1;
					dval(interior) = 1 ./ sqrt(1 - u(interior).^2);
					dval(abs(u) == 1) = 0;
					grad = du .* dval;
				else
					z = min(max(u, -1 + safety.eps), 1 - safety.eps);
					val = asin(z);
					grad = du .* (1 ./ sqrt(max(1 - z.^2, safety.eps)));
				end
			case 'log'
				if useRawPySRSemantics
					valid = u > 0;
					val = nan(size(u));
					val(valid) = log(u(valid));
					dval = nan(size(u));
					dval(valid) = 1 ./ u(valid);
					grad = du .* dval;
				else
					z = sign(u) .* max(abs(u), safety.eps);
					val = log(abs(z));
					grad = du .* (1 ./ z);
				end
			otherwise
				error('Unsupported explicit operator: %s', fname);
		end
		return;
	end

	% Variable vK.
	m = regexp(expr, '^v(\d+)$', 'tokens', 'once');
	if ~isempty(m)
		idx = str2double(m{1});
		if idx < 1 || idx > d
			error('Variable index v%d exceeds input dimension %d.', idx, d);
		end
		val = H(idx, :);
		grad = zeros(d, N);
		grad(idx, :) = 1;
		return;
	end

	% Numeric constant.
	num = str2double(expr);
	if isfinite(num)
		val = num * ones(1, N);
		grad = zeros(d, N);
		return;
	end

	error('Cannot parse explicit dictionary term: %s', expr);
end



function [parts, ops] = split_top_level_mul_div_local(s)
	parts = {};
	ops = [];
	level = 0;
	start = 1;
	for i = 1:numel(s)
		ch = s(i);
		if ch == '('
			level = level + 1;
		elseif ch == ')'
			level = level - 1;
		end
		if level == 0 && (ch == '*' || ch == '/')
			% Do not split the second star in Python-style exponentiation; these
			% should normally be converted to ^ before evaluation, but this guard
			% keeps the parser robust.
			if ch == '*' && ((i < numel(s) && s(i+1) == '*') || (i > 1 && s(i-1) == '*'))
				continue;
			end
			piece = s(start:i-1);
			if ~isempty(piece)
				parts{end+1} = piece; %#ok<AGROW>
				ops(end+1) = ch; %#ok<AGROW>
			end
			start = i + 1;
		end
	end
	piece = s(start:end);
	if ~isempty(piece)
		parts{end+1} = piece; %#ok<AGROW>
	end
	if numel(parts) <= 1
		ops = [];
	end
end

function [parts, signs] = split_top_level_add_sub_local(s)
	parts = {};
	signs = [];
	level = 0;
	start = 1;
	currentSign = 1;
	for i = 1:numel(s)
		ch = s(i);
		if ch == '('
			level = level + 1;
		elseif ch == ')'
			level = level - 1;
		end
		if level == 0 && (ch == '+' || ch == '-') && is_binary_add_sub_sign_local(s, i)
			piece = s(start:i-1);
			if ~isempty(piece)
				parts{end+1} = piece; %#ok<AGROW>
				signs(end+1) = currentSign; %#ok<AGROW>
			end
			if ch == '+'
				currentSign = 1;
			else
				currentSign = -1;
			end
			start = i + 1;
		end
	end
	piece = s(start:end);
	if ~isempty(piece)
		parts{end+1} = piece; %#ok<AGROW>
		signs(end+1) = currentSign; %#ok<AGROW>
	end
end

function tf = is_binary_add_sub_sign_local(s, pos)
	if pos <= 1
		tf = false;
		return;
	end
	prev = s(pos - 1);
	% Signs after these symbols are unary or exponent signs, not binary +/-.
	if any(prev == ['(', '*', '/', '^', '+', '-', 'e', 'E'])
		tf = false;
	else
		tf = true;
	end
end

function parts = split_top_level_local(s, sep)
	parts = {};
	level = 0;
	start = 1;
	for i = 1:numel(s)
		ch = s(i);
		if ch == '('; level = level + 1; elseif ch == ')'; level = level - 1; end
		if level == 0 && ch == sep
			parts{end+1} = s(start:i-1); %#ok<AGROW>
			start = i + 1;
		end
	end
	parts{end+1} = s(start:end);
end

function [base, exponent, hasPow] = split_power_local(s)
	level = 0; pos = 0;
	for i = numel(s):-1:1
		ch = s(i);
		if ch == ')'; level = level + 1; elseif ch == '('; level = level - 1; end
		if level == 0 && ch == '^'
			pos = i; break;
		end
	end
	if pos == 0
		base = ''; exponent = NaN; hasPow = false; return;
	end
	base = s(1:pos-1);
	exponentText = strip_outer_parentheses_local(strtrim(s(pos+1:end)));
	exponent = parse_numeric_exponent_local(exponentText);
	hasPow = isfinite(exponent);
end

function exponent = parse_numeric_exponent_local(text)
%PARSE_NUMERIC_EXPONENT_LOCAL Parse decimal, integer, or signed rational powers.
% Supports PySR/SymPy forms such as ^(-1/4), ^(3/2), ^-2, and ^0.5.
	text = strip_outer_parentheses_local(strtrim(char(text)));
	exponent = str2double(text);
	if isfinite(exponent)
		return;
	end
	tok = regexp(text, '^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)/([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$', 'tokens', 'once');
	if isempty(tok)
		exponent = NaN;
		return;
	end
	numerator = str2double(tok{1});
	denominator = str2double(tok{2});
	if ~isfinite(numerator) || ~isfinite(denominator) || denominator == 0
		exponent = NaN;
	else
		exponent = numerator / denominator;
	end
end


function [fname, arg1, arg2, isFun] = parse_binary_function_local(s)
	m = regexp(s, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
	if isempty(m)
		fname = ''; arg1 = ''; arg2 = ''; isFun = false; return;
	end
	inside = m{2};
	level = 0; commaIndex = [];
	for k = 1:numel(inside)
		if inside(k) == '('; level = level + 1;
		elseif inside(k) == ')'; level = level - 1;
		elseif inside(k) == ',' && level == 0
			if ~isempty(commaIndex)
				fname = ''; arg1 = ''; arg2 = ''; isFun = false; return;
			end
			commaIndex = k;
		end
	end
	if isempty(commaIndex) || level ~= 0
		fname = ''; arg1 = ''; arg2 = ''; isFun = false; return;
	end
	fname = m{1};
	arg1 = inside(1:commaIndex-1);
	arg2 = inside(commaIndex+1:end);
	isFun = ~isempty(arg1) && ~isempty(arg2) && ...
		parentheses_balanced_local(arg1) && parentheses_balanced_local(arg2);
end

function [fname, arg, isFun] = parse_function_local(s)
	m = regexp(s, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
	if isempty(m)
		fname = ''; arg = ''; isFun = false; return;
	end
	fname = m{1}; arg = m{2};
	isFun = parentheses_balanced_local(arg);
end


function order = parse_chebyshev_order_local(fname)
	m = regexp(char(fname), '^[Tt](\d+)$', 'tokens', 'once');
	if isempty(m)
		order = [];
	else
		order = str2double(m{1});
		if ~isfinite(order) || order < 0 || abs(order - round(order)) > 0
			order = [];
		else
			order = round(order);
		end
	end
end

function [val, dval] = chebyshev_value_derivative_local(x, order)
	if order == 0
		val = ones(size(x));
		dval = zeros(size(x));
		return;
	elseif order == 1
		val = x;
		dval = ones(size(x));
		return;
	end
	Tkm1 = ones(size(x));
	Tk = x;
	dTkm1 = zeros(size(x));
	dTk = ones(size(x));
	for k = 1:(order - 1)
		Tkp1 = 2 .* x .* Tk - Tkm1;
		dTkp1 = 2 .* Tk + 2 .* x .* dTk - dTkm1;
		Tkm1 = Tk;
		Tk = Tkp1;
		dTkm1 = dTk;
		dTk = dTkp1;
	end
	val = Tk;
	dval = dTk;
end

function tf = parentheses_balanced_local(s)
	level = 0; tf = true;
	for i = 1:numel(s)
		if s(i) == '('; level = level + 1; elseif s(i) == ')'; level = level - 1; end
		if level < 0; tf = false; return; end
	end
	tf = level == 0;
end

function s = strip_outer_parentheses_local(s)
	while numel(s) >= 2 && s(1) == '(' && s(end) == ')' && parentheses_balanced_local(s(2:end-1))
		s = s(2:end-1);
	end
end


function idx = parse_bspline_index_local(fname)
	idx = [];
	m = regexp(char(fname), '^(?:Bsp|bspline|Bspline|BSpline)(\d+)$', 'tokens', 'once');
	if ~isempty(m)
		idx = str2double(m{1});
		if ~isfinite(idx) || idx < 1; idx = []; end
	end
end

function [val, dval] = bspline_basis_value_derivative_local(x, basisIndex, arch)
%BSPLINE_BASIS_VALUE_DERIVATIVE_LOCAL Uniform clamped B-spline basis.
% Function names use Bsp1(v1), Bsp2(v1), ... .
	cfg = struct();
	if isfield(arch, 'spline') && isstruct(arch.spline)
		cfg = arch.spline;
	elseif isfield(arch, 'v56Spline') && isstruct(arch.v56Spline)
		cfg = arch.v56Spline;
	end
	degree = getfield_default_bspline_local(cfg, 'degree', 3);
	numIntervals = getfield_default_bspline_local(cfg, 'numIntervals', 8);
	range = getfield_default_bspline_local(cfg, 'range', [-1, 1]);
	if numel(range) ~= 2 || ~all(isfinite(range)) || range(2) <= range(1)
		range = [-1, 1];
	end
	knots = clamped_uniform_knots_local(range(1), range(2), numIntervals, degree);
	nBasis = numIntervals + degree;
	if basisIndex > nBasis
		val = zeros(size(x));
		dval = zeros(size(x));
		return;
	end
	val = bspline_basis_recursive_local(basisIndex, degree, x, knots);
	if degree <= 0
		dval = zeros(size(x));
		return;
	end
	den1 = knots(basisIndex + degree) - knots(basisIndex);
	den2 = knots(basisIndex + degree + 1) - knots(basisIndex + 1);
	term1 = zeros(size(x));
	term2 = zeros(size(x));
	if abs(den1) > eps
		term1 = degree / den1 .* bspline_basis_recursive_local(basisIndex, degree - 1, x, knots);
	end
	if abs(den2) > eps
		term2 = degree / den2 .* bspline_basis_recursive_local(basisIndex + 1, degree - 1, x, knots);
	end
	dval = term1 - term2;
	val(~isfinite(val)) = 0;
	dval(~isfinite(dval)) = 0;
end

function B = bspline_basis_recursive_local(i, p, x, knots)
	if p == 0
		B = double(x >= knots(i) & x < knots(i + 1));
		% Include the right endpoint in the last nonzero basis.
		if i + 1 == numel(knots)
			B(x == knots(end)) = 1;
		else
			if i + 1 < numel(knots) && knots(i + 1) == knots(end)
				B(x == knots(end)) = 1;
			end
		end
		return;
	end
	den1 = knots(i + p) - knots(i);
	den2 = knots(i + p + 1) - knots(i + 1);
	B = zeros(size(x));
	if abs(den1) > eps
		B = B + ((x - knots(i)) ./ den1) .* bspline_basis_recursive_local(i, p - 1, x, knots);
	end
	if abs(den2) > eps
		B = B + ((knots(i + p + 1) - x) ./ den2) .* bspline_basis_recursive_local(i + 1, p - 1, x, knots);
	end
	B(~isfinite(B)) = 0;
end

function knots = clamped_uniform_knots_local(lb, ub, numIntervals, degree)
	numIntervals = max(1, round(numIntervals));
	degree = max(0, round(degree));
	if numIntervals == 1
		internal = [];
	else
		grid = linspace(lb, ub, numIntervals + 1);
		internal = grid(2:end-1);
	end
	knots = [repmat(lb, 1, degree + 1), internal, repmat(ub, 1, degree + 1)];
end

function v = getfield_default_bspline_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		v = s.(name);
	else
		v = defaultVal;
	end
end
