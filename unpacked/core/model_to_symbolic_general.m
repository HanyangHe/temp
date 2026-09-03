function expr = model_to_symbolic_general(nx, Coef_M, arch)
%MODEL_TO_SYMBOLIC_GENERAL Convert explicit-case-dictionary PhDN to symbolic.
%
% previous-version uses task-defined compact dictionaries.  Symbolic conversion follows the
% explicit term list instead of reconstructing a broad generated dictionary.

	if nargin < 3
		error('Usage: expr = model_to_symbolic_general(nx, Coef_M, arch)');
	end
	dims = get_arch_dims(arch);
	x = sym('x', [nx, 1], 'real');
	h = cell(1, arch.layer + 1);
	h{1} = x;

	for ell = 1:arch.layer
		tmp = sym(zeros(dims(ell + 1), 1));
		for src = 1:ell
			k = ell - src + 1;
			A = Coef_M{src, ell};
			if isempty(A); continue; end
			terms = build_symbolic_explicit_terms_local(h{k}, arch, ell, src);
			for row = 1:size(A, 1)
				activeIdx = find(A(row, :) ~= 0);
				rowExpr = sym(0);
				for aa = 1:numel(activeIdx)
					col = activeIdx(aa);
					if col <= numel(terms)
						rowExpr = rowExpr + sym(A(row, col)) * terms(col);
					end
				end
				tmp(row) = tmp(row) + rowExpr;
			end
		end
		h{ell + 1} = tmp;
	end
	expr = simplify(h{arch.layer + 1});
end

function terms = build_symbolic_explicit_terms_local(H, arch, ell, src)
	if nargin < 4
		src = [];
	end
	names = explicit_case_dictionary_terms(numel(H), arch, ell, src);
	terms = sym(zeros(numel(names), 1));
	for i = 1:numel(names)
		terms(i) = parse_sym_term_local(names{i}, H);
	end
end

function out = parse_sym_term_local(expr, H)
	expr = strip_outer_local(strrep(strtrim(char(expr)), ' ', ''));
	if strcmp(expr, '1')
		out = sym(1); return;
	end
	parts = split_top_local(expr, '*');
	if numel(parts) > 1
		out = sym(1);
		for i = 1:numel(parts)
			out = out * parse_sym_term_local(parts{i}, H);
		end
		return;
	end
	[pbase, pexp, hasPow] = split_pow_local(expr);
	if hasPow
		out = parse_sym_term_local(pbase, H)^pexp;
		return;
	end
	m = regexp(expr, '^([A-Za-z]\w*)\((.*)\)$', 'tokens', 'once');
	if ~isempty(m)
		arg = parse_sym_term_local(m{2}, H);
		chebOrder = parse_chebyshev_order_local(m{1});
		if ~isempty(chebOrder)
			out = chebyshev_sym_local(arg, chebOrder);
			return;
		end
		switch lower(m{1})
			case 'inv'; out = 1/arg;
			case 'sqrt'; out = sqrt(arg);
			case 'exp'; out = exp(arg);
			case 'sin'; out = sin(arg);
			case 'cos'; out = cos(arg);
			case 'tanh'; out = tanh(arg);
			case 'asin'; out = asin(arg);
			case 'log'; out = log(arg);
			otherwise; error('Unsupported symbolic operator %s', m{1});
		end
		return;
	end
	m = regexp(expr, '^v(\d+)$', 'tokens', 'once');
	if ~isempty(m)
		idx = str2double(m{1});
		out = H(idx); return;
	end
	num = str2double(expr);
	if isfinite(num); out = sym(num); return; end
	error('Cannot parse symbolic explicit term: %s', expr);
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

function out = chebyshev_sym_local(x, order)
	if order == 0
		out = sym(1);
	elseif order == 1
		out = x;
	else
		Tkm1 = sym(1);
		Tk = x;
		for k = 1:(order - 1)
			Tkp1 = 2*x*Tk - Tkm1;
			Tkm1 = Tk;
			Tk = Tkp1;
		end
		out = Tk;
	end
end

function parts = split_top_local(s, sep)
	parts = {}; level = 0; st = 1;
	for i=1:numel(s)
		if s(i)=='('; level=level+1; elseif s(i)==')'; level=level-1; end
		if level==0 && s(i)==sep
			parts{end+1}=s(st:i-1); %#ok<AGROW>
			st=i+1;
		end
	end
	parts{end+1}=s(st:end);
end

function [base, expo, tf] = split_pow_local(s)
	level=0; pos=0;
	for i=numel(s):-1:1
		if s(i)==')'; level=level+1; elseif s(i)=='('; level=level-1; end
		if level==0 && s(i)=='^'; pos=i; break; end
	end
	if pos==0; base=''; expo=NaN; tf=false; return; end
	base=s(1:pos-1); expo=str2double(s(pos+1:end)); tf=isfinite(expo);
end

function s = strip_outer_local(s)
	while numel(s)>=2 && s(1)=='(' && s(end)==')'
		mid=s(2:end-1); level=0; ok=true;
		for i=1:numel(mid)
			if mid(i)=='('; level=level+1; elseif mid(i)==')'; level=level-1; end
			if level<0; ok=false; break; end
		end
		if ok && level==0; s=mid; else; break; end
	end
end
