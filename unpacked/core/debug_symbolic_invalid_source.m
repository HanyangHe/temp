function report = debug_symbolic_invalid_source(result, outputIdx, coefSelector)
%DEBUG_SYMBOLIC_INVALID_SOURCE Locate coefficients causing symbolic NaN/Inf.
%
% Usage:
%   report = debug_symbolic_invalid_source(result, 3);
%   report = debug_symbolic_invalid_source(result, 3, 'Coef_M_est');
%   report = debug_symbolic_invalid_source(result, 3, result.Coef_M_est);

	if nargin < 2 || isempty(outputIdx)
		outputIdx = 3;
	end

	if ~isfield(result, 'task') || ~isfield(result, 'arch')
		error('result.task and result.arch are required.');
	end

	task = result.task;
	arch = result.arch;

	fprintf('\n====================================================\n');
	fprintf('Symbolic invalid-source diagnostic for output %d\n', outputIdx);
	fprintf('====================================================\n');

	coefCases = {};

    if nargin >= 3 && ~isempty(coefSelector)
	    if ischar(coefSelector) || isstring(coefSelector)
		    fieldName = char(coefSelector);
    
		    if ~isfield(result, fieldName)
			    error('result does not contain field: %s', fieldName);
		    end
    
		    coefCases{end+1} = struct( ...
			    'name', ['result.', fieldName], ...
			    'Coef_M', {result.(fieldName)});   % Important: wrap cell in {}
    
	    else
		    coefCases{end+1} = struct( ...
			    'name', 'manual CoefInput', ...
			    'Coef_M', {coefSelector});         % Important: wrap cell in {}
	    end
    else
	    candidateFields = {'Coef_M_est', 'Coef_disp', 'Coef_sym'};
    
	    for k = 1:numel(candidateFields)
		    fieldName = candidateFields{k};
    
		    if isfield(result, fieldName)
			    coefCases{end+1} = struct( ...
				    'name', ['result.', fieldName], ...
				    'Coef_M', {result.(fieldName)});   % Important: wrap cell in {}
		    end
	    end
    end

	if isempty(coefCases)
		error('No coefficient candidates were found in result.');
	end

	report = struct();
	report.outputIdx = outputIdx;
	report.cases = struct([]);

	for iCase = 1:numel(coefCases)
		caseName = coefCases{iCase}.name;
		Coef_M = coefCases{iCase}.Coef_M;

		fprintf('\n----------------------------------------------------\n');
		fprintf('Checking %s\n', caseName);
		fprintf('class(Coef_M) = %s\n', class(Coef_M));
		fprintf('size(Coef_M)  = [%s]\n', num2str(size(Coef_M)));
		fprintf('iscell(Coef_M)= %d\n', iscell(Coef_M));
		fprintf('----------------------------------------------------\n');

		caseReport = struct();
		caseReport.name = caseName;
		caseReport.okAll = false;
		caseReport.message = '';
		caseReport.badOutput = false;
		caseReport.blockRowFixes = [];
		caseReport.singleCoefFixes = [];

		if ~is_valid_coef_cell(Coef_M, arch)
			fprintf('Skipped: this candidate is not a valid Coef_M cell.\n');
			fprintf('If class is double, you probably passed one block such as result.Coef_M_est{1,1}, not the full cell.\n');
			caseReport.message = 'Invalid Coef_M cell.';
			report.cases = [report.cases; caseReport];
			continue;
		end

		[expr, ok, msg] = try_symbolic_expr(task, Coef_M, arch);

		caseReport.okAll = ok;
		caseReport.message = msg;

		if ~ok
			fprintf('Full symbolic translation failed: %s\n', msg);
			report.cases = [report.cases; caseReport];
			continue;
		end

		exprOut = expr(outputIdx);
		bad0 = is_invalid_symbolic(exprOut);
		caseReport.badOutput = bad0;

		fprintf('Original output %d invalid? %d\n', outputIdx, bad0);
		fprintf('Original output %d expression:\n', outputIdx);
		disp(exprOut);

		if ~bad0
			fprintf('No NaN/Inf detected in this coefficient case.\n');
			report.cases = [report.cases; caseReport];
			continue;
		end

		fprintf('\nLevel 1: block-row ablation\n');

		blockRowFixes = [];

		for ell = 1:size(Coef_M, 2)
			for src = 1:ell
				A = Coef_M{src, ell};

				if isempty(A) || ~isnumeric(A)
					continue;
				end

				for r = 1:size(A, 1)
					if ~any(A(r, :) ~= 0)
						continue;
					end

					Coef_test = Coef_M;
					Coef_test{src, ell}(r, :) = 0;

					[exprTest, okTest, ~] = try_symbolic_expr(task, Coef_test, arch);

					if ~okTest
						continue;
					end

					badTest = is_invalid_symbolic(exprTest(outputIdx));

					if ~badTest
						fprintf('  FIX by zeroing block row: Coef_M{%d,%d}, row %d\n', ...
							src, ell, r);

						blockRowFixes = [blockRowFixes; src, ell, r]; %#ok<AGROW>
					end
				end
			end
		end

		if isempty(blockRowFixes)
			fprintf('  No single block row alone fixes the invalid output.\n');
		end

		caseReport.blockRowFixes = blockRowFixes;

		fprintf('\nLevel 2: single-coefficient ablation\n');

		singleCoefFixes = [];

		for ell = 1:size(Coef_M, 2)
			for src = 1:ell
				A = Coef_M{src, ell};

				if isempty(A) || ~isnumeric(A)
					continue;
				end

				[rowIdx, colIdx] = find(A ~= 0);

				for k = 1:numel(rowIdx)
					r = rowIdx(k);
					c = colIdx(k);
					oldVal = A(r, c);

					Coef_test = Coef_M;
					Coef_test{src, ell}(r, c) = 0;

					[exprTest, okTest, ~] = try_symbolic_expr(task, Coef_test, arch);

					if ~okTest
						continue;
					end

					badTest = is_invalid_symbolic(exprTest(outputIdx));

					if ~badTest
						fprintf('  FIX by zeroing coefficient: Coef_M{%d,%d}(%d,%d) = %.16e\n', ...
							src, ell, r, c, oldVal);

						singleCoefFixes = [singleCoefFixes; src, ell, r, c, oldVal]; %#ok<AGROW>
					end
				end
			end
		end

		if isempty(singleCoefFixes)
			fprintf('  No single coefficient alone fixes the invalid output.\n');
			fprintf('  This suggests a multi-coefficient path or translator-level full-basis construction issue.\n');
		end

		caseReport.singleCoefFixes = singleCoefFixes;
		report.cases = [report.cases; caseReport];
	end

	fprintf('\nDiagnostic finished.\n');
end

function tf = is_valid_coef_cell(C, arch)
%IS_VALID_COEF_CELL Check whether C looks like a PhDN coefficient cell.

	tf = false;

	if ~iscell(C)
		return;
	end

	if ndims(C) ~= 2
		return;
	end

	if isfield(arch, 'layer') && ~isempty(arch.layer)
		L = arch.layer;

		if size(C, 1) ~= L || size(C, 2) ~= L
			return;
		end
	end

	tf = true;
end

function [expr, ok, msg] = try_symbolic_expr(task, Coef_M, arch)
%TRY_SYMBOLIC_EXPR Safely call the model symbolic translator.

	ok = true;
	msg = '';
	expr = sym([]);

	try
		if isfield(task, 'modelToSymbolicFcn') && ~isempty(task.modelToSymbolicFcn)
			try
				expr = task.modelToSymbolicFcn(task.nx, Coef_M, arch);
			catch
				expr = task.modelToSymbolicFcn(task.nx, Coef_M, arch.layer, arch.polyOrder);
			end
		else
			ok = false;
			msg = 'task.modelToSymbolicFcn is missing.';
		end
	catch ME
		ok = false;
		msg = ME.message;
	end
end

function bad = is_invalid_symbolic(expr)
%IS_INVALID_SYMBOLIC Detect NaN/Inf/zoo in numeric or symbolic expression.

	bad = false;

	if isempty(expr)
		return;
	end

	% Case 1: numeric expression.
	if isnumeric(expr)
		bad = any(~isfinite(expr(:)));
		return;
	end

	% Case 2: try direct numeric conversion.
	% This works when expr is sym(NaN), sym(Inf), or a constant symbolic value.
	try
		val = double(expr);
		if isnumeric(val)
			bad = any(~isfinite(val(:)));
			if bad
				return;
			end
		end
	catch
		% Nonconstant symbolic expressions cannot be converted to double.
	end

	% Case 3: robust string/char check.
	try
		sList = cell(numel(expr), 1);

		for k = 1:numel(expr)
			sList{k} = lower(char(expr(k)));
		end

		s = strjoin(sList, ' ');

		bad = contains(s, 'nan') || ...
			  contains(s, 'inf') || ...
			  contains(s, 'zoo') || ...
			  contains(s, 'undefined');

		if bad
			return;
		end
	catch
	end

	% Case 4: fallback using displayed text.
	try
		s = lower(evalc('disp(expr)'));

		bad = contains(s, 'nan') || ...
			  contains(s, 'inf') || ...
			  contains(s, 'zoo') || ...
			  contains(s, 'undefined');
	catch
		bad = true;
	end
end