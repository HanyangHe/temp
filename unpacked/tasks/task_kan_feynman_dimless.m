function task = task_kan_feynman_dimless(caseName, casemode)
%TASK_KAN_FEYNMAN_DIMLESS Dimensionless Feynman cases used for KAN-style comparison.
%
% This task file follows the dimensionless formulas listed in the KAN paper
% Feynman symbolic-regression comparison table. The purpose is to compare
% PhDN symbolic recovery against the same dimensionless benchmark forms,
% rather than the original unit-dependent physical formulas.
%
% Case names:
%   I_6_2, I_6_2b, I_9_18, I_12_11, I_13_12, I_15_3x,
%   I_16_6, I_18_4, I_26_2, I_27_6, I_29_16,
%   I_30_3, I_30_5, I_37_4, I_40_1, I_44_4, I_50_26,
%   II_2_42, II_6_15a, II_11_7, II_11_27, II_35_18,
%   II_36_38, II_38_3, III_9_52, III_10_19, III_17_37
%
% casemode:
%   strong_prior      : case-defined compact dictionary plus an exact row-wise admissible mask.
%   weak_strong_prior : case-defined compact dictionary plus a layerwise/blockwise
%                       dictionary mask induced by the strong-prior support.
%   weak_prior/lv1    : priorLevel 1; official PySR Stage-0 with case-compact SR grammar.
%   general           : priorLevel 0; official PySR Stage-0 with universal SR grammar.
%
% Notes:
%   compact-dictionary route cleanup rule: Feynman strong/weak-prior cases no longer carry
%   per-case polynomial/operator/interact-generation options.  Strong/weak-prior
%   branches use the task-defined uniform compact case dictionary.  General/lv1
%   modes use official PySR Stage-0 and do not install predefined PhDN dictionaries.

	if nargin < 1 || isempty(caseName)
		caseName = 'I_16_6';
	end
	if nargin < 2 || isempty(casemode)
		casemode = 'strong_prior';
	end

	caseName = lower(strtrim(caseName));
	casemode = lower(strtrim(casemode));

	% ---------------------------------------------------------------------
	% Global task-file option for all v60c prior-level subcases.
	%   'prior' : use the case-specific interpretable branchActive structure.
	%   'full'  : use branchActive = true(task.arch.layer, task.arch.layer).
	% This option only changes the branch-activity prior.  The selected prior
	% level still controls the dictionary granularity and row/block terms.
	% ---------------------------------------------------------------------
	branchActiveMode = 'prior';  % 'prior' | 'full'

	task = struct();
	task.caseName = caseName;
	task.casemode = casemode;
	task.sourceName = 'KAN_Feynman_dimensionless';
	% Benchmark data allocation: 1500 train, 500 validation, 500 ID test,
	% and an independent 500-sample OOD challenge set.
	task.dataDefaults = struct('nSamples', 2500, 'ratioTrain', 0.6, ...
		'ratioVal', 0.2, 'nOODSamples', 500);
	task.branchActiveMode = branchActiveMode;

	switch caseName
		case 'i_6_2'
			task = setup_I_6_2(task);
		case 'i_6_2b'
			task = setup_I_6_2b(task);
		case 'i_9_18'
			task = setup_I_9_18(task);
		case 'i_12_11'
			task = setup_I_12_11(task);
		case 'i_13_12'
			task = setup_I_13_12(task);
		case 'i_15_3x'
			task = setup_I_15_3x(task);
		case 'i_16_6'
			task = setup_I_16_6(task);
		case 'i_18_4'
			task = setup_I_18_4(task);
		case 'i_26_2'
			task = setup_I_26_2(task);
		case 'i_27_6'
			task = setup_I_27_6(task);
		case 'i_29_16'
			task = setup_I_29_16(task);
		case 'i_30_3'
			task = setup_I_30_3(task);
		case 'i_30_5'
			task = setup_I_30_5(task);
		case 'i_37_4'
			task = setup_I_37_4(task);
		case 'i_40_1'
			task = setup_I_40_1(task);
		case 'i_44_4'
			task = setup_I_44_4(task);
		case 'i_50_26'
			task = setup_I_50_26(task);
		case 'ii_2_42'
			task = setup_II_2_42(task);
		case 'ii_6_15a'
			task = setup_II_6_15a(task);
		case 'ii_11_7'
			task = setup_II_11_7(task);
		case 'ii_11_27'
			task = setup_II_11_27(task);
		case 'ii_35_18'
			task = setup_II_35_18(task);
		case 'ii_36_38'
			task = setup_II_36_38(task);
		case 'ii_38_3'
			task = setup_II_38_3(task);
		case 'iii_9_52'
			task = setup_III_9_52(task);
		case 'iii_10_19'
			task = setup_III_10_19(task);
		case 'iii_17_37'
			task = setup_III_17_37(task);
		otherwise
			error('Unknown KAN-Feynman dimensionless case: %s', caseName);
	end

	task = apply_common_defaults(task);
	task = apply_sr_stage0_prior_grammar(task);
	task = cleanup_sr_stage0_prior_levels(task);
	if is_sr_stage0_no_predefined_phdn_task(task)
		% priorLevel 0/1 now use official PySR as Stage 0.  No predefined
		% PhDN dictionary or case-specific PhDN architecture is built here.
	elseif isfield(task, 'prior') && isfield(task.prior, 'priorInterfaceEnabled') && task.prior.priorInterfaceEnabled
		% Prior-interface cases with priorLevel >= 2 build their own prior-level dictionary in the setup function.
	elseif strcmpi(task.casemode, 'general')
		task = apply_general_low_order_dictionary(task);
	else
		task = apply_uniform_compact_case_dictionary(task);
	end

	if exist('model_to_symbolic_general', 'file') == 2
		task.modelToSymbolicFcn = @model_to_symbolic_general;
	else
		task.modelToSymbolicFcn = [];
	end
end

% -------------------------------------------------------------------------
% Common helpers
% -------------------------------------------------------------------------

function task = apply_sr_stage0_prior_grammar(task)
%APPLY_SR_STAGE0_PRIOR_GRAMMAR Define the PySR Stage-0 grammar interface.
%
% For priorLevel 0/1, the PhDN structure is no longer treated as a predefined
% prior.  Instead, Stage 0 uses official PySR to propose expressions, and the
% main PhDN route compiles the selected per-output cores into SR-determined
% PhDN candidate architectures.  For priorLevel >= 2, this interface is kept
% disabled so those levels remain ablation tests of the older prior dictionaries.

	if ~isfield(task, 'prior') || ~isstruct(task.prior) || ...
			~isfield(task.prior, 'level') || isempty(task.prior.level)
		return;
	end
	priorLevel = task.prior.level;
	if priorLevel == 0
		task.prior.srStage0Enable = true;
		task.prior.srStage0UsePredefinedPhdnDictionary = false;
		if ~isfield(task.prior, 'srGrammar') || ~isstruct(task.prior.srGrammar) || isempty(fieldnames(task.prior.srGrammar))
			task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		else
			task.prior.srGrammar = complete_sr_stage0_grammar_local(task.prior.srGrammar, make_universal_sr_stage0_grammar_local());
		end
	elseif priorLevel == 1
		task.prior.srStage0Enable = true;
		task.prior.srStage0UsePredefinedPhdnDictionary = false;
		if ~isfield(task.prior, 'srGrammar') || ~isstruct(task.prior.srGrammar) || isempty(fieldnames(task.prior.srGrammar))
			unaryOps = infer_case_compact_sr_unary_ops_local(task);
			task.prior.srGrammar = struct();
			task.prior.srGrammar.source = 'level 1 case-based compact official-PySR grammar inferred from compact terms; no predefined PhDN dictionary is used for Stage 0';
			task.prior.srGrammar.binaryOperators = {'+', '-', '*', '/'};
			task.prior.srGrammar.unaryOperators = unaryOps;
		else
			defaultG = struct();
			defaultG.source = 'level 1 case-based compact official-PySR grammar; no predefined PhDN dictionary is used for Stage 0';
			defaultG.binaryOperators = {'+', '-', '*', '/'};
			defaultG.unaryOperators = {'inv','sqrt','log'};
			task.prior.srGrammar = complete_sr_stage0_grammar_local(task.prior.srGrammar, defaultG);
		end
	else
		task.prior.srStage0Enable = false;
		task.prior.srStage0UsePredefinedPhdnDictionary = true;
	end
	if any(priorLevel == [0 1]) && isfield(task.prior, 'srGrammar') && ...
			isstruct(task.prior.srGrammar)
		task.prior.srGrammar.unaryOperators = ensure_operator_local( ...
			get_struct_field_local(task.prior.srGrammar, 'unaryOperators', {}), 'log');
	end
end


function tf = is_sr_stage0_no_predefined_phdn_task(task)
%IS_SR_STAGE0_NO_PREDEFINED_PHDN_TASK True for priorLevel 0/1 SR-only Stage 0.
% These modes intentionally do not expose a predefined PhDN dictionary.  The
% PhDN candidate dictionary/architecture is compiled later from each selected
% official-PySR expression inside the main identification route.
	tf = false;
	if ~isfield(task, 'prior') || ~isstruct(task.prior)
		return;
	end
	if ~isfield(task.prior, 'level') || isempty(task.prior.level)
		return;
	end
	if ~any(task.prior.level == [0 1])
		return;
	end
	if isfield(task.prior, 'srStage0Enable') && logical(task.prior.srStage0Enable) && ...
			isfield(task.prior, 'srStage0UsePredefinedPhdnDictionary') && ...
			~logical(task.prior.srStage0UsePredefinedPhdnDictionary)
		tf = true;
	end
end

function task = cleanup_sr_stage0_prior_levels(task)
%CLEANUP_SR_STAGE0_PRIOR_LEVELS Remove old PhDN priors from priorLevel 0/1.
%
% setup_* functions still contain compact prior-level definitions for level >=2
% ablations.  For level 0 and level 1, however, the old dictionaries are not a
% PhDN prior anymore.  Level 0/1 only define the PySR grammar; every selected
% PySR equation is compiled into its own candidate PhDN architecture later in
% phdnn_identify.m.
	if ~is_sr_stage0_no_predefined_phdn_task(task)
		return;
	end
	priorLevel = task.prior.level;
	levelName = get_struct_field_local(task.prior, 'levelName', task.casemode);

	task.prior.stage0Mode = 'official_pysr';
	task.prior.srGrammarMode = ternary_string_local(priorLevel == 0, 'universal', 'case_compact');
	task.prior.usePresetPhdnDictionary = false;
	task.prior.srStage0UsePredefinedPhdnDictionary = false;
	task.prior.phdnStructureSource = 'selected official-PySR expression';
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s_sr_stage0_only_no_predefined_phdn_dictionary', priorLevel, levelName);
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';
	task.prior.useTwoStageMlpRecovery = false;

	% Clean all old PhDN-architecture priors from the task object.  This prevents
	% general/weak_prior_lv1 from silently falling back to the previous predefined
	% case dictionary if PySR fails or is disabled.
	if ~isfield(task, 'arch') || isempty(task.arch)
		task.arch = struct();
	end
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.operatorMode = 'true';
	task.arch.dictionaryMode = 'sr_stage0_expression_determined_no_predefined_dictionary';
	task.arch.branchActiveMask = true(1,1);
	task.arch.branchActiveMode = 'sr_stage0_only_placeholder';
	task.arch.srStage0Only = true;
	task.arch.srStage0StructureSource = 'per-output selected symbolic core archive';
	if isfield(task.arch, 'generalDictionary')
		task.arch = rmfield(task.arch, 'generalDictionary');
	end
	if isfield(task.arch, '')
		task.arch = rmfield(task.arch, '');
	end

	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = levelName;
	D.noFallback = true;
	D.appendGlobalTerms = false;
	D.termsByDim = {};
	D.source = 'empty SR-Stage0 placeholder: no predefined PhDN dictionary; candidates are compiled from selected PySR expressions';
	task.arch.caseDictionary = D;

	task.operatorMode = 'true';
	task.training.operatorMode = 'true';
	task.DisplaySymbolic = false;
end

function out = ternary_string_local(cond, a, b)
	if cond
		out = a;
	else
		out = b;
	end
end



function G = complete_sr_stage0_grammar_local(G, defaultG)
%COMPLETE_SR_STAGE0_GRAMMAR_LOCAL Fill missing SR grammar fields without overwriting case settings.
	if ~isfield(G, 'source') || isempty(G.source)
		G.source = defaultG.source;
	end
	if ~isfield(G, 'binaryOperators') || isempty(G.binaryOperators)
		G.binaryOperators = defaultG.binaryOperators;
	end
	if ~isfield(G, 'unaryOperators') || isempty(G.unaryOperators)
		G.unaryOperators = defaultG.unaryOperators;
	end
end

function operators = ensure_operator_local(operators, operatorName)
%ENSURE_OPERATOR_LOCAL Append one operator without disturbing existing order.
	operators = normalize_terms_cell_local(operators);
	if ~any(strcmpi(operators, operatorName))
		operators{end+1} = operatorName;
	end
end


function G = make_universal_sr_stage0_grammar_local()
%MAKE_UNIVERSAL_SR_STAGE0_GRAMMAR_LOCAL Universal official-PySR grammar for priorLevel 0.
% This grammar defines only the SR search space. It is not a PhDN dictionary.
	G = struct();
	G.source = 'level 0 universal official-PySR grammar; no predefined PhDN dictionary is used for Stage 0';
	G.binaryOperators = {'+', '-', '*', '/'};
	G.unaryOperators = {'square', 'cube', 'inv', 'sqrt', 'exp', 'sin', 'cos', 'log'};
end

function G = make_compact_sr_stage0_grammar_from_terms_local(compactTerms)
%MAKE_COMPACT_SR_STAGE0_GRAMMAR_FROM_TERMS_LOCAL Compact official-PySR grammar for priorLevel 1.
% The supplied terms are only used to infer which unary operators PySR may use.
% They are not installed as a PhDN dictionary, layer dictionary, branch mask, or
% predefined architecture.
	compactTerms = normalize_terms_cell_local(compactTerms);
	G = struct();
	G.source = 'level 1 case-based compact official-PySR grammar inferred from former compact terms; no predefined PhDN dictionary is used for Stage 0';
	G.binaryOperators = {'+', '-', '*', '/'};
	G.unaryOperators = infer_sr_unary_ops_from_term_list_local(compactTerms);
	G.compactTermSourceForGrammarOnly = compactTerms;
end

function unaryOps = infer_sr_unary_ops_from_term_list_local(terms)
%INFER_SR_UNARY_OPS_FROM_TERM_LIST_LOCAL Infer PySR unary operators from compact term strings.
	terms = normalize_terms_cell_local(terms);
	unaryOps = {};
	known = {'square','cube','inv','sqrt','exp','sin','cos','asin','log'};
	for i = 1:numel(known)
		op = known{i};
		if any(contains(terms, [op '(']))
			unaryOps{end+1} = op; %#ok<AGROW>
		end
	end
	if any(contains(terms, '^2')) && ~ismember('square', unaryOps)
		unaryOps = [{'square'}, unaryOps];
	end
	if any(contains(terms, '^3')) && ~ismember('cube', unaryOps)
		unaryOps = [{'cube'}, unaryOps];
	end
	if isempty(unaryOps)
		unaryOps = {'inv','sqrt'};
	end
	unaryOps = unique_stable_terms_local(unaryOps);
end

function task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName)
%APPLY_SR_STAGE0_NO_PREDEFINED_PHDN_PLACEHOLDER_LOCAL Remove old level-0/1 PhDN priors.
% For priorLevel 0/1, the actual PhDN architecture is determined later by each
% selected official-PySR expression. This placeholder only carries dimensions and
% SR grammar metadata so no old general/lv1 dictionary can be used accidentally.
	if nargin < 2 || isempty(D)
		D = struct();
	end
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;
	D.termsByDim = {};
	D.source = 'empty SR-Stage0 placeholder: no predefined PhDN dictionary; candidates are compiled from selected official-PySR expressions';

	task.prior.srStage0Enable = true;
	task.prior.srStage0UsePredefinedPhdnDictionary = false;
	task.prior.stage0Mode = 'official_pysr';
	task.prior.srGrammarMode = ternary_string_local(priorLevel == 0, 'universal', 'case_compact');
	task.prior.usePresetPhdnDictionary = false;
	task.prior.phdnStructureSource = 'selected official-PySR expression';
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s_sr_stage0_only_no_predefined_phdn_dictionary', priorLevel, priorName);
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';
	task.prior.useTwoStageMlpRecovery = false;

	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.operatorMode = 'true';
	task.arch.dictionaryMode = 'sr_stage0_expression_determined_no_predefined_dictionary';
	task.arch.branchActiveMask = true(1,1);
	task.arch.branchActiveMode = 'sr_stage0_only_placeholder';
	task.arch.srStage0Only = true;
	task.arch.srStage0StructureSource = 'per-output selected symbolic core archive';
	if isfield(task.arch, 'generalDictionary')
		task.arch = rmfield(task.arch, 'generalDictionary');
	end
	if isfield(task.arch, '')
		task.arch = rmfield(task.arch, '');
	end
	task.arch.caseDictionary = D;

	task.operatorMode = 'true';
	task.training.operatorMode = 'true';
	task.DisplaySymbolic = false;
end

function unaryOps = infer_case_compact_sr_unary_ops_local(task)
%INFER_CASE_COMPACT_SR_UNARY_OPS_LOCAL Use the case-defined compact prior terms
%to define a compact SR grammar for weak_prior_lv1.
	unaryOps = {};
	terms = {};
	if isfield(task, 'arch') && isfield(task.arch, 'caseDictionary') && isstruct(task.arch.caseDictionary)
		D = task.arch.caseDictionary;
		if isfield(D, 'termsByDim') && iscell(D.termsByDim)
			terms = [terms; collect_terms_from_cell_local(D.termsByDim)]; 
		end
		if isfield(D, 'termsByLayer') && iscell(D.termsByLayer)
			terms = [terms; collect_terms_from_cell_local(D.termsByLayer)]; 
		end
		if isfield(D, 'termsByBlock') && iscell(D.termsByBlock)
			terms = [terms; collect_terms_from_cell_local(D.termsByBlock)]; 
		end
	end
	terms = unique_stable_terms_local(terms);
	known = {'square','cube','inv','sqrt','exp','sin','cos','asin','log'};
	for i = 1:numel(known)
		op = known{i};
		if any(contains(terms, [op '(']))
			unaryOps{end+1} = op; 
		end
	end
	if any(contains(terms, '^2')) && ~ismember('square', unaryOps)
		unaryOps = [{'square'}, unaryOps];
	end
	if any(contains(terms, '^3')) && ~ismember('cube', unaryOps)
		unaryOps = [{'cube'}, unaryOps];
	end
	% Keep inv/sqrt available for compact physics-like rational/root cases when
	% the task setup did not expose enough term detail.
	if isempty(unaryOps)
		unaryOps = {'inv','sqrt'};
	end
	unaryOps = unique_stable_terms_local(unaryOps);
end

function terms = collect_terms_from_cell_local(C)
	terms = {};
	if ~iscell(C)
		return;
	end
	for i = 1:numel(C)
		ci = C{i};
		if isempty(ci)
			continue;
		elseif ischar(ci) || (isstring(ci) && isscalar(ci))
			terms{end+1,1} = char(ci); %#ok<AGROW>
		elseif iscell(ci)
			terms = [terms; collect_terms_from_cell_local(ci)]; %#ok<AGROW>
		end
	end
end

function task = apply_common_defaults(task)
	if ~isfield(task, 'prior')
		task.prior = struct();
	end
	if ~isfield(task.prior, 'useAdmissibleMask')
		task.prior.useAdmissibleMask = false;
	end
	if ~isfield(task.prior, 'maskType')
		task.prior.maskType = '';
	end
	% compact-mask route dictionary policy: for weak-prior Feynman cases, the compact
	% case dictionary is treated as a direct uniform prior dictionary for every
	% compatible PhDN branch. It is not built from an admissible mask at runtime.
	if ~isfield(task.prior, 'dictionaryMode') || isempty(task.prior.dictionaryMode)
		if any(strcmpi(task.casemode, {'strong_prior','weak_strong_prior','weak_prior'}))
			task.prior.dictionaryMode = 'uniform_compact_case_dictionary';
		else
			task.prior.dictionaryMode = 'general_low_order';
		end
	end
	if strcmpi(task.casemode, 'weak_strong_prior')
		% Intermediate prior mode: use the weak-prior compact dictionary, then
		% apply a dictionary support mask induced by the strong-prior support.
		% This is not the exact row-wise strong-prior mask.
		if ~isfield(task.prior, 'useWeakStrongDictionaryMask') || isempty(task.prior.useWeakStrongDictionaryMask)
			task.prior.useWeakStrongDictionaryMask = true;
		end
		if ~isfield(task.prior, 'dictionaryMaskGranularity') || isempty(task.prior.dictionaryMaskGranularity)
			task.prior.dictionaryMaskGranularity = 'layerwise'; % 'layerwise' | 'blockwise'
		end
		task.prior.useAdmissibleMask = false;
	else
		if ~isfield(task.prior, 'useWeakStrongDictionaryMask') || isempty(task.prior.useWeakStrongDictionaryMask)
			task.prior.useWeakStrongDictionaryMask = false;
		end
		if ~isfield(task.prior, 'dictionaryMaskGranularity') || isempty(task.prior.dictionaryMaskGranularity)
			task.prior.dictionaryMaskGranularity = 'none';
		end
	end
	if ~isfield(task, 'DisplaySymbolic')
		task.DisplaySymbolic = true;
	end
	if ~isfield(task, 'training')
		task.training = struct();
	end
	if ~isfield(task.training, 'lambda1List')
		task.training.lambda1List = [1e-8, 1e-6, 1e-4];
	end
	if ~isfield(task.training, 'tauList')
		task.training.tauList = [0, 1e-8, 1e-6];
	end
	if ~isfield(task.training, 'opArgPolyOrderList') || isempty(task.training.opArgPolyOrderList)
		% Explicit case dictionaries do not use generated operator-argument orders.
		task.training.opArgPolyOrderList = 1;
	end

	% Synchronize the operator-mode fields after each case-specific setup.
	% Rule: single-layer PhDN uses true operators by default and cannot keep a
	% surrogate operator mode.  This keeps one-layer PhDN comparable with the
	% true-operator single-layer SINDy baseline.
	if ~isfield(task, 'operatorControl') || isempty(task.operatorControl)
		task.operatorControl = struct();
	end
	if ~isfield(task, 'arch') || isempty(task.arch)
		task.arch = struct();
	end
	% previous-version fix: get_arch_dims requires arch.nx and arch.ny.  They must be
	% synchronized before apply_uniform_compact_case_dictionary builds the
	% compact case-level dictionary.
	if isfield(task, 'nx') && ~isempty(task.nx)
		task.arch.nx = task.nx;
	end
	if isfield(task, 'ny') && ~isempty(task.ny)
		task.arch.ny = task.ny;
	end
	if ~isfield(task.arch, 'layer') || isempty(task.arch.layer)
		task.arch.layer = 1;
	end
	if ~isfield(task.operatorControl, 'caseDefault') || isempty(task.operatorControl.caseDefault)
		if any(strcmpi(task.casemode, {'strong_prior','weak_strong_prior'})) || task.arch.layer == 1
			task.operatorControl.caseDefault = 'true';
		else
			task.operatorControl.caseDefault = 'surrogate';
		end
	end
	if task.arch.layer == 1
		task.operatorControl.caseDefault = 'true';
		task.operatorControl.singleLayerForceTrue = true;
	end
	if ~isfield(task, 'operatorMode') || isempty(task.operatorMode)
		task.operatorMode = task.operatorControl.caseDefault;
	end
	if task.arch.layer == 1
		task.operatorMode = 'true';
	end
	task.arch.operatorMode = task.operatorMode;
	task.training.operatorMode = task.operatorMode;
end


function task = apply_uniform_compact_case_dictionary(task)
%APPLY_UNIFORM_COMPACT_CASE_DICTIONARY Build the task-defined uniform compact case dictionary.
%
% compact-dictionary route uses a direct task-defined compact dictionary as the strong/weak-prior input.
% The same compact term list for a given input dimension is available to every
% compatible branch before BSP-LSQ searches branchwise submasks. This is not
% an oracle/admissible-mask-derived support restriction.

	if ~isfield(task, 'arch') || ~isfield(task.arch, 'layer')
		return;
	end
	% Defensive synchronization in case this helper is called directly.
	if isfield(task, 'nx') && ~isempty(task.nx)
		task.arch.nx = task.nx;
	end
	if isfield(task, 'ny') && ~isempty(task.ny)
		task.arch.ny = task.ny;
	end

	dims = get_arch_dims(task.arch);
	maxDim = max(dims);
	termsByDim = cell(1, maxDim);
	% Always include constant and first-order variables as a safe compact base for
	% each actually used input dimension.  This is case-local, not a generated
	% polynomial/order expansion.
	for d = unique(dims)
		termsByDim{d} = [{'1'}, arrayfun(@(k) sprintf('v%d', k), 1:d, 'UniformOutput', false)];
	end

	% No weak-prior term lists are harvested here in compact-dictionary route.  The compact
	% dictionary is defined only by add_named_case_supplement_local below.

	% Strong-prior masks may reference terms not present in the weak-prior helper
	% lists.  Add a small case-specific supplement for known compact Feynman forms.
	termsByDim = add_named_case_supplement_local(termsByDim, task);

	for d = 1:numel(termsByDim)
		if ~isempty(termsByDim{d})
			termsByDim{d} = unique_stable_terms_local(termsByDim{d});
		end
	end

	task.arch.dictionaryMode = 'case_specific_explicit';
	task.arch.caseDictionary = struct();
	task.arch.caseDictionary.caseId = task.name;
	task.arch.caseDictionary.termsByDim = termsByDim;
	task.arch.caseDictionary.source = 'compact-dictionary route task-defined uniform compact prior dictionary';
end

function task = apply_general_low_order_dictionary(task)
%APPLY_GENERAL_LOW_ORDER_DICTIONARY Build the generated dictionary for general mode.
%
% The default general dictionary is a total-degree dictionary with order p=2.
% The zero-order term is handled separately as the special constant basis '1'.
% Positive-degree polynomial multi-indices are generated by a general
% total-degree routine.  If useChebyshev is true, each polynomial multi-index
% is represented by the corresponding product of first-kind Chebyshev terms,
% for example v1^2*v2 -> T2(v1)*T1(v2).  T1(v) is kept explicitly in term
% names for a uniform Chebyshev dictionary format, even though T1(v)=v.

	if ~isfield(task, 'arch') || ~isfield(task.arch, 'layer')
		return;
	end
	if isfield(task, 'nx') && ~isempty(task.nx)
		task.arch.nx = task.nx;
	end
	if isfield(task, 'ny') && ~isempty(task.ny)
		task.arch.ny = task.ny;
	end
	if ~isfield(task.arch, 'generalDictionary') || isempty(task.arch.generalDictionary)
		task.arch.generalDictionary = struct();
	end
	gd = task.arch.generalDictionary;
	if ~isfield(gd, 'polyOrder') || isempty(gd.polyOrder)
		gd.polyOrder = 2;
	end
	if ~isfield(gd, 'useChebyshev') || isempty(gd.useChebyshev)
		gd.useChebyshev = true;
	end
	if ~isfield(gd, 'includeSinCos') || isempty(gd.includeSinCos)
		gd.includeSinCos = false;
	end
	gd = fill_general_prior_basis_defaults_local(gd);

	dims = get_arch_dims(task.arch);
	maxDim = max(dims);
	termsByDim = cell(1, maxDim);
	for d = unique(dims)
		termsByDim{d} = build_general_total_degree_terms_local(d, gd.polyOrder, gd.useChebyshev, gd.includeSinCos);
	end
	[termsByDim, priorReport] = append_general_prior_basis_terms_local(termsByDim, dims, task, gd);

	task.arch.dictionaryMode = 'general_explicit';
	task.arch.generalDictionary = gd;
	task.arch.caseDictionary = struct();
	task.arch.caseDictionary.caseId = task.name;
	task.arch.caseDictionary.termsByDim = termsByDim;
	if gd.useChebyshev
		sourceName = sprintf('general dictionary route general total-degree Chebyshev dictionary, order %d', gd.polyOrder);
	else
		sourceName = sprintf('general dictionary route general total-degree polynomial dictionary, order %d', gd.polyOrder);
	end
	if gd.includeSinCos
		sourceName = [sourceName ', with sin/cos variable terms'];
	end
	if priorReport.enabled
		sourceName = sprintf('%s, priorBasis unary=%d custom=%d', sourceName, priorReport.nUnaryTermsAdded, priorReport.nCustomTermsAdded);
	end
	task.arch.caseDictionary.source = sourceName;
	% Keep prior-basis report fields out of no-prior runs so the console report
	% remains clean when priorBasis is disabled.
	if priorReport.enabled
		task.arch.caseDictionary.priorBasisReport = priorReport;
	end
end

function task = apply_general_mode_defaults(task)
%APPLY_GENERAL_MODE_DEFAULTS Set the general-mode PhDN architecture and dictionary controls.
%
% General mode is not a strong/weak-prior structure-recovery mode.  It must
% therefore define its own architecture explicitly instead of borrowing the
% case-specific strong-prior hidden dimensions.  The defaults are intentionally
% simple:
%   layer      = 4
%   hiddenDim  = input dimension nx for each hidden layer
%   dictionary = total-degree polynomial/Chebyshev order 2
%
% If a subcase sets task.arch.generalLayer, task.arch.generalHiddenDims, or
% task.arch.generalDictionary.* before calling this helper, those values are
% preserved.
	if ~isfield(task, 'arch') || isempty(task.arch)
		task.arch = struct();
	end

	% Explicit general-mode architecture.  Do not inherit the prior-case
	% architecture such as [2,2] or [1,1,1], because those dimensions were
	% designed for specific symbolic decompositions rather than general fitting.
	if ~isfield(task.arch, 'generalLayer') || isempty(task.arch.generalLayer)
		task.arch.generalLayer = 4;
	end
	if ~isfield(task.arch, 'generalHiddenDims') || isempty(task.arch.generalHiddenDims)
		defaultHiddenWidth = max(task.nx, task.ny);
		task.arch.generalHiddenDims = repmat(defaultHiddenWidth, 2, max(task.arch.generalLayer - 1, 0));
	end
	task.arch.layer = task.arch.generalLayer;
	task.arch.hiddenDims = task.arch.generalHiddenDims;

	if ~isfield(task.arch, 'generalDictionary') || isempty(task.arch.generalDictionary)
		task.arch.generalDictionary = struct();
	end
	if ~isfield(task.arch.generalDictionary, 'polyOrder') || isempty(task.arch.generalDictionary.polyOrder)
		task.arch.generalDictionary.polyOrder = 2;
	end
	if ~isfield(task.arch.generalDictionary, 'useChebyshev') || isempty(task.arch.generalDictionary.useChebyshev)
		task.arch.generalDictionary.useChebyshev = true;
	end
	if ~isfield(task.arch.generalDictionary, 'includeSinCos') || isempty(task.arch.generalDictionary.includeSinCos)
		task.arch.generalDictionary.includeSinCos = false;
	end
	task.arch.generalDictionary = fill_general_prior_basis_defaults_local(task.arch.generalDictionary);
	task.arch.dictionaryMode = 'general_explicit';
	task.arch.polyOrder = task.arch.generalDictionary.polyOrder;
	task.prior.dictionaryMode = 'general_low_order_dictionary';
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';
	task.DisplaySymbolic = false;
end


function gd = fill_general_prior_basis_defaults_local(gd)
%FILL_GENERAL_PRIOR_BASIS_DEFAULTS_LOCAL Default optional prior-basis controls.
%
% These controls are used only by general mode.  They extend the generated
% polynomial/Chebyshev dictionary without reviving the old arch.opNames path.
%
% Example:
%   gd.priorBasis.enable = true;
%   gd.priorBasis.unaryOps = {'inv'};
%   gd.priorBasis.unaryApplyTo = 'hidden_only';  % 'all' | 'raw_only' | 'hidden_only' | 'dims'
%   gd.priorBasis.unaryDims = [1 2];             % used only for 'dims'
%   gd.priorBasis.customTerms = {'v1*inv(v2)'};  % applied to every compatible dimension
%   gd.priorBasis.customTermsByDim{2} = {'v1*inv(v2)', '(v1-v2)^2'};
	if ~isfield(gd, 'priorBasis') || isempty(gd.priorBasis)
		gd.priorBasis = struct();
	end
	pb = gd.priorBasis;
	if ~isfield(pb, 'enable') || isempty(pb.enable)
		pb.enable = false;
	end
	if ~isfield(pb, 'unaryOps') || isempty(pb.unaryOps)
		pb.unaryOps = {};
	end
	if ~isfield(pb, 'unaryApplyTo') || isempty(pb.unaryApplyTo)
		pb.unaryApplyTo = 'all';
	end
	if ~isfield(pb, 'unaryDims') || isempty(pb.unaryDims)
		pb.unaryDims = [];
	end
	if ~isfield(pb, 'customTerms') || isempty(pb.customTerms)
		pb.customTerms = {};
	end
	if ~isfield(pb, 'customTermsByDim') || isempty(pb.customTermsByDim)
		pb.customTermsByDim = {};
	end
	gd.priorBasis = pb;
end

function [termsByDim, report] = append_general_prior_basis_terms_local(termsByDim, dims, task, gd)
%APPEND_GENERAL_PRIOR_BASIS_TERMS_LOCAL Add optional unary/custom prior basis.
%
% The base dictionary is still the generated total-degree polynomial/Chebyshev
% dictionary.  priorBasis only appends extra explicit terms for general mode.
	report = struct();
	report.enabled = false;
	report.applyTo = 'none';
	report.nUnaryTermsAdded = 0;
	report.nCustomTermsAdded = 0;
	report.unaryOps = {};
	report.unaryDims = [];
	report.customDims = [];

	if ~isfield(gd, 'priorBasis') || isempty(gd.priorBasis)
		return;
	end
	pb = gd.priorBasis;
	unaryOps = normalize_terms_cell_local(get_struct_field_local(pb, 'unaryOps', {}));
	customTerms = normalize_terms_cell_local(get_struct_field_local(pb, 'customTerms', {}));
	customTermsByDim = get_struct_field_local(pb, 'customTermsByDim', {{}});
	hasCustomByDim = iscell(customTermsByDim) && any(cellfun(@(c) ~isempty(c), customTermsByDim(:)));
	enabled = logical(get_struct_field_local(pb, 'enable', false)) || ~isempty(unaryOps) || ~isempty(customTerms) || hasCustomByDim;
	if ~enabled
		return;
	end

	report.enabled = true;
	applyTo = lower(strtrim(char(get_struct_field_local(pb, 'unaryApplyTo', 'all'))));
	report.applyTo = applyTo;
	report.unaryOps = unaryOps(:).';

	unaryDims = select_general_prior_unary_dims_local(termsByDim, dims, task, pb, applyTo);
	report.unaryDims = unaryDims;
	for id = 1:numel(unaryDims)
		d = unaryDims(id);
		newTerms = build_unary_prior_terms_local(d, unaryOps);
		nBefore = numel(termsByDim{d});
		termsByDim = append_terms_for_dim_local(termsByDim, d, newTerms);
		report.nUnaryTermsAdded = report.nUnaryTermsAdded + max(0, numel(termsByDim{d}) - nBefore);
	end

	activeDims = find(~cellfun(@isempty, termsByDim));
	for id = 1:numel(activeDims)
		d = activeDims(id);
		newTerms = filter_terms_compatible_with_dim_local(customTerms, d);
		if iscell(customTermsByDim) && numel(customTermsByDim) >= d && ~isempty(customTermsByDim{d})
			newTerms = [newTerms(:); filter_terms_compatible_with_dim_local(customTermsByDim{d}, d)]; %#ok<AGROW>
		end
		newTerms = unique_stable_terms_local(newTerms);
		if isempty(newTerms)
			continue;
		end
		nBefore = numel(termsByDim{d});
		termsByDim = append_terms_for_dim_local(termsByDim, d, newTerms);
		nAdded = max(0, numel(termsByDim{d}) - nBefore);
		if nAdded > 0
			report.customDims(end+1) = d; %#ok<AGROW>
			report.nCustomTermsAdded = report.nCustomTermsAdded + nAdded;
		end
	end
end

function dimsOut = select_general_prior_unary_dims_local(termsByDim, dims, task, pb, applyTo)
	activeDims = find(~cellfun(@isempty, termsByDim));
	if isempty(activeDims)
		dimsOut = [];
		return;
	end
	switch applyTo
		case 'all'
			dimsOut = activeDims;
		case 'raw_only'
			dimsOut = intersect(activeDims, task.nx, 'stable');
		case 'hidden_only'
			if numel(dims) >= 3
				hiddenDims = unique(dims(2:end-1), 'stable');
			else
				hiddenDims = [];
			end
			dimsOut = intersect(activeDims, hiddenDims, 'stable');
		case 'dims'
			requested = get_struct_field_local(pb, 'unaryDims', []);
			requested = unique(round(requested(:).'), 'stable');
			requested = requested(isfinite(requested) & requested >= 1);
			dimsOut = intersect(activeDims, requested, 'stable');
		otherwise
			error('Unknown general priorBasis.unaryApplyTo: %s. Use all, raw_only, hidden_only, or dims.', applyTo);
	end
end

function terms = build_unary_prior_terms_local(inputDim, unaryOps)
	terms = {};
	unaryOps = normalize_terms_cell_local(unaryOps);
	for i = 1:numel(unaryOps)
		op = strtrim(char(unaryOps{i}));
		if isempty(op)
			continue;
		end
		for k = 1:inputDim
			terms{end+1,1} = sprintf('%s(v%d)', op, k); %#ok<AGROW>
		end
	end
	terms = unique_stable_terms_local(terms);
end

function terms = filter_terms_compatible_with_dim_local(terms, inputDim)
	terms = normalize_terms_cell_local(terms);
	out = {};
	for i = 1:numel(terms)
		term = canonicalize_xh_to_v_local(terms{i});
		m = regexp(term, 'v(\d+)', 'tokens');
		if isempty(m)
			maxVar = 0;
		else
			maxVar = max(cellfun(@(z) str2double(z{1}), m));
		end
		if maxVar <= inputDim
			out{end+1,1} = term; %#ok<AGROW>
		end
	end
	terms = unique_stable_terms_local(out);
end

function val = get_struct_field_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

function terms = build_general_total_degree_terms_local(inputDim, polyOrder, useChebyshev, includeSinCos)
	if nargin < 4 || isempty(includeSinCos)
		includeSinCos = false;
	end
	polyOrder = max(0, round(polyOrder));

	% Treat the zero-order constant as a special identity basis.  The positive
	% polynomial/Chebyshev part is generated from all nonzero total-degree
	% multi-indices with degree <= polyOrder.
	terms = {'1'};
	alphas = generate_positive_total_degree_multi_indices_local(inputDim, polyOrder);
	for r = 1:size(alphas, 1)
		terms{end+1,1} = multi_index_to_term_local(alphas(r, :), useChebyshev); %#ok<AGROW>
	end
	if includeSinCos
		for k = 1:inputDim
			terms{end+1,1} = sprintf('sin(v%d)', k); %#ok<AGROW>
			terms{end+1,1} = sprintf('cos(v%d)', k); %#ok<AGROW>
		end
	end
	terms = unique_stable_terms_local(terms);
end

function alphas = generate_positive_total_degree_multi_indices_local(inputDim, maxOrder)
	rows = zeros(0, inputDim);
	for total = 1:maxOrder
		rows = [rows; generate_fixed_degree_multi_indices_local(inputDim, total)]; %#ok<AGROW>
	end
	alphas = rows;
end

function rows = generate_fixed_degree_multi_indices_local(inputDim, totalDegree)
	if inputDim == 1
		rows = totalDegree;
		return;
	end
	rows = zeros(0, inputDim);
	for a = totalDegree:-1:0
		tail = generate_fixed_degree_multi_indices_local(inputDim - 1, totalDegree - a);
		rows = [rows; [a * ones(size(tail, 1), 1), tail]]; %#ok<AGROW>
	end
end

function term = multi_index_to_term_local(alpha, useChebyshev)
	if all(alpha == 0)
		term = '1';
		return;
	end
	parts = {};
	for k = 1:numel(alpha)
		a = alpha(k);
		if a == 0
			continue;
		elseif useChebyshev
			parts{end+1} = sprintf('T%d(v%d)', a, k); %#ok<AGROW>
		elseif a == 1
			parts{end+1} = sprintf('v%d', k); %#ok<AGROW>
		else
			parts{end+1} = sprintf('v%d^%d', k, a); %#ok<AGROW>
		end
	end
	term = strjoin(parts, '*');
end

function termsByDim = append_terms_for_dim_local(termsByDim, d, terms)
	if d < 1 || isempty(terms); return; end
	if numel(termsByDim) < d; termsByDim{d} = {}; end
	termsByDim{d} = unique_stable_terms_local([termsByDim{d}(:); terms(:)]);
end

function terms = normalize_terms_cell_local(terms)
	if ischar(terms) || isstring(terms)
		terms = cellstr(terms);
	end
	terms = terms(:);
	out = {};
	for i = 1:numel(terms)
		if iscell(terms{i})
			out = [out; normalize_terms_cell_local(terms{i})]; %#ok<AGROW>
		else
			name = strrep(strtrim(char(terms{i})), ' ', '');
			if ~isempty(name)
				out{end+1,1} = name; %#ok<AGROW>
			end
		end
	end
	terms = unique_stable_terms_local(out);
end

function terms = unique_stable_terms_local(terms)
	out = {};
	for i = 1:numel(terms)
		name = char(terms{i});
		if ~any(strcmp(out, name))
			out{end+1,1} = name; %#ok<AGROW>
		end
	end
	terms = out(:);
end

function termsByDim = add_named_case_supplement_local(termsByDim, task)
	caseId = '';
	if isfield(task, 'name')
		caseId = regexprep(task.name, '^KAN_Feynman_', '');
	end
	switch caseId
		case 'I_6_2'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'exp(h1)', 'h1*h2', 'inv(h2)', 'sqrt(x2^2)', 'x1^2*inv(x2^2)'});
		case 'I_6_2b'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'exp(h1)', 'h1*h2', 'inv(h2)', 'sqrt(x3^2)', 'x1*x2*inv(x3^2)', 'x1^2*inv(x3^2)', 'x2^2*inv(x3^2)'});
		case 'I_9_18'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'h1*inv(h2)', 'h1^2', 'h2^2', 'h3^2', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6'});
		case 'I_12_11'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'x1*sin(x2)'});
		case 'I_13_12'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1', 'x1*inv(x2)'});
		case 'I_15_3x'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'h1*h2', 'h2', 'inv(h1)', 'sqrt(h1)', 'x1', 'x2^2'});
		case 'I_16_6'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'h1', 'h1*h2', 'inv(h2)', 'x1', 'x1*x2', 'x2'});
		case 'I_18_4'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'h1', 'h1*h2', 'inv(h2)', 'x1', 'x1*x2'});
		case 'I_26_2'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'asin(h1)', 'x1*sin(x2)'});
		case 'I_27_6'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'inv(h1)', 'x1*x2'});
		case 'I_29_16'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'cos(h3)', 'h1', 'h1*h3', 'h2', 'sqrt(h1)', 'x1', 'x1^2', 'x2', 'x3'});
		case 'I_30_3'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'h1*h2', 'h1^2', 'inv(h2^2)', 'sin(h1)', 'sin(h2)', 'x1*x2', 'x2'});
		case 'I_30_5'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'asin(h1)', 'x1*inv(x2)'});
		case 'I_37_4'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'cos(x2)', 'h1', 'h1*h2', 'h2', 'h3', 'sqrt(x1)', 'x1'});
		case 'I_40_1'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'exp(h2)', 'h1', 'h1*h2', 'x1', 'x2'});
		case 'I_44_4'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1*log(x2)'});
		case 'I_50_26'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'cos(x1)', 'h1', 'h1^2', 'h2', 'h2*h3', 'x2'});
		case 'II_2_42'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1*x2', 'x2'});
		case 'II_6_15a'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'h1*h2', 'h2', 'sqrt(h1)', 'x1^2', 'x2^2', 'x3'});
		case 'II_11_7'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1', 'x1*x2*cos(x3)'});
		case 'II_11_27'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'h1', 'h1*h2', 'inv(h2)', 'x1*x2'});
		case 'II_35_18'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'exp(h2)', 'exp(h3)', 'h1', 'h1*h2', 'h2', 'h3', 'inv(h2)', 'x1', 'x2'});
		case 'II_36_38'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1', 'x2*x3'});
		case 'II_38_3'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1*inv(x2)'});
		case 'III_9_52'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'h1', 'h1*h2', 'h2*h3', 'h2^2', 'inv(h2)', 'sin(h2)', 'x1', 'x2', 'x3'});
		case 'III_10_19'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'1', 'sqrt(h1)', 'x1^2', 'x2^2'});
		case 'III_17_37'
			termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, {'x1*x2*cos(x3)', 'x2'});
		otherwise
			% No additional static supplement.  Fallback remains the compact case lists.
	end
end

function termsByDim = add_case_terms_to_compatible_dims_local(termsByDim, terms)
	terms = normalize_terms_cell_local(terms);
	for i = 1:numel(terms)
		term = canonicalize_xh_to_v_local(terms{i});
		m = regexp(term, 'v(\d+)', 'tokens');
		if isempty(m)
			minDim = 1;
		else
			idx = cellfun(@(z) str2double(z{1}), m);
			minDim = max(idx);
		end
		for d = minDim:numel(termsByDim)
			if ~isempty(termsByDim{d})
				termsByDim = append_terms_for_dim_local(termsByDim, d, {term});
			end
		end
	end
end

function term = canonicalize_xh_to_v_local(term)
	term = strrep(strtrim(char(term)), ' ', '');
	m = regexp(term, '[xh](\d+)', 'tokens');
	if isempty(m); return; end
	idx = cellfun(@(z) str2double(z{1}), m);
	for k = max(idx):-1:1
		term = regexprep(term, sprintf('(?<![A-Za-z0-9_])[xh]%d(?![A-Za-z0-9_])', k), sprintf('v%d', k));
	end
end

% Legacy weak-prior Phi-prior helper functions were removed in compact-mask route.


function task = init_case(task, caseId, description, varNames, domainSpec, rhsFcn, symFcn, ops, polyOrder, layer)
	task.name = ['KAN_Feynman_' caseId];
	task.description = description;
	task.nx = numel(varNames);
	task.ny = 1;
	task.variableNames = varNames;
	task.outputNames = {'y'};
	task.domain = make_variable_union_domain(varNames, domainSpec);
	task.rhsFcn = rhsFcn;
	task.referenceSymbolicFcn = symFcn;

	task.arch = struct();
	task.arch.layer = layer;
	task.arch.hiddenDims = repmat(task.ny, 1, max(layer - 1, 0));
	task.arch.dictionaryMode = 'case_specific_explicit';
	% Legacy scalar only: explicit case dictionaries do not generate terms from
	% polynomial/operator/interact options.  Keep this field only for old
	% reporting/fallback code that expects arch.polyOrder to exist.
	task.arch.polyOrder = 1;

	% Operator mode is now only a safety/reporting label for Feynman tasks.
	% The actual branch dictionary is the explicit compact case dictionary.
	task.operatorControl = struct();
	if any(strcmpi(task.casemode, {'strong_prior','weak_strong_prior'})) || layer == 1
		task.operatorControl.caseDefault = 'true';
	else
		task.operatorControl.caseDefault = 'surrogate';
	end
	task.operatorControl.demoOverride = 'task_default';
	task.operatorControl.singleLayerForceTrue = (layer == 1);
	task.operatorMode = task.operatorControl.caseDefault;
	task.arch.operatorMode = task.operatorMode;

	task.training = struct();
	task.training.operatorMode = task.operatorMode;
	task.training.lambda1List = [1e-6];
	task.training.tauList = [0];
	task.training.opArgPolyOrderList = 1;

	task.prior = struct();
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.DisplaySymbolic = true;
end

% -------------------------------------------------------------------------
% Case definitions from the provided dimensionless formula table
function task = setup_I_6_2(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_6_2', ...
			'Dimensionless Gaussian density: exp(-theta^2/(2*sigma^2))/sqrt(2*pi*sigma^2).', ...
			{'theta','sigma'}, {[-3, 3], [0.5, 2.5]}, ...
			@rhs_I_6_2, @sym_I_6_2, ...
			{'exp','inv'}, 2, 8);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_6_2 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1^2*inv(v2^2)','sqrt(v2^2)','exp(v1)','inv(v2)','v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1^2*inv(v2^2)', 'sqrt(v2^2)'};
		D.termsByLayer{2} = {'exp(v1)', 'inv(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2*inv(v2^2)', 'sqrt(v2^2)'};
		D.termsByBlock{1,2} = {'exp(v1)', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2*inv(v2^2)', 'sqrt(v2^2)'};
		D.rowTerms{1,1} = {{'v1^2*inv(v2^2)'}, {'sqrt(v2^2)'}};
		D.termsByBlock{1,2} = {'exp(v1)', 'inv(v2)'};
		D.rowTerms{1,2} = {{'exp(v1)'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_6_2b(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_6_2b', ...
			'Dimensionless shifted Gaussian density: exp(-(theta-theta1)^2/(2*sigma^2))/sqrt(2*pi*sigma^2).', ...
			{'theta','theta1','sigma'}, {[-3, 3], [-3, 3], [0.5, 2.5]}, @rhs_I_6_2b, @sym_I_6_2b, {'exp','invsqrt'}, 2, 3);
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[3.3, 4.8], [-4.8, -3.3], [2.6, 3.1]});
	task.oodDomain.source = 'task_effective_difference_ood';
	task.oodDomain.effectiveVariable = '[theta-theta1, sigma]';
	task.oodDomain.note = 'Use disjoint upper/lower angle intervals so theta-theta1 is strictly [6.6,9.6], outside the ID difference [-6,6]; sigma is also outside its ID interval and remains positive.';

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_6_2b casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1^2*inv(v3^2)','v1*v2*inv(v3^2)','v2^2*inv(v3^2)','sqrt(v3^2)','exp(v1)','inv(v2)', ...
			'v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1^2*inv(v3^2)', 'v1*v2*inv(v3^2)', 'v2^2*inv(v3^2)', 'sqrt(v3^2)'};
		D.termsByLayer{2} = {'exp(v1)', 'inv(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2*inv(v3^2)', 'v1*v2*inv(v3^2)', 'v2^2*inv(v3^2)', 'sqrt(v3^2)'};
		D.termsByBlock{1,2} = {'exp(v1)', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2*inv(v3^2)', 'v1*v2*inv(v3^2)', 'v2^2*inv(v3^2)', 'sqrt(v3^2)'};
		D.rowTerms{1,1} = {{'v1^2*inv(v3^2)', 'v1*v2*inv(v3^2)', 'v2^2*inv(v3^2)'}, {'sqrt(v3^2)'}};
		D.termsByBlock{1,2} = {'exp(v1)', 'inv(v2)'};
		D.rowTerms{1,2} = {{'exp(v1)'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_9_18(task)
	% Dimensionless formula:
	%   y = x1 / ((x2 - 1)^2 + (x3 - x4)^2 + (x5 - x6)^2)
	%
	% unified prior levels for the two-stage MLP-surrogate + exact-recovery route:
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_9_18', ...
		'Dimensionless inverse-square distance form: a/((b-1)^2 + (c-d)^2 + (e-f)^2).', ...
		{'a','b','c','d','e','f'}, ...
		{[0.25, 3], [1.3, 4], [-2, 2], [-2, 4], [-2, 2], [-2, 4]}, ...
		@rhs_I_9_18, @sym_I_9_18, ...
		{'inv'}, 2, 3);
	% OOD challenge approaches the inverse-square singular surface while retaining
	% a strict finite safety margin: denominator >= 0.055.
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[0.25, 3], [1.10, 1.25], [1.05, 1.25], [1.40, 1.55], [1.05, 1.25], [1.40, 1.55]});
	task.oodDomain.source = 'task_effective_difference_ood';
	task.oodDomain.effectiveVariable = '[(b-1),(c-d),(e-f)]';
	task.oodDomain.minimumDenominator = 0.055;

	% Common symbolic decomposition used to define the prior levels:
	%   h2 = [x2 - 1; x3 - x4; x5 - x6]
	%   h3 = [x1; h2_1^2 + h2_2^2 + h2_3^2]
	%   y  = h3_1 * inv(h3_2)
	%
	% Active branches under PhDN skip-branch indexing:
	%   A{1,1}: x  -> h2
	%   A{1,2}: h2 -> h3
	%   A{2,2}: x  -> h3
	%   A{1,3}: h3 -> y
	% Inactive/pruned branches:
	%   A{2,3}: h2 -> y
	%   A{3,3}: x  -> y

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_9_18 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container for I_9_18.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [3, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior.  true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(2,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings can vary by prior level.  Strong row-wise prior is less
	% dependent on Stage I, so it uses a deliberately cheap surrogate.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	% Stage-I uses both the direct MLP prediction residual and the fixed-LS
	% dictionary-PhDN compatibility residual.  The compatibility residual fixes
	% the LS coefficient blocks inside each Jacobian evaluation and only
	% differentiates through the MLP hidden states / MLP parameters.
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	if priorLevel == 4
		task.mlpSurrogate.mlpHiddenDims = [8];
		task.mlpSurrogate.surrogateNumStarts = 6;
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
		task.mlpSurrogate.useParallelStarts = true;
		task.mlpSurrogate.parallelAutoStartPool = true;
	elseif priorLevel == 3
		task.mlpSurrogate.mlpHiddenDims = [8];
		task.mlpSurrogate.surrogateNumStarts = 6;
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
		task.mlpSurrogate.useParallelStarts = true;
		task.mlpSurrogate.parallelAutoStartPool = true;
	else
		task.mlpSurrogate.mlpHiddenDims = [8];
		task.mlpSurrogate.surrogateNumStarts = 6;
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
		task.mlpSurrogate.useParallelStarts = true;
		task.mlpSurrogate.parallelAutoStartPool = true;
	end

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v1','v2','v3','v4','v5','v6','v1^2','v2^2','v3^2','v1*inv(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1','v2','v3','v4','v5','v6'};
		D.termsByLayer{2} = {'v1','v1^2','v2^2','v3^2'};
		D.termsByLayer{3} = {'v1*inv(v2)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1','v2','v3','v4','v5','v6'};
		D.termsByBlock{1,2} = {'v1^2','v2^2','v3^2'};
		D.termsByBlock{2,2} = {'v1'};
		D.termsByBlock{1,3} = {'v1*inv(v2)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside each branch block.  The branch dictionary
		% is the row-wise union; rowTerms then imposes the strong-prior mask.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1','v2','v3','v4','v5','v6'};
		D.rowTerms{1,1} = {{'1','v2'}, {'v3','v4'}, {'v5','v6'}};
		D.termsByBlock{1,2} = {'v1^2','v2^2','v3^2'};
		D.rowTerms{1,2} = {{}, {'v1^2','v2^2','v3^2'}};
		D.termsByBlock{2,2} = {'v1'};
		D.rowTerms{2,2} = {{'v1'}, {}};
		D.termsByBlock{1,3} = {'v1*inv(v2)'};
		D.rowTerms{1,3} = {{'v1*inv(v2)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end



function [branchActive, modeApplied] = apply_branch_active_option_local(task, branchActivePrior)
%APPLY_BRANCH_ACTIVE_OPTION_LOCAL Select case-specific or full branch activity.
%
% This is a single task-file switch used by every setup_* function.
% It does not change the prior-level dictionary itself; it only controls
% whether the PhDN branch graph follows the interpretable case prior or uses
% the fully active L-by-L branch graph requested for general branch testing.
	modeApplied = 'prior';
	if isfield(task, 'branchActiveMode') && ~isempty(task.branchActiveMode)
		modeApplied = lower(strtrim(char(task.branchActiveMode)));
	end

	switch modeApplied
		case {'prior','case','case_specific','structured','interpretable'}
			modeApplied = 'prior';
			branchActive = logical(branchActivePrior);
		case {'full','all','general','dense','fully_active'}
			modeApplied = 'full';
			branchActive = true(task.arch.layer, task.arch.layer);
		otherwise
			error('Unknown task.branchActiveMode: %s. Use ''prior'' or ''full''.', modeApplied);
	end
end


function C = make_empty_block_terms(L)
	C = cell(L, L);
	for ell = 1:L
		for src = 1:ell
			C{src, ell} = {};
		end
	end
end

function termsByDim = build_terms_by_dim_from_all_terms(task, allTerms)
	dims = get_arch_dims(task.arch);
	maxDim = max(dims);
	termsByDim = cell(1, maxDim);
	for d = unique(dims)
		termsByDim{d} = filter_terms_compatible_with_dim_local(allTerms, d);
	end
end

function task = setup_I_12_11(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_12_11', ...
			'Dimensionless Lorentz-like formula: 1 + a*sin(theta).', ...
			{'a','theta'}, {[-3, 3], [-pi, pi]}, ...
			@rhs_I_12_11, @sym_I_12_11, ...
			{'sin'}, 1, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_12_11 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v1*sin(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1', 'v1*sin(v2)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*sin(v2)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*sin(v2)'};
		D.rowTerms{1,1} = {{'1', 'v1*sin(v2)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_13_12(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Structural-identifiability domain: sample both signs of the reciprocal
	% argument while excluding a finite neighborhood of the pole b=0.
	task = init_case(task, 'I_13_12', ...
			'Dimensionless potential difference: a*(1/b - 1).', ...
			{'a','b'}, {[-3, 3], [-3, -0.4; 0.4, 3]}, ...
			@rhs_I_13_12, @sym_I_13_12, ...
			{'inv'}, 1, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_13_12 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*inv(v2)','v1'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*inv(v2)', 'v1'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)', 'v1'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)', 'v1'};
		D.rowTerms{1,1} = {{'v1*inv(v2)', 'v1'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_15_3x(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_15_3x', ...
			'Dimensionless relativistic form: (1-a)/sqrt(1-b^2).', ...
			{'a','b'}, {[-3, 3], [-0.75, 0.75]}, ...
			@rhs_I_15_3x, @sym_I_15_3x, ...
			{'invsqrt'}, 2, 4);
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[3.3, 4.8], [-0.90, -0.80; 0.80, 0.90]});
	task.oodDomain.source = 'task_safe_boundary_ood';
	task.oodDomain.effectiveVariable = 'b near the |b|=1 boundary';
	task.oodDomain.minimumSqrtArgument = 0.19;

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_15_3x casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 4;
	task.arch.hiddenDims = [2, 2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v2^2','v1','sqrt(v1)','v2','inv(v1)', ...
			'v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1', 'v2^2', 'v1'};
		D.termsByLayer{2} = {'sqrt(v1)', 'v2'};
		D.termsByLayer{3} = {'inv(v1)', 'v2'};
		D.termsByLayer{4} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v2^2', 'v1'};
		D.termsByBlock{1,2} = {'sqrt(v1)', 'v2'};
		D.termsByBlock{1,3} = {'inv(v1)', 'v2'};
		D.termsByBlock{1,4} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v2^2', 'v1'};
		D.rowTerms{1,1} = {{'1', 'v2^2'}, {'1', 'v1'}};
		D.termsByBlock{1,2} = {'sqrt(v1)', 'v2'};
		D.rowTerms{1,2} = {{'sqrt(v1)'}, {'v2'}};
		D.termsByBlock{1,3} = {'inv(v1)', 'v2'};
		D.rowTerms{1,3} = {{'inv(v1)'}, {'v2'}};
		D.termsByBlock{1,4} = {'v1*v2'};
		D.rowTerms{1,4} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_16_6(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Use the physically natural signed subluminal range.  Both variables cross
	% zero while 1+a*b remains bounded below by 0.36.
	task = init_case(task, 'I_16_6', ...
			'Dimensionless velocity addition: (a+b)/(1+a*b).', ...
			{'a','b'}, {[-0.8, 0.8], [-0.8, 0.8]}, ...
			@rhs_I_16_6, @sym_I_16_6, ...
			{'inv'}, 2, 4);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_16_6 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v2','1','v1*v2','inv(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v2', '1', 'v1*v2'};
		D.termsByLayer{2} = {'v1', 'inv(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2', '1', 'v1*v2'};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2', '1', 'v1*v2'};
		D.rowTerms{1,1} = {{'v1', 'v2'}, {'1', 'v1*v2'}};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.rowTerms{1,2} = {{'v1'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_18_4(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Let a cross zero to distinguish numerator and denominator roles, but keep
	% a safely above the pole a=-1 so 1+a >= 0.25.
	task = init_case(task, 'I_18_4', ...
			'Dimensionless rational expression: (1+a*b)/(1+a).', ...
			{'a','b'}, {[-0.75, 4], [-3, 3]}, ...
			@rhs_I_18_4, @sym_I_18_4, ...
			{'inv'}, 2, 4);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_18_4 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v1*v2','v1','inv(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1', 'v1*v2', 'v1'};
		D.termsByLayer{2} = {'v1', 'inv(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*v2', 'v1'};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*v2', 'v1'};
		D.rowTerms{1,1} = {{'1', 'v1*v2'}, {'1', 'v1'}};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.rowTerms{1,2} = {{'v1'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_26_2(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_26_2', ...
			'Dimensionless Snell formula: asin(n*sin(theta2)).', ...
			{'n','theta2'}, {[-0.55, 0.55], [-pi, pi]}, ...
			@rhs_I_26_2, @sym_I_26_2, ...
			{'sin','asin'}, 1, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_26_2 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 2;
	task.arch.hiddenDims = [1];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*sin(v2)','asin(v1)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*sin(v2)'};
		D.termsByLayer{2} = {'asin(v1)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*sin(v2)'};
		D.termsByBlock{1,2} = {'asin(v1)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*sin(v2)'};
		D.rowTerms{1,1} = {{'v1*sin(v2)'}};
		D.termsByBlock{1,2} = {'asin(v1)'};
		D.rowTerms{1,2} = {{'asin(v1)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_27_6(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Increase signed product excitation without approaching the pole: with
	% a,b in [-0.9,0.9], 1+a*b remains at least 0.19.
	task = init_case(task, 'I_27_6', ...
			'Dimensionless inverse denominator: 1/(1+a*b).', ...
			{'a','b'}, {[-0.9, 0.9], [-0.9, 0.9]}, ...
			@rhs_I_27_6, @sym_I_27_6, ...
			{'inv'}, 2, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_27_6 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 2;
	task.arch.hiddenDims = [1];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v1*v2','inv(v1)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1', 'v1*v2'};
		D.termsByLayer{2} = {'inv(v1)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*v2'};
		D.termsByBlock{1,2} = {'inv(v1)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1*v2'};
		D.rowTerms{1,1} = {{'1', 'v1*v2'}};
		D.termsByBlock{1,2} = {'inv(v1)'};
		D.rowTerms{1,2} = {{'inv(v1)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_29_16(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_29_16', ...
			'Dimensionless law of cosines: sqrt(1+a^2-2*a*cos(theta1-theta2)).', ...
			{'a','theta1','theta2'}, {[0.05, 0.9; 1.1, 3], [-pi, pi], [-pi, pi]}, ...
			@rhs_I_29_16, @sym_I_29_16, ...
			{'sqrt','cos'}, 2, 5);
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[3.15, 3.90], [-pi, pi], [-pi, pi]});
	task.oodDomain.source = 'task_effective_variable_ood';
	task.oodDomain.effectiveVariable = '[a, cos(theta1-theta2)]';
	task.oodDomain.note = 'The ID angles already cover a full period, so no nonredundant angular extrapolation exists; OOD is placed only on a outside the ID envelope while retaining full phase coverage.';

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_29_16 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 4;
	task.arch.hiddenDims = [3, 3, 1];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v1^2','v2','v3','cos(v3)','1', ...
			'v1*v3','sqrt(v1)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v1^2', 'v2', 'v3'};
		D.termsByLayer{2} = {'v1', 'v2', 'cos(v3)'};
		D.termsByLayer{3} = {'1', 'v2', 'v1*v3'};
		D.termsByLayer{4} = {'sqrt(v1)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v1^2', 'v2', 'v3'};
		D.termsByBlock{1,2} = {'v1', 'v2', 'cos(v3)'};
		D.termsByBlock{1,3} = {'1', 'v2', 'v1*v3'};
		D.termsByBlock{1,4} = {'sqrt(v1)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v1^2', 'v2', 'v3'};
		D.rowTerms{1,1} = {{'v1'}, {'v1^2'}, {'v2', 'v3'}};
		D.termsByBlock{1,2} = {'v1', 'v2', 'cos(v3)'};
		D.rowTerms{1,2} = {{'v1'}, {'v2'}, {'cos(v3)'}};
		D.termsByBlock{1,3} = {'1', 'v2', 'v1*v3'};
		D.rowTerms{1,3} = {{'1', 'v2', 'v1*v3'}};
		D.termsByBlock{1,4} = {'sqrt(v1)'};
		D.rowTerms{1,4} = {{'sqrt(v1)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_30_3(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Sample both signs of theta to expose the even symmetry of the squared-sine
	% ratio, while the union interval excludes the removable 0/0 point theta=0.
	task = init_case(task, 'I_30_3', ...
			'Dimensionless diffraction ratio: sin(n*theta/2)^2 / sin(theta/2)^2.', ...
			{'n','theta'}, {[0.5, 5], [-3.0, -0.5; 0.5, 3.0]}, ...
			@rhs_I_30_3, @sym_I_30_3, ...
			{'sin','inv'}, 2, 7);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_30_3 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 4;
	task.arch.hiddenDims = [2, 2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*v2','v2','sin(v1)','sin(v2)','v1^2','inv(v2^2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*v2', 'v2'};
		D.termsByLayer{2} = {'sin(v1)', 'sin(v2)'};
		D.termsByLayer{3} = {'v1^2', 'inv(v2^2)'};
		D.termsByLayer{4} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', 'v2'};
		D.termsByBlock{1,2} = {'sin(v1)', 'sin(v2)'};
		D.termsByBlock{1,3} = {'v1^2', 'inv(v2^2)'};
		D.termsByBlock{1,4} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', 'v2'};
		D.rowTerms{1,1} = {{'v1*v2'}, {'v2'}};
		D.termsByBlock{1,2} = {'sin(v1)', 'sin(v2)'};
		D.rowTerms{1,2} = {{'sin(v1)'}, {'sin(v2)'}};
		D.termsByBlock{1,3} = {'v1^2', 'inv(v2^2)'};
		D.rowTerms{1,3} = {{'v1^2'}, {'inv(v2^2)'}};
		D.termsByBlock{1,4} = {'v1*v2'};
		D.rowTerms{1,4} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_30_5(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_30_5', ...
			'Dimensionless arcsin formula: asin(a/n).', ...
			{'a','n'}, {[-1.2, 1.2], [1.5, 3]}, ...
			@rhs_I_30_5, @sym_I_30_5, ...
			{'inv','asin'}, 1, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_30_5 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 2;
	task.arch.hiddenDims = [1];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*inv(v2)','asin(v1)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*inv(v2)'};
		D.termsByLayer{2} = {'asin(v1)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)'};
		D.termsByBlock{1,2} = {'asin(v1)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)'};
		D.rowTerms{1,1} = {{'v1*inv(v2)'}};
		D.termsByBlock{1,2} = {'asin(v1)'};
		D.rowTerms{1,2} = {{'asin(v1)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_37_4(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_37_4', ...
			'Dimensionless interference formula: 1+a+2*sqrt(a)*cos(delta).', ...
			{'a','delta'}, {[0.1, 4], [-pi, pi]}, ...
			@rhs_I_37_4, @sym_I_37_4, ...
			{'sqrt','cos'}, 1, 3);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_37_4 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [3, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'sqrt(v1)','cos(v2)','v1','v1*v2','v3','1', ...
			'v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'sqrt(v1)', 'cos(v2)', 'v1'};
		D.termsByLayer{2} = {'v1*v2', 'v3'};
		D.termsByLayer{3} = {'1', 'v1', 'v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'sqrt(v1)', 'cos(v2)', 'v1'};
		D.termsByBlock{1,2} = {'v1*v2', 'v3'};
		D.termsByBlock{1,3} = {'1', 'v1', 'v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'sqrt(v1)', 'cos(v2)', 'v1'};
		D.rowTerms{1,1} = {{'sqrt(v1)'}, {'cos(v2)'}, {'v1'}};
		D.termsByBlock{1,2} = {'v1*v2', 'v3'};
		D.rowTerms{1,2} = {{'v1*v2'}, {'v3'}};
		D.termsByBlock{1,3} = {'1', 'v1', 'v2'};
		D.rowTerms{1,3} = {{'1', 'v1', 'v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_40_1(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_40_1', ...
			'Dimensionless exponential decay: n0*exp(-a).', ...
			{'n0','a'}, {[-3, 3], [-3, 3]}, ...
			@rhs_I_40_1, @sym_I_40_1, ...
			{'exp'}, 2, 4);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_40_1 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v2','exp(v2)','v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v2'};
		D.termsByLayer{2} = {'v1', 'exp(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2'};
		D.termsByBlock{1,2} = {'v1', 'exp(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2'};
		D.rowTerms{1,1} = {{'v1'}, {'v2'}};
		D.termsByBlock{1,2} = {'v1', 'exp(v2)'};
		D.rowTerms{1,2} = {{'v1'}, {'exp(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_44_4(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_44_4', ...
			'Dimensionless logarithmic formula: n*log(a).', ...
			{'n','a'}, {[-3, 3], [0.25, 4]}, ...
			@rhs_I_44_4, @sym_I_44_4, ...
			{'log'}, 1, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_44_4 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*log(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*log(v2)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*log(v2)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*log(v2)'};
		D.rowTerms{1,1} = {{'v1*log(v2)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_I_50_26(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'I_50_26', ...
			'Dimensionless trigonometric expression: cos(a) + alpha*cos(a)^2.', ...
			{'a','alpha'}, {[-pi, pi], [-3, 3]}, ...
			@rhs_I_50_26, @sym_I_50_26, ...
			{'cos'}, 2, 4);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported I_50_26 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 4;
	task.arch.hiddenDims = [2, 3, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'cos(v1)','v2','v1','v1^2','v2*v3'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'cos(v1)', 'v2'};
		D.termsByLayer{2} = {'v1', 'v1^2', 'v2'};
		D.termsByLayer{3} = {'v1', 'v2*v3'};
		D.termsByLayer{4} = {'v1', 'v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'cos(v1)', 'v2'};
		D.termsByBlock{1,2} = {'v1', 'v1^2', 'v2'};
		D.termsByBlock{1,3} = {'v1', 'v2*v3'};
		D.termsByBlock{1,4} = {'v1', 'v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'cos(v1)', 'v2'};
		D.rowTerms{1,1} = {{'cos(v1)'}, {'v2'}};
		D.termsByBlock{1,2} = {'v1', 'v1^2', 'v2'};
		D.rowTerms{1,2} = {{'v1'}, {'v1^2'}, {'v2'}};
		D.termsByBlock{1,3} = {'v1', 'v2*v3'};
		D.rowTerms{1,3} = {{'v1'}, {'v2*v3'}};
		D.termsByBlock{1,4} = {'v1', 'v2'};
		D.rowTerms{1,4} = {{'v1', 'v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_2_42(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_2_42', ...
			'Dimensionless linear product: (a-1)*b.', ...
			{'a','b'}, {[-4, 4], [-4, 4]}, @rhs_II_2_42, @sym_II_2_42, {}, 2, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_2_42 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*v2','v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*v2', 'v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', 'v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', 'v2'};
		D.rowTerms{1,1} = {{'v1*v2', 'v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_6_15a(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_6_15a', ...
			'Dimensionless magnetic-field magnitude: (1/(4*pi))*c*sqrt(a^2+b^2).', ...
			{'a','b','c'}, {[-4, 4], [-4, 4], [-3, 3]}, @rhs_II_6_15a, @sym_II_6_15a, {'sqrt'}, 2, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_6_15a casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1^2','v2^2','v3','sqrt(v1)','v2','v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1^2', 'v2^2', 'v3'};
		D.termsByLayer{2} = {'sqrt(v1)', 'v2'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2', 'v2^2', 'v3'};
		D.termsByBlock{1,2} = {'sqrt(v1)', 'v2'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1^2', 'v2^2', 'v3'};
		D.rowTerms{1,1} = {{'v1^2', 'v2^2'}, {'v3'}};
		D.termsByBlock{1,2} = {'sqrt(v1)', 'v2'};
		D.rowTerms{1,2} = {{'sqrt(v1)'}, {'v2'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_11_7(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_11_7', ...
			'Dimensionless dipole-like formula: n0*(1+a*cos(theta)).', ...
			{'n0','a','theta'}, {[-3, 3], [-3, 3], [-pi, pi]}, @rhs_II_11_7, @sym_II_11_7, {'cos'}, 1, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_11_7 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v1*v2*cos(v3)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v1*v2*cos(v3)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v1*v2*cos(v3)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v1*v2*cos(v3)'};
		D.rowTerms{1,1} = {{'v1', 'v1*v2*cos(v3)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_11_27(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% Broaden the signed product range so the inverse denominator has observable
	% curvature; |a*b| <= 1.96 keeps 1-a*b/3 >= 0.3467 in the ID domain.
	task = init_case(task, 'II_11_27', ...
			'Dimensionless rational formula: a*b/(1-a*b/3).', ...
			{'a','b'}, {[-1.4, 1.4], [-1.4, 1.4]}, ...
			@rhs_II_11_27, @sym_II_11_27, ...
			{'inv'}, 2, 4);
	% Explicit near-pole OOD challenge that remains finite and does not cross
	% a*b=3: here a*b is in [2.25,2.56] and the denominator is >= 0.1467.
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[1.50, 1.60], [1.50, 1.60]});
	task.oodDomain.source = 'task_safe_near_pole_ood';
	task.oodDomain.effectiveVariable = 'p=a*b';
	task.oodDomain.note = 'Challenge inverse-denominator extrapolation with p in [2.25,2.56], safely below the pole p=3.';

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_11_27 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 3;
	task.arch.hiddenDims = [2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*v2','1','v1','inv(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*v2', '1'};
		D.termsByLayer{2} = {'v1', 'inv(v2)'};
		D.termsByLayer{3} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', '1'};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*v2', '1'};
		D.rowTerms{1,1} = {{'v1*v2'}, {'1', 'v1*v2'}};
		D.termsByBlock{1,2} = {'v1', 'inv(v2)'};
		D.rowTerms{1,2} = {{'v1'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1*v2'};
		D.rowTerms{1,3} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_35_18(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_35_18', ...
			'Dimensionless symmetric exponential denominator: a/(exp(b)+exp(-b)).', ...
			{'a','b'}, {[-3.0, 3.0], [-3.0, 3.0]}, ...
			@rhs_II_35_18, @sym_II_35_18, ...
			{'exp','inv'}, 2, 8);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_35_18 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 5;
	task.arch.hiddenDims = [3, 3, 2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	branchActive(1,5) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v2','exp(v2)','exp(v3)','v3','inv(v2)', ...
			'v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v2'};
		D.termsByLayer{2} = {'v1', 'exp(v2)', 'exp(v3)'};
		D.termsByLayer{3} = {'v1', 'v2', 'v3'};
		D.termsByLayer{4} = {'v1', 'inv(v2)'};
		D.termsByLayer{5} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2'};
		D.termsByBlock{1,2} = {'v1', 'exp(v2)', 'exp(v3)'};
		D.termsByBlock{1,3} = {'v1', 'v2', 'v3'};
		D.termsByBlock{1,4} = {'v1', 'inv(v2)'};
		D.termsByBlock{1,5} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2'};
		D.rowTerms{1,1} = {{'v1'}, {'v2'}, {'v2'}};
		D.termsByBlock{1,2} = {'v1', 'exp(v2)', 'exp(v3)'};
		D.rowTerms{1,2} = {{'v1'}, {'exp(v2)'}, {'exp(v3)'}};
		D.termsByBlock{1,3} = {'v1', 'v2', 'v3'};
		D.rowTerms{1,3} = {{'v1'}, {'v2', 'v3'}};
		D.termsByBlock{1,4} = {'v1', 'inv(v2)'};
		D.rowTerms{1,4} = {{'v1'}, {'inv(v2)'}};
		D.termsByBlock{1,5} = {'v1*v2'};
		D.rowTerms{1,5} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_36_38(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_36_38', ...
			'Dimensionless affine expression: a + alpha*b.', ...
			{'a','alpha','b'}, {[-4, 4], [-4, 4], [-4, 4]}, ...
			@rhs_II_36_38, @sym_II_36_38, ...
			{}, 2, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_36_38 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v2*v3'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v2*v3'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2*v3'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2*v3'};
		D.rowTerms{1,1} = {{'v1', 'v2*v3'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_II_38_3(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'II_38_3', ...
			'Dimensionless ratio: a/b.', ...
			{'a','b'}, {[-4, 4], [-4, -0.5; 0.5, 4]}, ...
			@rhs_II_38_3, @sym_II_38_3, ...
			{'inv'}, 1, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported II_38_3 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1*inv(v2)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1*inv(v2)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1*inv(v2)'};
		D.rowTerms{1,1} = {{'v1*inv(v2)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_III_9_52(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	% The old ranges forced z=(b-c)/2 to one sign.  Use two separated b
	% intervals and a narrow centered c interval so z covers both signs while
	% retaining |z| >= 0.25 and avoiding the removable 0/0 point.
	task = init_case(task, 'III_9_52', ...
			'Dimensionless sinc-squared expression: a*sin((b-c)/2)^2/((b-c)/2)^2.', ...
			{'a','b','c'}, {[-3, 3], [-3, -1; 1, 3], [-0.5, 0.5]}, ...
			@rhs_III_9_52, @sym_III_9_52, ...
			{'sin','inv'}, 2, 7);
	% Near-zero OOD coverage on both sides of z=0, with |z| in [0.05,0.25].
	task.oodDomain = make_variable_union_domain(task.variableNames, ...
		{[-3, 3], [-0.45, -0.15; 0.15, 0.45], [-0.05, 0.05]});
	task.oodDomain.source = 'task_effective_difference_ood';
	task.oodDomain.effectiveVariable = 'z=(b-c)/2';
	task.oodDomain.note = 'Challenge the near-zero but nonzero sinc regime symmetrically with |z| in [0.05,0.25], avoiding 0/0.';

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported III_9_52 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 5;
	task.arch.hiddenDims = [2, 3, 2, 2];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	branchActive(1,3) = true;
	branchActive(1,4) = true;
	branchActive(1,5) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v1','v2','v3','sin(v2)','inv(v2)','v2*v3', ...
			'v2^2','v1*v2'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v1', 'v2', 'v3'};
		D.termsByLayer{2} = {'v1', 'sin(v2)', 'inv(v2)'};
		D.termsByLayer{3} = {'v1', 'v2*v3'};
		D.termsByLayer{4} = {'v1', 'v2^2'};
		D.termsByLayer{5} = {'v1*v2'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2', 'v3'};
		D.termsByBlock{1,2} = {'v1', 'sin(v2)', 'inv(v2)'};
		D.termsByBlock{1,3} = {'v1', 'v2*v3'};
		D.termsByBlock{1,4} = {'v1', 'v2^2'};
		D.termsByBlock{1,5} = {'v1*v2'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v1', 'v2', 'v3'};
		D.rowTerms{1,1} = {{'v1'}, {'v2', 'v3'}};
		D.termsByBlock{1,2} = {'v1', 'sin(v2)', 'inv(v2)'};
		D.rowTerms{1,2} = {{'v1'}, {'sin(v2)'}, {'inv(v2)'}};
		D.termsByBlock{1,3} = {'v1', 'v2*v3'};
		D.rowTerms{1,3} = {{'v1'}, {'v2*v3'}};
		D.termsByBlock{1,4} = {'v1', 'v2^2'};
		D.rowTerms{1,4} = {{'v1'}, {'v2^2'}};
		D.termsByBlock{1,5} = {'v1*v2'};
		D.rowTerms{1,5} = {{'v1*v2'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_III_10_19(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'III_10_19', ...
			'Dimensionless square-root norm: sqrt(1+a^2+b^2).', ...
			{'a','b'}, {[-4, 4], [-4, 4]}, ...
			@rhs_III_10_19, @sym_III_10_19, ...
			{'sqrt'}, 2, 2);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported III_10_19 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 2;
	task.arch.hiddenDims = [1];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	branchActive(1,2) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'1','v1^2','v2^2','sqrt(v1)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'1', 'v1^2', 'v2^2'};
		D.termsByLayer{2} = {'sqrt(v1)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1^2', 'v2^2'};
		D.termsByBlock{1,2} = {'sqrt(v1)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'1', 'v1^2', 'v2^2'};
		D.rowTerms{1,1} = {{'1', 'v1^2', 'v2^2'}};
		D.termsByBlock{1,2} = {'sqrt(v1)'};
		D.rowTerms{1,2} = {{'sqrt(v1)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

function task = setup_III_17_37(task)
	% v60c unified prior levels for this dimensionless Feynman subcase.
	%   general / level 0       : official PySR Stage-0 with universal SR grammar
	%   weak_prior_lv1 / level 1: official PySR Stage-0 with case-compact SR grammar
	%   weak_prior_lv2 / level 2: one compact dictionary per PhDN layer
	%   weak_prior_lv3 / level 3: one compact dictionary per active branch block
	%   strong_prior / level 4  : row-wise compact support inside every active branch block

	task = init_case(task, 'III_17_37', ...
			'Dimensionless beta expression: beta*(1+alpha*cos(theta)).', ...
			{'alpha','beta','theta'}, {[-3, 3], [-3, 3], [-pi, pi]}, ...
			@rhs_III_17_37, @sym_III_17_37, ...
			{'cos'}, 2, 1);

	mode = lower(strtrim(task.casemode));
	switch mode
		case {'general','level0','prior0'}
			priorLevel = 0;
			priorName = 'general';
		case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
			priorLevel = 1;
			priorName = 'weak_prior_lv1';
		case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
			priorLevel = 2;
			priorName = 'weak_prior_lv2';
		case {'weak_prior_lv3','weak_prior_l3','weak_strong_prior','level3','prior3'}
			priorLevel = 3;
			priorName = 'weak_prior_lv3';
		case {'strong_prior','level4','prior4'}
			priorLevel = 4;
			priorName = 'strong_prior';
		otherwise
			error('Unsupported III_17_37 casemode: %s.', task.casemode);
	end

	% All prior levels share the same interpretable container.
	task.casemode = priorName;
	task.arch.layer = 1;
	task.arch.hiddenDims = [];
	task.arch.nx = task.nx;
	task.arch.ny = task.ny;
	task.operatorMode = 'true';
	task.arch.operatorMode = 'true';
	task.training.operatorMode = 'true';

	% Branch-pruning prior. true = active branch, false = pruned branch.
	branchActive = false(task.arch.layer, task.arch.layer);
	branchActive(1,1) = true;
	[branchActive, branchActiveModeApplied] = apply_branch_active_option_local(task, branchActive);
	task.arch.branchActiveMask = branchActive;
	task.arch.branchActiveMode = branchActiveModeApplied;

	task.prior.priorInterfaceEnabled = true;
	task.prior.useTwoStageMlpRecovery = true;
	task.prior.level = priorLevel;
	task.prior.levelName = priorName;
	task.prior.dictionaryMode = sprintf('prior_level_%d_%s', priorLevel, priorName);
	task.prior.branchActiveMask = branchActive;
	task.prior.branchActiveMode = branchActiveModeApplied;
	task.prior.useAdmissibleMask = false;
	task.prior.maskType = '';
	task.prior.useWeakStrongDictionaryMask = false;
	task.prior.dictionaryMaskGranularity = 'none';

	% Stage-I MLP settings. These are used only by the mlp_recovery route.
	task.mlpSurrogate = struct();
	task.mlpSurrogate.mlpActivation = 'tanh';
	task.mlpSurrogate.mlpInitMode = 'xavier';
	task.mlpSurrogate.mlpInitScale = 1.0;
	task.mlpSurrogate.reportStage1DenseDictionary = false;
	task.mlpSurrogate.stage1MlpPredictionWeight = 1;
	task.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0;
	task.mlpSurrogate.mlpHiddenDims = [8];
	task.mlpSurrogate.surrogateNumStarts = 6;
	if priorLevel >= 3
		task.mlpSurrogate.surrogateMaxIter = 200;
		task.mlpSurrogate.surrogateMaxFunEvals = 3e4;
	else
		task.mlpSurrogate.surrogateMaxIter = 300;
		task.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	end
	task.mlpSurrogate.useParallelStarts = true;
	task.mlpSurrogate.parallelAutoStartPool = true;

	% Stage-II exact recovery controls.
	task.exactRecovery = struct();
	task.exactRecovery.stlsThreshold = 1e-3;
	task.exactRecovery.stlsMaxIter = 10;
	task.exactRecovery.stlsLambda2 = 1e-10;
	task.exactRecovery.stlsMinTermsPerRow = 1;
	task.exactRecovery.finalBP = struct('enable', true, 'maxIter', 200, 'maxFunEvals', 3e4, 'maxRelValIncrease', 1e-4);

	% Dictionary interface.
	D = struct();
	D.caseId = task.name;
	D.priorLevel = priorLevel;
	D.priorLevelName = priorName;
	D.noFallback = true;
	D.appendGlobalTerms = false;

	if priorLevel == 0
		% Level 0: official PySR Stage-0 with universal SR grammar.
		% No generated/general PhDN dictionary or fixed PhDN architecture is built here.
		task.prior.srGrammar = make_universal_sr_stage0_grammar_local();
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 1
		% Level 1: official PySR Stage-0 with case-based compact SR grammar.
		% The former compact terms are used only to infer the PySR operator grammar;
		% they are not installed as a predefined PhDN dictionary.
		task.prior.srGrammar = make_compact_sr_stage0_grammar_from_terms_local({ ...
			'v2','v1*v2*cos(v3)'
			});
		task = apply_sr_stage0_no_predefined_phdn_placeholder_local(task, D, priorLevel, priorName);
	elseif priorLevel == 2
		% Level 2: compact dictionary per layer.
		D.termsByLayer = cell(1, task.arch.layer);
		D.termsByLayer{1} = {'v2', 'v1*v2*cos(v3)'};
		D.source = 'level 2 layerwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'layerwise_compact_prior';
	elseif priorLevel == 3
		% Level 3: compact dictionary per active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v2', 'v1*v2*cos(v3)'};
		D.source = 'level 3 branchwise compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'branchwise_compact_prior';
	else
		% Level 4: row-wise support inside every active branch block.
		D.termsByBlock = make_empty_block_terms(task.arch.layer);
		D.rowTerms = make_empty_block_terms(task.arch.layer);
		D.termsByBlock{1,1} = {'v2', 'v1*v2*cos(v3)'};
		D.rowTerms{1,1} = {{'v2', 'v1*v2*cos(v3)'}};
		D.source = 'level 4 rowwise strong compact prior dictionary';
		task.arch.caseDictionary = D;
		task.arch.dictionaryMode = 'rowwise_strong_compact_prior';
		task.DisplaySymbolic = true;
	end

	if priorLevel < 4
		task.DisplaySymbolic = false;
	end
end

% -------------------------------------------------------------------------
% RHS functions
% -------------------------------------------------------------------------
function Y = rhs_I_6_2(X)
	theta = X(:,1); sigma = X(:,2);
	Y = exp(-(theta.^2) ./ (2*sigma.^2)) ./ sqrt(2*pi*sigma.^2);
end

function Y = rhs_I_6_2b(X)
	theta = X(:,1); theta1 = X(:,2); sigma = X(:,3);
	Y = exp(-((theta-theta1).^2) ./ (2*sigma.^2)) ./ sqrt(2*pi*sigma.^2);
end

function Y = rhs_I_9_18(X)
	a = X(:,1); b = X(:,2);c = X(:,3); d = X(:,4);e = X(:,5); f = X(:,6);
	Y = a ./ ((b - 1).^2 + (c-d).^2 + (e-f).^2);
end

function Y = rhs_I_12_11(X)
	a = X(:,1); theta = X(:,2);
	Y = 1 + a .* sin(theta);
end

function Y = rhs_I_13_12(X)
	a = X(:,1); b = X(:,2);
	Y = a .* (1 ./ b - 1);
end

function Y = rhs_I_15_3x(X)
	a = X(:,1); b = X(:,2);
	Y = (1 - a) ./ sqrt(1 - b.^2);
end

function Y = rhs_I_16_6(X)
	a = X(:,1); b = X(:,2);
	Y = (a + b) ./ (1 + a.*b);
end

function Y = rhs_I_18_4(X)
	a = X(:,1); b = X(:,2);
	Y = (1 + a.*b) ./ (1 + a);
end

function Y = rhs_I_26_2(X)
	n = X(:,1); theta2 = X(:,2);
	Y = asin(n .* sin(theta2));
end

function Y = rhs_I_27_6(X)
	a = X(:,1); b = X(:,2);
	Y = 1 ./ (1 + a.*b);
end

function Y = rhs_I_29_16(X)
	a = X(:,1); theta1 = X(:,2); theta2 = X(:,3);
	Y = sqrt(1 + a.^2 - 2*a.*cos(theta1-theta2));
end

function Y = rhs_I_30_3(X)
	n = X(:,1); theta = X(:,2);
	den = sin(theta/2).^2;
	den = max(den, 1e-12);
	Y = sin(n.*theta/2).^2 ./ den;
end

function Y = rhs_I_30_5(X)
	a = X(:,1); n = X(:,2);
	Y = asin(a ./ n);
end

function Y = rhs_I_37_4(X)
	a = X(:,1); delta = X(:,2);
	Y = 1 + a + 2*sqrt(a).*cos(delta);
end

function Y = rhs_I_40_1(X)
	n0 = X(:,1); a = X(:,2);
	Y = n0 .* exp(-a);
end

function Y = rhs_I_44_4(X)
	n = X(:,1); a = X(:,2);
	Y = n .* log(a);
end

function Y = rhs_I_50_26(X)
	a = X(:,1); alpha = X(:,2);
	Y = cos(a) + alpha .* cos(a).^2;
end

function Y = rhs_II_2_42(X)
	a = X(:,1); b = X(:,2);
	Y = (a - 1) .* b;
end

function Y = rhs_II_6_15a(X)
	a = X(:,1); b = X(:,2); c = X(:,3);
	Y = (1/(4*pi)) .* c .* sqrt(a.^2 + b.^2);
end

function Y = rhs_II_11_7(X)
	n0 = X(:,1); a = X(:,2); theta = X(:,3);
	Y = n0 .* (1 + a.*cos(theta));
end

function Y = rhs_II_11_27(X)
	n = X(:,1); alpha = X(:,2);
	Y = n.*alpha ./ (1 - n.*alpha/3);
end

function Y = rhs_II_35_18(X)
	n0 = X(:,1); a = X(:,2);
	Y = n0 ./ (exp(a) + exp(-a));
end

function Y = rhs_II_36_38(X)
	a = X(:,1); alpha = X(:,2); b = X(:,3);
	Y = a + alpha.*b;
end

function Y = rhs_II_38_3(X)
	a = X(:,1); b = X(:,2);
	Y = a ./ b;
end

function Y = rhs_III_9_52(X)
	a = X(:,1); b = X(:,2); c = X(:,3);
	z = (b - c)/2;
	Y = a .* (sin(z).^2) ./ (z.^2);
end

function Y = rhs_III_10_19(X)
	a = X(:,1); b = X(:,2);
	Y = sqrt(1 + a.^2 + b.^2);
end

function Y = rhs_III_17_37(X)
	alpha = X(:,1); beta = X(:,2); theta = X(:,3);
	Y = beta .* (1 + alpha.*cos(theta));
end

% -------------------------------------------------------------------------
% Symbolic reference functions
% -------------------------------------------------------------------------
function expr = sym_I_6_2()
	syms x1 x2 real
	expr = exp(-x1^2/(2*x2^2)) / sqrt(2*sym(pi)*x2^2);
end

function expr = sym_I_6_2b()
	syms x1 x2 x3 real
	expr = exp(-(x1-x2)^2/(2*x3^2)) / sqrt(2*sym(pi)*x3^2);
end

function expr = sym_I_9_18()
	syms x1 x2 x3 x4 x5 x6 real 
	expr = x1 / ((x2 - 1)^2 + (x3-x4)^2 + (x5-x6)^2);
end

function expr = sym_I_12_11()
	syms x1 x2 real
	expr = 1 + x1*sin(x2);
end

function expr = sym_I_13_12()
	syms x1 x2 real
	expr = x1*(1/x2 - 1);
end

function expr = sym_I_15_3x()
	syms x1 x2 real
	expr = (1 - x1) / sqrt(1 - x2^2);
end

function expr = sym_I_16_6()
	syms x1 x2 real
	expr = (x1 + x2) / (1 + x1*x2);
end

function expr = sym_I_18_4()
	syms x1 x2 real
	expr = (1 + x1*x2) / (1 + x1);
end

function expr = sym_I_26_2()
	syms x1 x2 real
	expr = asin(x1*sin(x2));
end

function expr = sym_I_27_6()
	syms x1 x2 real
	expr = 1 / (1 + x1*x2);
end

function expr = sym_I_29_16()
	syms x1 x2 x3 real
	expr = sqrt(1 + x1^2 - 2*x1*cos(x2-x3));
end

function expr = sym_I_30_3()
	syms x1 x2 real
	expr = sin(x1*x2/2)^2 / sin(x2/2)^2;
end

function expr = sym_I_30_5()
	syms x1 x2 real
	expr = asin(x1/x2);
end

function expr = sym_I_37_4()
	syms x1 x2 real
	expr = 1 + x1 + 2*sqrt(x1)*cos(x2);
end

function expr = sym_I_40_1()
	syms x1 x2 real
	expr = x1*exp(-x2);
end

function expr = sym_I_44_4()
	syms x1 x2 real
	expr = x1*log(x2);
end

function expr = sym_I_50_26()
	syms x1 x2 real
	expr = cos(x1) + x2*cos(x1)^2;
end

function expr = sym_II_2_42()
	syms x1 x2 real
	expr = (x1 - 1)*x2;
end

function expr = sym_II_6_15a()
	syms x1 x2 x3 real
	expr = sym(1)/(4*sym(pi))*x3*sqrt(x1^2 + x2^2);
end

function expr = sym_II_11_7()
	syms x1 x2 x3 real
	expr = x1*(1 + x2*cos(x3));
end

function expr = sym_II_11_27()
	syms x1 x2 real
	expr = x1*x2/(1 - x1*x2/3);
end

function expr = sym_II_35_18()
	syms x1 x2 real
	expr = x1/(exp(x2) + exp(-x2));
end

function expr = sym_II_36_38()
	syms x1 x2 x3 real
	expr = x1 + x2*x3;
end

function expr = sym_II_38_3()
	syms x1 x2 real
	expr = x1/x2;
end

function expr = sym_III_9_52()
	syms x1 x2 x3 real
	expr = x1*sin((x2-x3)/2)^2 / (((x2-x3)/2)^2);
end

function expr = sym_III_10_19()
	syms x1 x2 real
	expr = sqrt(1 + x1^2 + x2^2);
end

function expr = sym_III_17_37()
	syms x1 x2 x3 real
	expr = x2*(1 + x1*cos(x3));
end
