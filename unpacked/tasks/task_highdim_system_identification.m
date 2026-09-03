function task = task_highdim_system_identification(caseName, casemode)
%TASK_HIGHDIM_SYSTEM_IDENTIFICATION Multi-input/multi-output SI benchmarks.
%
% The case registry is intentionally independent of the Feynman registry so
% that additional high-dimensional numerical systems can be appended without
% changing the Feynman benchmark definitions.
%
% Case names:
%   HD_1 : three inputs and three nonlinear outputs.
%
% Supported casemodes in v73d:
%   general           : official PySR Stage 0 with a broad operator grammar.
%   weak_prior_lv1    : official PySR Stage 0 with the case operator grammar.
%   weak_prior_lv2    : the level-1 grammar plus the shared physical motif
%                       op_custom1(a,b)=log(exp(a)+inv(b)).

    if nargin < 1 || isempty(caseName)
        caseName = 'HD_1';
    end
    if nargin < 2 || isempty(casemode)
        casemode = 'general';
    end

    caseName = lower(strtrim(char(caseName)));
    casemode = lower(strtrim(char(casemode)));

    task = struct();
    task.caseName = caseName;
    task.casemode = casemode;
    task.sourceName = 'HighDimensional_System_Identification';
    task.dataDefaults = struct('nSamples', 2500, 'ratioTrain', 0.6, ...
        'ratioVal', 0.2, 'nOODSamples', 500);

    switch caseName
        case {'hd_1','hd1','case_1','case1'}
            task = setup_HD_1(task);
        otherwise
            error('Unknown high-dimensional system-identification case: %s', caseName);
    end

    if exist('model_to_symbolic_general', 'file') == 2
        task.modelToSymbolicFcn = @model_to_symbolic_general;
    else
        task.modelToSymbolicFcn = [];
    end
end

function task = setup_HD_1(task)
%SETUP_HD_1 Three-input/three-output nonlinear numerical benchmark.
%
% y1 = x2
% y2 = log(exp(x1*x2) + 1/(x1^2 + x2 + 1)) + sin(x2)
% y3 = log(exp(x2) + 1/(x1^2 + x3 + x2*x3)) + x1/x3

    task.caseName = 'HD_1';
    task.name = 'HighDimSI_HD_1';
    task.description = ['Three-input/three-output nonlinear SI benchmark with ', ...
        'one direct state output and nested exp/log, rational, polynomial, ', ...
        'product, and sinusoidal structure.'];
    task.nx = 3;
    task.ny = 3;
    task.variableNames = {'x1','x2','x3'};
    task.outputNames = {'y1','y2','y3'};

    % The ID box keeps x3 and both logarithm arguments strictly positive.
    % It also limits exponential scale so the comparison is structural rather
    % than dominated by avoidable overflow or near-singular samples.
    task.domain = make_variable_union_domain(task.variableNames, { ...
        [0.2, 1.5], ...
        [-1.00, 2.00], ...
        [0.50, 2.50]});

    % A disjoint upper-range challenge set. These bounds remain safely inside
    % the real domain of both logarithms and away from x3 = 0.
    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [1.6, 2.00], ...
        [2.10, 2.50], ...
        [2.60, 3.00]});
    task.oodDomain.source = 'task_defined_disjoint_upper_challenge';

    task.rhsFcn = @rhs_HD_1;
    task.referenceSymbolicFcn = @sym_HD_1;

    [priorLevel, priorName, grammar] = resolve_prior_mode_local(task.casemode);
    task.casemode = priorName;

    % For prior levels 0/1/2, this is only a dimension-carrying placeholder.
    % The actual multi-output PhDN DAG is compiled from the independently
    % selected per-output PySR expressions in Stage 0.
    task.arch = struct();
    task.arch.nx = task.nx;
    task.arch.ny = task.ny;
    task.arch.layer = 1;
    task.arch.hiddenDims = [];
    task.arch.polyOrder = 1;
    task.arch.operatorMode = 'true';
    task.arch.dictionaryMode = 'sr_stage0_expression_determined_no_predefined_dictionary';
    task.arch.branchActiveMask = true(1,1);
    task.arch.branchActiveMode = 'sr_stage0_only_placeholder';
    task.arch.srStage0Only = true;
    task.arch.srStage0StructureSource = 'per-output SINDy/PySR selected symbolic core archive';

    D = struct();
    D.caseId = task.name;
    D.priorLevel = priorLevel;
    D.priorLevelName = priorName;
    D.noFallback = true;
    D.appendGlobalTerms = false;
    D.termsByDim = {};
    D.source = ['empty SR-Stage0 placeholder: no predefined PhDN dictionary; ', ...
        'the candidate DAG is compiled from selected PySR expressions'];
    task.arch.caseDictionary = D;

    task.operatorMode = 'true';
    task.operatorControl = struct('caseDefault','true', ...
        'demoOverride','task_default','singleLayerForceTrue',true);

    task.training = struct();
    task.training.operatorMode = 'true';
    task.training.lambda1List = 1e-6;
    task.training.tauList = 0;
    task.training.opArgPolyOrderList = 1;

    task.prior = struct();
    task.prior.priorInterfaceEnabled = true;
    task.prior.level = priorLevel;
    task.prior.levelName = priorName;
    task.prior.dictionaryMode = sprintf('prior_level_%d_%s_sr_stage0_only', ...
        priorLevel, priorName);
    task.prior.useAdmissibleMask = false;
    task.prior.maskType = '';
    task.prior.useWeakStrongDictionaryMask = false;
    task.prior.dictionaryMaskGranularity = 'none';
    task.prior.useTwoStageMlpRecovery = false;
    task.prior.srStage0Enable = true;
    task.prior.srStage0UsePredefinedPhdnDictionary = false;
    task.prior.stage0Mode = 'official_pysr';
    task.prior.srGrammarMode = grammar.mode;
    task.prior.phdnStructureSource = 'selected official-PySR expressions';
    task.prior.srGrammar = grammar;

    task.DisplaySymbolic = false;
end

function [priorLevel, priorName, grammar] = resolve_prior_mode_local(casemode)
    switch lower(strtrim(char(casemode)))
        case {'general','level0','prior0'}
            priorLevel = 0;
            priorName = 'general';
            grammar = struct( ...
                'mode', 'universal', ...
                'source', ['level 0 broad official-PySR grammar for the ', ...
                    'high-dimensional SI registry'], ...
                'binaryOperators', {{'+','-','*','/'}}, ...
                'unaryOperators', {{'square','cube','inv','sqrt','exp','sin','cos','log'}}, ...
                'operatorComplexities', struct());
        case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
            priorLevel = 1;
            priorName = 'weak_prior_lv1';
            grammar = struct( ...
                'mode', 'case_compact', ...
                'source', ['level 1 case-compact official-PySR grammar; ', ...
                    'operators only, with no predefined PhDN dictionary'], ...
                'binaryOperators', {{'+','-','*','/'}}, ...
                'unaryOperators', {{'square','inv','exp','sin','log'}}, ...
                'operatorComplexities', struct());
        case {'weak_prior_lv2','weak_prior_l2','level2','prior2'}
            priorLevel = 2;
            priorName = 'weak_prior_lv2';
            grammar = struct( ...
                'mode', 'case_compact_with_shared_custom_motif', ...
                'source', ['level 2 case-compact official-PySR grammar plus ', ...
                    'the shared physical motif op_custom1(a,b)=', ...
                    'log(exp(a)+inv(b)); no variable-specific answer is supplied'], ...
                'binaryOperators', {{'+','-','*','/','op_custom1'}}, ...
                'unaryOperators', {{'square','inv','exp','sin','log'}}, ...
                'operatorComplexities', struct('op_custom1', 4), ...
                'customOperatorSemantics', struct( ...
                    'op_custom1', 'log(exp(a)+inv(b))'));
        otherwise
            error(['Unsupported high-dimensional casemode in v73d: %s. ', ...
                'Use general, weak_prior_lv1, or weak_prior_lv2.'], casemode);
    end
end

function Y = rhs_HD_1(X)
    validateattributes(X, {'numeric'}, {'2d','ncols',3,'real','finite'}, ...
        mfilename, 'X');
    x1 = X(:,1);
    x2 = X(:,2);
    x3 = X(:,3);

    den1 = x1.^2 + x2 + 1;
    den2 = x1.^2 + x3 + x2.*x3;
    if any(den1 <= 0) || any(den2 <= 0) || any(x3 == 0)
        error('HD_1 received samples outside its real, nonsingular domain.');
    end

    y1 = x2;
    y2 = log(exp(x1.*x2) + 1./den1) + sin(x2);
    y3 = log(exp(x2) + 1./den2) + x1./x3;
    Y = [y1, y2, y3];
end

function expr = sym_HD_1()
    syms x1 x2 x3 real
    y1 = x2;
    y2 = log(exp(x1*x2) + 1/(x1^2 + x2 + 1)) + sin(x2);
    y3 = log(exp(x2) + 1/(x1^2 + x3 + x2*x3)) + x1/x3;
    expr = [y1; y2; y3];
end
