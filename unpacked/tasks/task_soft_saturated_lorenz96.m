function task = task_soft_saturated_lorenz96(caseName,casemode,F,kappa)
%TASK_SOFT_SATURATED_LORENZ96 Soft-saturated Lorenz--96 SI cases.
%
% Cases:
%   SS_L96_K8  : K=8 with demo-supplied F and kappa.
%   SS_L96_K12 : K=12 with demo-supplied F and kappa.
% Legacy names ending in _KAPPA1 remain accepted.
%
% Supported casemodes:
%   general        : broad official-PySR grammar.
%   weak_prior_lv1 : compact arithmetic/square/sqrt operator grammar.

    if nargin < 1 || isempty(caseName); caseName = 'SS_L96_K8'; end
    if nargin < 2 || isempty(casemode); casemode = 'general'; end
    if nargin < 4 || isempty(F) || isempty(kappa)
        error('F and kappa must be supplied explicitly by the Lorenz demo.');
    end
    caseName = lower(strtrim(char(caseName)));
    casemode = lower(strtrim(char(casemode)));

    K = parse_dimension_local(caseName);
    p = soft_saturated_lorenz96_parameters(K,F,kappa);

    task = struct();
    parameterTag = sprintf('F%s_KAPPA%s',numeric_tag_local(p.F),numeric_tag_local(p.kappa));
    task.caseName = sprintf('SS_L96_K%d_%s',p.K,upper(parameterTag));
    task.casemode = casemode;
    task.sourceName = 'SoftSaturatedLorenz96';
    task.name = sprintf('SoftSaturatedLorenz96_K%d_F%s_kappa%s', ...
        p.K,numeric_tag_local(p.F),numeric_tag_local(p.kappa));
    task.description = sprintf(['%d-state cyclic Lorenz--96 system with ', ...
        'F=%.6g and kappa=%.6g, using smooth saturation on the nonlinear ', ...
        'transport term.'],p.K,p.F,p.kappa);
    task.nx = p.K;
    task.ny = p.K;
    task.variableNames = p.stateNames;
    task.physicalVariableNames = p.stateNames;
    task.displayVariableNames = p.stateNames;
    task.outputNames = p.derivativeNames;
    task.parameters = p;
    task.equilibrium = p.xEquilibrium;
    task.variableMappingDescription = sprintf('x1,...,x%d are cyclic Lorenz--96 states',p.K);
    task.modelVariant = p.modelVariant;

    idIntervals = repmat({p.F+[-1.5,1.5]},1,p.K);
    task.domain = make_variable_union_domain(task.variableNames,idIntervals);
    task.domain.source = sprintf('%s_ID_box',task.caseName);

    oodIntervals = cell(1,p.K);
    rolloutIntervals = cell(1,p.K);
    xRef = zeros(1,p.K);
    for i = 1:p.K
        if mod(i,2)==1
            oodIntervals{i} = p.F+[-2.8,-1.8];
            rolloutIntervals{i} = p.F+[-2.3,-1.8];
            xRef(i) = p.F-2.0;
        else
            oodIntervals{i} = p.F+[1.8,2.8];
            rolloutIntervals{i} = p.F+[1.8,2.3];
            xRef(i) = p.F+2.0;
        end
    end
    task.oodDomain = make_variable_union_domain(task.variableNames,oodIntervals);
    task.oodDomain.source = sprintf('%s_alternating_joint_OOD_box',task.caseName);

    task.rhsFcn = @(X) soft_saturated_lorenz96_rhs(X,p);
    task.referenceSymbolicFcn = @() reference_symbolic_local(p);
    task.dataDefaults = struct('nSamples',2500,'ratioTrain',0.6, ...
        'ratioVal',0.2,'nOODSamples',1000);

    [priorLevel,priorName,grammar] = resolve_prior_mode_local(casemode);
    task.casemode = priorName;

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
    task.arch.srStage0StructureSource = 'per-output selected symbolic core archive';

    D = struct();
    D.caseId = task.name;
    D.priorLevel = priorLevel;
    D.priorLevelName = priorName;
    D.noFallback = true;
    D.appendGlobalTerms = false;
    D.termsByDim = {};
    D.source = sprintf(['empty SR-Stage0 placeholder; the selected expressions ', ...
        'are compiled and augmented by %d fixed neural-ridge bases per branch'],p.K);
    task.arch.caseDictionary = D;

    task.operatorMode = 'true';
    task.operatorControl = struct('caseDefault','true', ...
        'demoOverride','task_default','singleLayerForceTrue',true);
    task.training = struct('operatorMode','true','lambda1List',1e-6, ...
        'tauList',0,'opArgPolyOrderList',1);

    task.prior = struct();
    task.prior.priorInterfaceEnabled = true;
    task.prior.level = priorLevel;
    task.prior.levelName = priorName;
    task.prior.dictionaryMode = sprintf('prior_level_%d_%s_sr_stage0_only', ...
        priorLevel,priorName);
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

    task.rollout = struct();
    task.rollout.solver = 'ode4';
    task.rollout.horizon = 5.0;
    task.rollout.fixedStep = 0.01;
    task.rollout.nOutputTimes = round(task.rollout.horizon/task.rollout.fixedStep)+1;
    task.rollout.nInitialConditions = 12;
    task.rollout.initialConditionSeed = 9201;
    task.rollout.maxStateAbs = max(20,abs(p.F)+12)*ones(1,p.K);
    task.rollout.maxDerivativeAbs = 1e4;
    task.rollout.initialConditionDomain = make_variable_union_domain( ...
        task.variableNames,rolloutIntervals);
    task.rollout.initialConditionDomain.source = sprintf( ...
        '%s_alternating_joint_OOD_IC_box',task.caseName);
    task.rollout.initialConditionShift = ...
        'alternating low/high cyclic states outside the identification box';
    task.rollout.stateScale = max(reshape(task.domain.ub-task.domain.lb,1,[]), ...
        reshape(task.rollout.initialConditionDomain.ub- ...
        task.rollout.initialConditionDomain.lb,1,[]));
    task.rollout.referenceInitialCondition = xRef;
    task.rollout.representativeSelection = 'hardest_common_normalized_rmse';
    task.DisplaySymbolic = false;

    if exist('model_to_symbolic_general','file')==2
        task.modelToSymbolicFcn = @model_to_symbolic_general;
    else
        task.modelToSymbolicFcn = [];
    end
end

function K = parse_dimension_local(caseName)
    if any(strcmp(caseName,{'soft_saturated_lorenz96','l96'}))
        K = 8;
        return;
    end
    token = regexp(caseName,'^ss_l96_k([0-9]+)(?:_.*)?$','tokens','once');
    if isempty(token)
        error('Unknown soft-saturated Lorenz--96 case: %s',caseName);
    end
    K = str2double(token{1});
    if ~isfinite(K) || K<4 || K~=round(K)
        error('Lorenz--96 case dimension must be an integer K>=4.');
    end
end

function [priorLevel,priorName,grammar] = resolve_prior_mode_local(casemode)
    switch lower(strtrim(char(casemode)))
        case {'general','level0','prior0'}
            priorLevel = 0;
            priorName = 'general';
            grammar = struct('mode','universal', ...
                'source','broad official-PySR grammar for soft-saturated L96', ...
                'binaryOperators',{{'+','-','*','/'}}, ...
                'unaryOperators',{{'square','cube','inv','sqrt','exp','sin','cos','log'}}, ...
                'operatorComplexities',struct());
        case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
            priorLevel = 1;
            priorName = 'weak_prior_lv1';
            grammar = struct('mode','case_compact', ...
                'source','operator-only saturation grammar without target composites', ...
                'binaryOperators',{{'+','-','*','/'}}, ...
                'unaryOperators',{{'square','sqrt'}}, ...
                'operatorComplexities',struct());
        otherwise
            error('Unknown soft-saturated Lorenz--96 casemode: %s',casemode);
    end
end

function tag = numeric_tag_local(value)
    tag = sprintf('%.8g',double(value));
    tag = strrep(tag,'-','m');
    tag = strrep(tag,'+','');
    tag = strrep(tag,'.','p');
end

function expr = reference_symbolic_local(p)
    x = sym('x',[p.K,1],'real');
    expr = sym(zeros(p.K,1));
    for i = 1:p.K
        im2 = mod(i-3,p.K)+1;
        im1 = mod(i-2,p.K)+1;
        ip1 = mod(i,p.K)+1;
        z = x(im1)*(x(ip1)-x(im2));
        expr(i) = z/sqrt(1+(z/p.kappa)^2)-x(i)+p.F;
    end
end
