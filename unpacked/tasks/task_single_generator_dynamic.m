function task = task_single_generator_dynamic(caseName, casemode)
%TASK_SINGLE_GENERATOR_DYNAMIC Low-dimensional power-system SI benchmarks.
%
% Case:
%   SMIB_AVR : four-state salient-pole single-machine infinite-bus model
%              with flux decay and a linear AVR.
%
% Supported casemodes:
%   general        : broad official-PySR grammar, no case-specific expression.
%   weak_prior_lv1 : compact operator grammar {sin,cos,sqrt,square} but no
%                    variable-specific composite terms or predefined PhDN DAG.
%
% The task target is the explicit standard ODE map x -> dot(x). Conventional
% coefficients 2H, T'_do, and T_A are divided into the right-hand sides.

    if nargin < 1 || isempty(caseName)
        caseName = 'SMIB_AVR';
    end
    if nargin < 2 || isempty(casemode)
        casemode = 'general';
    end

    caseName = lower(strtrim(char(caseName)));
    casemode = lower(strtrim(char(casemode)));

    task = struct();
    task.caseName = caseName;
    task.casemode = casemode;
    task.sourceName = 'SingleGeneratorDynamic';

    switch caseName
        case {'smib_avr','smib-avr','single_generator','single_generator_dynamic'}
            task = setup_smib_avr_local(task);
        otherwise
            error('Unknown SingleGeneratorDynamic case: %s', caseName);
    end

    if exist('model_to_symbolic_general', 'file') == 2
        task.modelToSymbolicFcn = @model_to_symbolic_general;
    else
        task.modelToSymbolicFcn = [];
    end
end

function task = setup_smib_avr_local(task)
    p = single_generator_dynamic_parameters();

    task.caseName = 'SMIB_AVR';
    task.name = 'SingleGeneratorDynamic_SMIB_AVR';
    task.description = ['Four-state salient-pole SMIB flux-decay model with ', ...
        'reluctance power, unequal d/q terminal-voltage channels, and a ', ...
        'linear AVR. The target retains multi-level trigonometric and ', ...
        'terminal-voltage magnitude composition with a linear AVR.'];
    task.nx = 4;
    task.ny = 4;
    % Case-local variable-name policy.  The SR/compiler-facing identifiers are
    % deliberately canonical one-based names, matching the established
    % Feynman/high-dimensional task convention.  Physical names are retained
    % separately for equations, figures, and manuscript reporting.  This
    % prevents a PySR fallback token such as x1 from being ambiguously treated
    % as both a zero-based second feature and a one-based first feature.
    task.variableNames = p.srVariableNames;
    task.physicalVariableNames = p.stateNames;
    task.displayVariableNames = p.stateNames;
    task.variableNameMap = p.stateNameMap;
    task.outputNames = p.derivativeNames;
    task.parameters = p;
    task.equilibrium = p.xEquilibrium;
    task.variableMappingDescription = ...
        'x1=delta, x2=domega, x3=Eqp, x4=Efd';
    task.modelVariant = p.modelVariant;

    % Identification state box. Samples are independent operating points,
    % not adjacent observations from one trajectory, so temporal leakage does
    % not arise. The demo installs a deterministic nested train split.
    % Broad ID box: wide enough that local polynomial combinations cannot
    % cheaply imitate the salient-pole terminal-voltage composition over the
    % whole domain. The independent SINDy baseline keeps its unchanged generic
    % flat dictionary; no target-specific Vq, Vd, Vt, voltage-error, or
    % sin(2*delta) composite columns are supplied.
    task.domain = make_variable_union_domain(task.variableNames, { ...
        [0.15, 1.05], ...       % delta [rad]
        [-0.012, 0.012], ...    % domega [p.u.]
        [0.75, 1.45], ...       % E'_q [p.u.]
        [0.65, 2.55]});         % E_fd [p.u.]
    task.domain.source = 'SMIB_salient_linear_AVR_broad_ID_box';

    % Hard joint OOD box designed to expose the terminal-voltage/AVR channel.
    % delta is above the ID range, E'_q is below it, and E_fd is above it.
    % This jointly perturbs both arguments controlling V_t and the fast y4
    % leakage/forcing balance instead of testing only a tiny angle shift.
    task.oodDomain = make_variable_union_domain(task.variableNames, { ...
        [1.15, 1.35], ...       % above ID delta upper bound 1.05
        [-0.018, 0.018], ...    % wider than ID speed range
        [0.55, 0.70], ...       % below ID E'_q lower bound 0.75
        [2.70, 3.10]});         % above ID E_fd upper bound 2.55
    task.oodDomain.source = 'SMIB_salient_linear_AVR_joint_voltage_channel_OOD';

    task.rhsFcn = @(X) smib_avr_reference_rhs(X, p);
    task.referenceSymbolicFcn = @() reference_symbolic_local(p);

    % nSamples is overwritten by the demo because the requested sample list
    % refers to actual training samples. Validation and ID-test sample counts
    % are held fixed by sample_single_generator_dynamic_split.m.
    task.dataDefaults = struct('nSamples', 2500, 'ratioTrain', 0.6, ...
        'ratioVal', 0.2, 'nOODSamples', 1000);

    [priorLevel, priorName, grammar] = resolve_prior_mode_local(task.casemode);
    task.casemode = priorName;

    % Dimension-only placeholder. Stage 0 determines one symbolic core per
    % output and the compiler builds the actual shared PhDN DAG.
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

    % Hard joint-OOD unseen-initial-condition rollout definition.
    task.rollout = struct();
    task.rollout.solver = 'ode4';
    task.rollout.horizon = 5.0;
    task.rollout.fixedStep = 0.005;
    task.rollout.nOutputTimes = round(task.rollout.horizon/task.rollout.fixedStep)+1;
    task.rollout.nInitialConditions = 12;
    task.rollout.initialConditionSeed = 9101;
    % Fixed-step ODE4 has a bounded cost: four RHS calls per time step.
    % Safety guards convert pathological learned dynamics into rollout failures.
    task.rollout.maxStateAbs = [10, 1, 10, 20];
    task.rollout.maxDerivativeAbs = 1e4;
    task.rollout.initialConditionDomain = make_variable_union_domain(task.variableNames, { ...
        [1.18, 1.32], ...       % clearly above ID delta upper bound 1.05
        [-0.015, 0.015], ...    % modest speed extrapolation
        [0.58, 0.70], ...       % below ID E'_q lower bound 0.75
        [2.75, 3.00]});         % above ID E_fd upper bound 2.55
    task.rollout.initialConditionDomain.source = ...
        'SMIB_salient_linear_AVR_joint_voltage_channel_OOD_IC_box';
    task.rollout.initialConditionShift = ...
        'joint delta-Eqp-Efd OOD emphasizing terminal-voltage and y4 dynamics';
    task.rollout.stateScale = max( ...
        reshape(task.domain.ub-task.domain.lb,1,[]), ...
        reshape(task.rollout.initialConditionDomain.ub- ...
            task.rollout.initialConditionDomain.lb,1,[]));
    task.rollout.referenceInitialCondition = [1.28, 0.012, 0.62, 2.90];
    task.rollout.representativeSelection = 'hardest_common_normalized_rmse';

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
                    'SingleGeneratorDynamic registry'], ...
                'binaryOperators', {{'+','-','*','/'}}, ...
                'unaryOperators', {{'square','cube','inv','sqrt','exp','sin','cos','log'}}, ...
                'operatorComplexities', struct());
        case {'weak_prior','weak_prior_lv1','weak_prior_l1','level1','prior1'}
            priorLevel = 1;
            priorName = 'weak_prior_lv1';
            grammar = struct( ...
                'mode', 'case_compact', ...
                'source', ['level 1 operator-only SMIB grammar; no variable-', ...
                    'specific composite term or predefined PhDN DAG'], ...
                'binaryOperators', {{'+','-','*','/'}}, ...
                'unaryOperators', {{'square','sqrt','sin','cos'}}, ...
                'operatorComplexities', struct());
        otherwise
            error(['Unsupported SingleGeneratorDynamic casemode: %s. ', ...
                'Use general or weak_prior_lv1.'], casemode);
    end
end

function expr = reference_symbolic_local(p)
    syms delta domega Eqp Efd real
    Pe = Eqp*p.Vinf/p.XdNet*sin(delta) + p.Ksal*sin(2*delta);
    Vq = (p.Xe*Eqp + p.Xdp*p.Vinf*cos(delta))/p.XdNet;
    Vd = p.Xq*p.Vinf*sin(delta)/p.XqNet;
    Vt = sqrt(Vq^2 + Vd^2);
    voltageError = p.Vref-Vt;
    amplifierVoltage = p.KA*voltageError;

    deltaDot = p.omegaBase*domega;
    domegaDot = (p.Pm-Pe-p.D*domega)/(2*p.H);
    EqpDot = (-(p.Xd+p.Xe)/p.XdNet*Eqp ...
        +(p.Xd-p.Xdp)*p.Vinf/p.XdNet*cos(delta)+Efd)/p.TdoPrime;
    EfdDot = (amplifierVoltage-(Efd-p.Efd0))/p.TA;
    expr = [deltaDot; domegaDot; EqpDot; EfdDot];
end
