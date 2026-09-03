function info = verify_single_generator_dynamic_case_setup(task)
%VERIFY_SINGLE_GENERATOR_DYNAMIC_CASE_SETUP Check case-local SMIB setup.
%
% This guard is intentionally case specific. It does not modify or depend on
% shared SR-to-PhDN compiler or baseline logic.

    if nargin < 1 || isempty(task)
        task = task_single_generator_dynamic('SMIB_AVR','general');
    end

    expected = arrayfun(@(k) sprintf('x%d',k),1:task.nx,'UniformOutput',false);
    assert(isequal(task.variableNames,expected), ...
        'Expected canonical task variables x1,...,x%d.',task.nx);
    assert(isfield(task,'physicalVariableNames') && ...
        numel(task.physicalVariableNames) == task.nx, ...
        'Physical variable-name metadata is missing or incomplete.');

    p = task.parameters;
    assert(isfield(p,'Xq') && isfinite(p.Xq) && abs(p.Xq-p.Xdp) > 1e-8, ...
        'Salient-pole setup requires Xq distinct from Xdp.');
    assert(isfield(p,'Ksal') && isfinite(p.Ksal) && abs(p.Ksal) > 1e-8, ...
        'The salient-pole reluctance-power coefficient is missing or zero.');

    rhsEq = task.rhsFcn(reshape(task.equilibrium,1,[]));
    equilibriumMaxAbsRhs = max(abs(rhsEq(:)));
    assert(isfinite(equilibriumMaxAbsRhs) && equilibriumMaxAbsRhs <= 1e-10, ...
        'SMIB equilibrium RHS check failed: max abs RHS = %.3e.', ...
        equilibriumMaxAbsRhs);

    % The dynamic test must jointly excite the terminal-voltage/AVR path:
    % delta is high, E'_q is low, and E_fd is high relative to the ID box.
    idLb = reshape(task.domain.lb,1,[]);
    idUb = reshape(task.domain.ub,1,[]);
    oodLb = reshape(task.oodDomain.lb,1,[]);
    oodUb = reshape(task.oodDomain.ub,1,[]);
    icLb = reshape(task.rollout.initialConditionDomain.lb,1,[]);
    icUb = reshape(task.rollout.initialConditionDomain.ub,1,[]);
    assert(oodLb(1) > idUb(1) && oodUb(3) < idLb(3) && oodLb(4) > idUb(4), ...
        ['Static OOD must jointly shift delta upward, E''_q downward, ', ...
         'and E_fd upward to challenge y4.']);
    assert(icLb(1) > idUb(1) && icUb(3) < idLb(3) && icLb(4) > idUb(4), ...
        ['Rollout ICs must jointly shift delta upward, E''_q downward, ', ...
         'and E_fd upward to challenge y4.']);
    assert(icLb(2) < idLb(2) || icUb(2) > idUb(2), ...
        'The rollout speed interval should also extend modestly beyond ID.');

    % Check finite RHS values at the representative OOD initial point and at
    % the corners used by the sampler before any expensive PySR run.
    probe = [task.rollout.referenceInitialCondition; icLb; icUb];
    probeRhs = task.rhsFcn(probe);
    assert(all(isfinite(probeRhs(:))), ...
        'The salient-pole linear-AVR RHS is nonfinite at an OOD probe.');

    % Fixed-step ODE4 reference guard before the expensive PySR stage.
    assert(strcmpi(task.rollout.solver,'ode4'), ...
        'This case expects fixed-step ODE4 rollout evaluation.');
    nExpected = round(task.rollout.horizon/task.rollout.fixedStep)+1;
    assert(task.rollout.nOutputTimes == nExpected, ...
        'nOutputTimes must equal horizon/fixedStep+1 for ODE4.');
    xProbe = ode4_reference_local(task,task.rollout.referenceInitialCondition);
    assert(all(isfinite(xProbe(:))), ...
        'Reference joint-OOD ODE4 rollout is nonfinite.');


    info = struct();
    info.srVariableNames = task.variableNames;
    info.physicalVariableNames = task.physicalVariableNames;
    info.mappingDescription = task.variableMappingDescription;
    info.modelVariant = task.modelVariant;
    info.equilibriumMaxAbsRhs = equilibriumMaxAbsRhs;
    info.saliencyCoefficient = p.Ksal;
    info.rolloutInitialConditionSource = ...
        task.rollout.initialConditionDomain.source;

    fprintf('SingleGeneratorDynamic case-local mapping verified: %s\n', ...
        task.variableMappingDescription);
    fprintf('Model variant: %s\n',task.modelVariant);
    fprintf('Equilibrium max |RHS| = %.3e\n',equilibriumMaxAbsRhs);
    fprintf('Saliency Ksal = %.6g | linear AVR gain KA = %.6g\n', ...
        p.Ksal,p.KA);
    fprintf(['Rollout IC design verified: hard joint delta-Eqp-Efd OOD ', ...
        'with y4/terminal-voltage excitation.\n']);
end

function X = ode4_reference_local(task,x0)
    t = (0:task.rollout.fixedStep:task.rollout.horizon).';
    X = nan(numel(t),task.nx);
    X(1,:) = reshape(x0,1,[]);
    lim = reshape(task.rollout.maxStateAbs,[],1);
    for k = 1:numel(t)-1
        h = t(k+1)-t(k);
        x = X(k,:).';
        f = @(z) reshape(task.rhsFcn(reshape(z,1,[])),[],1);
        k1 = f(x);
        k2 = f(x+0.5*h*k1);
        k3 = f(x+0.5*h*k2);
        k4 = f(x+h*k3);
        dxSet = [k1;k2;k3;k4];
        assert(all(isfinite(dxSet)) && max(abs(dxSet)) <= task.rollout.maxDerivativeAbs, ...
            'Reference ODE4 derivative guard failed.');
        xn = x+(h/6)*(k1+2*k2+2*k3+k4);
        assert(all(isfinite(xn)) && all(abs(xn)<=lim), ...
            'Reference ODE4 state safety envelope failed.');
        X(k+1,:) = xn.';
    end
end
