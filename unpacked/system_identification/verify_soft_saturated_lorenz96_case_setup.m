function info = verify_soft_saturated_lorenz96_case_setup(task)
%VERIFY_SOFT_SATURATED_LORENZ96_CASE_SETUP Check a parameterized K-state SI case.

    if nargin < 1 || isempty(task)
        error('A parameterized Lorenz--96 task must be supplied explicitly.');
    end

    p = task.parameters;
    assert(task.nx == p.K && task.ny == p.K && p.K>=4 && p.K==round(p.K), ...
        'The Lorenz--96 dimension must be an integer K>=4.');
    assert(isscalar(p.F) && isfinite(p.F), ...
        'The requested forcing F must be a finite scalar.');
    assert(isscalar(p.kappa) && isfinite(p.kappa) && p.kappa>0, ...
        'The requested saturation parameter kappa must be positive and finite.');

    expected = arrayfun(@(k) sprintf('x%d',k),1:p.K,'UniformOutput',false);
    assert(isequal(task.variableNames,expected), ...
        sprintf('Expected canonical task variables x1,...,x%d.',p.K));

    rhsEq = task.rhsFcn(reshape(task.equilibrium,1,[]));
    equilibriumMaxAbsRhs = max(abs(rhsEq(:)));
    assert(isfinite(equilibriumMaxAbsRhs) && equilibriumMaxAbsRhs <= 1e-12, ...
        'Uniform equilibrium RHS check failed: max abs RHS = %.3e.', ...
        equilibriumMaxAbsRhs);

    idLb = reshape(task.domain.lb,1,[]);
    idUb = reshape(task.domain.ub,1,[]);
    oodLb = reshape(task.oodDomain.lb,1,[]);
    oodUb = reshape(task.oodDomain.ub,1,[]);
    icLb = reshape(task.rollout.initialConditionDomain.lb,1,[]);
    icUb = reshape(task.rollout.initialConditionDomain.ub,1,[]);
    odd = 1:2:p.K;
    even = 2:2:p.K;
    assert(all(oodUb(odd) < idLb(odd)) && all(oodLb(even) > idUb(even)), ...
        'The static OOD box must alternate below-ID and above-ID coordinates.');
    assert(all(icUb(odd) < idLb(odd)) && all(icLb(even) > idUb(even)), ...
        'The rollout IC box must alternate below-ID and above-ID coordinates.');

    probe = [task.rollout.referenceInitialCondition; icLb; icUb];
    probeRhs = task.rhsFcn(probe);
    assert(all(isfinite(probeRhs(:))), ...
        'The soft-saturated Lorenz--96 RHS is nonfinite at an OOD probe.');
    transport = probeRhs + probe - p.F;
    assert(max(abs(transport(:))) <= p.kappa*(1+1e-12), ...
        'The nonlinear transport violates the prescribed soft-saturation bound.');

    assert(strcmpi(task.rollout.solver,'ode4'), ...
        'This case expects fixed-step ODE4 rollout evaluation.');
    nExpected = round(task.rollout.horizon/task.rollout.fixedStep)+1;
    assert(task.rollout.nOutputTimes == nExpected, ...
        'nOutputTimes must equal horizon/fixedStep+1 for ODE4.');
    xProbe = ode4_reference_local(task,task.rollout.referenceInitialCondition);
    assert(all(isfinite(xProbe(:))), ...
        'Reference ODE4 rollout is nonfinite.');

    info = struct();
    info.srVariableNames = task.variableNames;
    info.mappingDescription = task.variableMappingDescription;
    info.modelVariant = task.modelVariant;
    info.dimension = p.K;
    info.forcing = p.F;
    info.saturationKappa = p.kappa;
    info.equilibriumMaxAbsRhs = equilibriumMaxAbsRhs;
    info.rolloutInitialConditionSource = task.rollout.initialConditionDomain.source;

    fprintf('SoftSaturatedLorenz96 mapping verified: %s\n', ...
        task.variableMappingDescription);
    fprintf('K=%d | F=%.6g | kappa=%.6g | equilibrium max |RHS|=%.3e\n', ...
        p.K,p.F,p.kappa,equilibriumMaxAbsRhs);
    fprintf('Alternating joint-OOD static and rollout designs verified.\n');
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
