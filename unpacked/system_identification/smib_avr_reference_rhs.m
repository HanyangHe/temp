function dX = smib_avr_reference_rhs(X, p)
%SMIB_AVR_REFERENCE_RHS Explicit salient-pole SMIB + linear-AVR ODE.
%
%   dot(delta)  = omega_b*domega
%
%   Pe          = Eqp*Vinf/(Xdp+Xe)*sin(delta)
%                 + Ksal*sin(2*delta)
%
%   dot(domega) = [Pm - Pe - D*domega]/(2H)
%
%   dot(Eqp)    = [-(Xd+Xe)/(Xdp+Xe)*Eqp
%                  +(Xd-Xdp)/(Xdp+Xe)*Vinf*cos(delta)+Efd]/TdoPrime
%
%   Vq          = [Xe*Eqp + Xdp*Vinf*cos(delta)]/(Xdp+Xe)
%   Vd          = Xq*Vinf*sin(delta)/(Xq+Xe)
%   Vt          = sqrt(Vq^2+Vd^2)
%   ev          = Vref-Vt
%   VA          = KA*ev
%   dot(Efd)    = [VA-(Efd-Efd0)]/TA
%
% X is N-by-4 and dX is N-by-4. All conventional derivative coefficients
% have been divided into the RHS. The terminal-voltage magnitude remains
% compositionally nested and is not exactly represented by a generic flat
% one-layer dictionary unless target-specific composite columns are supplied.

    if nargin < 2 || isempty(p)
        p = single_generator_dynamic_parameters();
    end
    validateattributes(X, {'numeric'}, {'2d','ncols',4,'real','finite'}, ...
        mfilename, 'X');

    delta = X(:,1);
    domega = X(:,2);
    Eqp = X(:,3);
    Efd = X(:,4);

    Pe = (Eqp*p.Vinf/p.XdNet).*sin(delta) ...
        +p.Ksal.*sin(2.*delta);

    Vq = (p.Xe.*Eqp + p.Xdp*p.Vinf.*cos(delta))./p.XdNet;
    Vd = (p.Xq*p.Vinf.*sin(delta))./p.XqNet;
    Vt = sqrt(Vq.^2 + Vd.^2);

    voltageError = p.Vref - Vt;
    amplifierVoltage = p.KA.*voltageError;

    deltaDot = p.omegaBase.*domega;
    domegaDot = (p.Pm - Pe - p.D.*domega)./(2*p.H);
    EqpDot = ( ...
        -((p.Xd + p.Xe)/p.XdNet).*Eqp ...
        +((p.Xd - p.Xdp)*p.Vinf/p.XdNet).*cos(delta) ...
        +Efd)./p.TdoPrime;
    EfdDot = (amplifierVoltage - (Efd - p.Efd0))./p.TA;

    dX = [deltaDot, domegaDot, EqpDot, EfdDot];
    if any(~isfinite(dX(:)))
        error('Salient-pole SMIB linear-AVR RHS produced NaN or Inf values.');
    end
end
