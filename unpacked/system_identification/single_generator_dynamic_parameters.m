function p = single_generator_dynamic_parameters()
%SINGLE_GENERATOR_DYNAMIC_PARAMETERS Salient-pole SMIB + linear-AVR parameters.
%
% State convention:
%   x1 = delta   : rotor electrical angle relative to the infinite bus [rad]
%   x2 = domega  : per-unit rotor-speed deviation
%   x3 = Eqp     : q-axis transient internal voltage E'_q [p.u.]
%   x4 = Efd     : field/exciter voltage [p.u.]
%
% Benchmark-specific nonlinearities:
%   1) salient-pole reluctance power proportional to sin(2*delta);
%   2) unequal d/q network reactances in the terminal-voltage magnitude;
%   3) a linear AVR amplifier VA = KA*ev, ev = Vref - Vt.
%
% The operating point is constructed exactly so all four right-hand sides
% vanish at p.xEquilibrium. No hidden left-side coefficients remain in the
% task equations; H, T'_do, and T_A are divided into the explicit RHS.

    p = struct();
    p.modelVariant = 'salient_pole_flux_decay_linear_avr';

    p.fNominal = 60;                 % Hz
    p.omegaBase = 2*pi*p.fNominal;   % electrical rad/s
    p.H = 3.5;                       % s
    p.D = 5.0;                       % p.u. damping coefficient

    p.Xd = 1.8;                      % p.u. synchronous d-axis reactance
    p.Xdp = 0.30;                    % p.u. transient d-axis reactance
    p.Xq = 1.70;                     % p.u. synchronous q-axis reactance
    p.Xe = 0.40;                     % p.u. external/network reactance
    p.TdoPrime = 8.0;                % s

    % Linear AVR amplifier.
    p.KA = 10.0;
    p.TA = 0.10;                     % s
    p.Vinf = 1.0;                    % p.u.

    % Selected stable nominal operating point.
    p.delta0 = 0.50;                 % rad
    p.Eqp0 = 1.10;                   % p.u.

    p.XdNet = p.Xdp + p.Xe;
    p.XqNet = p.Xq + p.Xe;
    % Backward-compatible alias used by older case-local reporting code.
    p.Xsum = p.XdNet;

    % Salient-pole reluctance-power coefficient. The complete electrical
    % power is Eqp*Vinf/XdNet*sin(delta) + Ksal*sin(2*delta).
    p.Ksal = 0.5*p.Vinf^2*(1/p.XqNet - 1/p.XdNet);
    p.Pm = p.Eqp0*p.Vinf*sin(p.delta0)/p.XdNet ...
        +p.Ksal*sin(2*p.delta0);

    aFlux = (p.Xd + p.Xe)/p.XdNet;
    bFlux = (p.Xd - p.Xdp)*p.Vinf/p.XdNet;
    p.Efd0 = aFlux*p.Eqp0 - bFlux*cos(p.delta0);

    p.Vt0 = smib_terminal_voltage_local(p.delta0, p.Eqp0, p);
    p.Vref = p.Vt0;
    p.xEquilibrium = [p.delta0, 0, p.Eqp0, p.Efd0];

    % Keep physical names separate from the canonical one-based symbols used
    % by the Stage-0 SR/PhDN interface.
    p.stateNames = {'delta','domega','Eqp','Efd'};
    p.srVariableNames = {'x1','x2','x3','x4'};
    p.derivativeNames = {'delta_dot','domega_dot','Eqp_dot','Efd_dot'};
    p.stateNameMap = struct( ...
        'sr', p.srVariableNames, ...
        'physical', p.stateNames);
end

function Vt = smib_terminal_voltage_local(delta, Eqp, p)
    % Salient-pole terminal voltage: q- and d-axis components use different
    % effective network reactances. Setting Xq=Xdp reduces this to the older
    % round-rotor expression.
    Vq = (p.Xe.*Eqp + p.Xdp*p.Vinf.*cos(delta))./p.XdNet;
    Vd = (p.Xq*p.Vinf.*sin(delta))./p.XqNet;
    Vt = sqrt(Vq.^2 + Vd.^2);
end
