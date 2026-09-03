function [guesses,info] = make_lorenz96_unsaturated_initial_guesses(K,F,priorLevel)
%MAKE_LORENZ96_UNSATURATED_INITIAL_GUESSES Build G1--G3 L96 SR seeds.
%
% [GUESSES,INFO] = MAKE_LORENZ96_UNSATURATED_INITIAL_GUESSES(K,F,LEVEL)
% uses the true forcing F only when LEVEL='G3'. The true data-generating model
% is not changed by this helper.
%
%   G1 (theory level 0): no Stage-0 initial guesses.
%   G2 (theory level 2): x_{i-1}(x_{i+1}-x_{i-2}) - x_i.
%                         The forcing F and smooth saturation are unknown.
%   G3 (theory level 3): x_{i-1}(x_{i+1}-x_{i-2}) - x_i + F.
%                         Only the smooth saturation is unknown.
%
% A legacy two-argument call defaults to G3, preserving the earlier behavior.

    if nargin < 1 || isempty(K)
        error('K must be supplied explicitly.');
    end
    if nargin < 2 || isempty(F)
        error('The true Lorenz--96 forcing F must be supplied explicitly.');
    end
    if nargin < 3 || isempty(priorLevel)
        priorLevel = 'G3';
    end

    K = round(double(K));
    F = double(F);
    assert(isfinite(K) && K>=4, 'K must be an integer at least four.');
    assert(isfinite(F) && isscalar(F), 'F must be a finite scalar.');

    [label,theoreticalLevel,name,description,includesTransport, ...
        includesDamping,includesForcing] = resolve_prior_level_local(priorLevel);

    if strcmp(label,'G1')
        guesses = {};
    else
        guesses = cell(1,K);
        for i = 1:K
            im2 = mod(i-3,K)+1;
            im1 = mod(i-2,K)+1;
            ip1 = mod(i,K)+1;
            core = sprintf('x%d*(x%d-x%d)-x%d',im1,ip1,im2,i);
            if includesForcing
                guesses{i} = sprintf('%s+%.17g',core,F);
            else
                guesses{i} = core;
            end
        end
    end

    info = struct();
    info.label = label;
    info.theoreticalLevel = theoreticalLevel;
    info.name = name;
    info.description = description;
    info.includesTransport = includesTransport;
    info.includesDamping = includesDamping;
    info.includesForcing = includesForcing;
    info.trueForcing = F;
    if includesForcing
        info.priorForcing = F;
    else
        info.priorForcing = NaN;
    end
    info.nInitialGuesses = numel(guesses);
    info.dimension = K;
    info.scope = 'shared_all_unresolved_outputs';
end

function [label,theoreticalLevel,name,description,includesTransport, ...
        includesDamping,includesForcing] = resolve_prior_level_local(priorLevel)
    token = upper(strtrim(char(string(priorLevel))));
    switch token
        case {'G1','LEVEL0','L0','0','NONE','NO_PRIOR'}
            label = 'G1';
            theoreticalLevel = 0;
            name = 'no_initial_guess';
            description = 'No Stage-0 SR initial guesses.';
            includesTransport = false;
            includesDamping = false;
            includesForcing = false;
        case {'G2','LEVEL2','L2','2','UNKNOWN_FORCING'}
            label = 'G2';
            theoreticalLevel = 2;
            name = 'unsaturated_transport_and_damping_unknown_forcing';
            description = ['Ideal unsaturated cyclic transport and -x_i damping; ', ...
                'the forcing and smooth saturation are unknown.'];
            includesTransport = true;
            includesDamping = true;
            includesForcing = false;
        case {'G3','LEVEL3','L3','3','FULL_UNSATURATED'}
            label = 'G3';
            theoreticalLevel = 3;
            name = 'full_unsaturated_lorenz96';
            description = ['Complete ideal unsaturated Lorenz--96 law with the ', ...
                'true forcing; only the smooth saturation is unknown.'];
            includesTransport = true;
            includesDamping = true;
            includesForcing = true;
        otherwise
            error('Unknown Lorenz--96 Stage-0 prior level: %s. Use G1, G2, or G3.', ...
                char(string(priorLevel)));
    end
end
