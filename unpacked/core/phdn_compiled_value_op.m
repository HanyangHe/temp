function y = phdn_compiled_value_op(op,varargin)
%PHDN_COMPILED_VALUE_OP Consolidated value-only operators for rollout inference.
%
% This helper is intentionally limited to the compiled arbitrary-state
% inference path. It replaces a collection of one-line operator files while
% preserving the same numerical semantics used by the previous patch.
%
% Supported op values:
%   square, cube, sqrt_abs, raw_inv,
%   inv_eps, sqrt_eps, exp_clip, asin_eps, log_eps

    switch lower(char(op))
        case 'square'
            x = varargin{1};
            y = x.^2;

        case 'cube'
            x = varargin{1};
            y = x.^3;

        case 'sqrt_abs'
            x = varargin{1};
            y = sqrt(abs(x));

        case 'raw_inv'
            x = varargin{1};
            y = 1./x;

        case 'inv_eps'
            x = varargin{1};
            epsVal = optional_eps_local(varargin,2);
            s = sign(x);
            s(s==0) = 1;
            y = 1./(s.*max(abs(x),epsVal));

        case 'sqrt_eps'
            x = varargin{1};
            epsVal = optional_eps_local(varargin,2);
            y = sqrt(max(x,epsVal));

        case 'exp_clip'
            x = varargin{1};
            y = exp(min(max(x,-50),50));

        case 'asin_eps'
            x = varargin{1};
            epsVal = optional_eps_local(varargin,2);
            y = asin(min(max(x,-1+epsVal),1-epsVal));

        case 'log_eps'
            x = varargin{1};
            epsVal = optional_eps_local(varargin,2);
            z = sign(x).*max(abs(x),epsVal);
            z(z==0) = epsVal;
            y = log(abs(z));

        otherwise
            error('phdn_compiled_value_op:UnknownOperator', ...
                'Unknown compiled value operator: %s',char(op));
    end
end

function epsVal = optional_eps_local(args,index)
    if numel(args) >= index && ~isempty(args{index})
        epsVal = args{index};
    else
        epsVal = 1e-8;
    end
end
