function [Xi, activeMask] = stlsq_multioutput(Theta, Y, threshold, options)
%STLSQ_MULTIOUTPUT Sequential thresholded least squares for shared dictionary.

    Xi = solve_local(Theta, Y, options.ridgeLambda);
    if threshold <= 0
        activeMask = abs(Xi) > 0;
        return;
    end
    for iter = 1:options.maxSTLSQIter
        previous = abs(Xi) >= threshold;
        Xi(~previous) = 0;
        for k = 1:size(Y,2)
            keep = previous(:,k);
            if any(keep)
                Xi(keep,k) = solve_local(Theta(:,keep), Y(:,k), options.ridgeLambda);
            else
                Xi(:,k) = 0;
            end
        end
        current = abs(Xi) >= threshold;
        if isequal(current, previous)
            break;
        end
    end
    Xi(abs(Xi) < threshold) = 0;
    activeMask = abs(Xi) > 0;
end

function Xi = solve_local(Theta, Y, lambda)
%SOLVE_LOCAL Stable least-squares solve for wide/rank-deficient dictionaries.
%
% Evolved genes can duplicate or become nearly collinear with fixed SINDy
% terms. A direct backslash solve may therefore emit singular-matrix/RCOND
% warnings and can return unstable coefficients. Ridge systems are solved by
% an augmented least-squares problem, while the unregularized case uses the
% minimum-norm SVD solver when available.

    if isempty(Theta)
        Xi = zeros(0,size(Y,2));
        return;
    end

    if any(~isfinite(Theta(:))) || any(~isfinite(Y(:)))
        Xi = NaN(size(Theta,2), size(Y,2));
        return;
    end

    nTerms = size(Theta,2);
    if lambda > 0
        lambda = max(double(lambda), 0);
        A = [Theta; sqrt(lambda) * eye(nTerms)];
        B = [Y; zeros(nTerms,size(Y,2))];
        Xi = stable_min_norm_local(A, B);
    else
        Xi = stable_min_norm_local(Theta, Y);
    end
end

function Xi = stable_min_norm_local(A, B)
% Use LSQMINNORM when available; otherwise use an explicit SVD pseudoinverse.
    if exist('lsqminnorm', 'file') == 2 || exist('lsqminnorm', 'builtin') == 5
        Xi = lsqminnorm(A, B);
        return;
    end

    [U,S,V] = svd(A, 'econ');
    singularValues = diag(S);
    if isempty(singularValues)
        Xi = zeros(size(A,2),size(B,2));
        return;
    end
    tolerance = max(size(A)) * eps(max(singularValues));
    keep = singularValues > tolerance;
    if ~any(keep)
        Xi = zeros(size(A,2),size(B,2));
    else
        Xi = V(:,keep) * ((U(:,keep)' * B) ./ singularValues(keep));
    end
end
