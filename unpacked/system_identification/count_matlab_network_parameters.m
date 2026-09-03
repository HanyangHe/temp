function n = count_matlab_network_parameters(net)
%COUNT_MATLAB_NETWORK_PARAMETERS Count stored weights and biases in a network.
    n = 0;
    for i = 1:numel(net.IW)
        if ~isempty(net.IW{i}); n = n + numel(net.IW{i}); end
    end
    for i = 1:numel(net.LW)
        if ~isempty(net.LW{i}); n = n + numel(net.LW{i}); end
    end
    for i = 1:numel(net.b)
        if ~isempty(net.b{i}); n = n + numel(net.b{i}); end
    end
end
