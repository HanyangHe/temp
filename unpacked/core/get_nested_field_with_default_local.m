function val = get_nested_field_with_default_local(s, names, defaultVal)
%GET_NESTED_FIELD_WITH_DEFAULT_LOCAL Safe nested-struct lookup helper.
%
% This standalone helper fixes calls that refer to
% get_nested_field_with_default_local outside the file where it was originally
% defined as a local subfunction.  It intentionally keeps the same name and
% behavior: if any field in the requested path is missing or empty, return the
% supplied default value.

    if nargin < 3
        defaultVal = [];
    end
    val = defaultVal;
    if ~isstruct(s)
        return;
    end
    if ischar(names) || (isstring(names) && isscalar(names))
        names = cellstr(names);
    end
    if ~iscell(names) || isempty(names)
        return;
    end

    cur = s;
    for ii = 1:numel(names)
        name = names{ii};
        if isstring(name)
            name = char(name);
        end
        if ~isstruct(cur) || ~isfield(cur, name) || isempty(cur.(name))
            val = defaultVal;
            return;
        end
        cur = cur.(name);
    end
    val = cur;
end
