function v = fr_get(s, f)
%FR_GET  Field value or NaN when the field is absent.
    if isstruct(s) && isfield(s, f), v = s.(f); else, v = NaN; end
end
