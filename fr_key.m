function k = fr_key(name)
%FR_KEY  Filename -> valid struct field name (portable across MATLAB/Octave).
    k = regexprep(name, '[^A-Za-z0-9]', '_');
    if isempty(k) || ~isletter(k(1)), k = ['x' k]; end
end
