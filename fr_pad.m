function r = fr_pad(r, n)
%FR_PAD  Pad a result row to N columns with NaN.
    while numel(r) < n, r{end+1} = NaN; end %#ok<AGROW>
end
