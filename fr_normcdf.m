function p = fr_normcdf(z)
%FR_NORMCDF  Standard normal CDF.
    p = 0.5 * erfc(-z / sqrt(2));
end
