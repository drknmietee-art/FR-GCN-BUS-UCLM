function s = fr_seed(k)
%FR_SEED  Initialise the deterministic generator used throughout FR-GCN.
%   A self-contained xorshift32 stream is used in place of MATLAB's RandStream
%   so that every result in the paper is bit-reproducible on any MATLAB or
%   Octave version, independent of the platform's default RNG.
    s = uint32(mod(double(k) * 2654435761 + 1, 4294967291) + 1);
end
