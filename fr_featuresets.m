function S = fr_featuresets(rawdim)
%FR_FEATURESETS  Column indices per configuration.
%   The fuzzy cues always occupy the last three columns: membership, lower
%   approximation, boundary uncertainty.
    r   = 1:rawdim;
    mem = rawdim + 1; low = rawdim + 2; bnd = rawdim + 3;
    S.full  = [r mem low bnd];   % FR-GCN (proposed)
    S.base  =  r;                % plain GCN
    S.lean  = [r bnd];           % FR-GCN-lean (recommended)
    S.nomem = [r low bnd];       % ablation: no fuzzy membership
    S.nobnd = [r mem low];       % ablation: no rough boundary
end
