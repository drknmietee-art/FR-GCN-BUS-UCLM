function fold = fr_patientfolds(patients, k)
%FR_PATIENTFOLDS  Patient-grouped folds by greedy bin packing (Section 6.5).
%   Every image of a patient goes to the same fold, so no patient contributes
%   to both a training fold and the corresponding test fold.  Deterministic:
%   patients are ordered by image count (descending), ties broken by name, and
%   each is assigned to the currently lightest fold.
    if nargin < 2 || isempty(k), k = 5; end
    [uniq, ~, idx] = unique(patients);
    counts = accumarray(idx, 1);
    [~, ord] = sortrows([-counts, (1:numel(counts))']);
    load_ = zeros(k, 1);
    foldOf = zeros(numel(uniq), 1);
    for i = 1:numel(ord)
        [~, f] = min(load_);
        foldOf(ord(i)) = f;
        load_(f) = load_(f) + counts(ord(i));
    end
    fold = foldOf(idx);
end
