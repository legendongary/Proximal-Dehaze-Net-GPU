function res = dy_nnpsnr(x1, x2)

N = numel(x1);
R = x1(:) - x2(:);
S = sum(R.^2);
M = S / N;

res = -10 * log10(M);

end

