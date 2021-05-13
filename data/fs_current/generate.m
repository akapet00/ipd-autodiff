clear;

% constants
global MU_0;
global EPS_0;
global XI;
global W;
MU_0 = 1.25663706212e-6;                         %  permeability in H/m
EPS_0 = 8.8541878128e-12;                        %  permittivity in F/m
C = sqrt(1 / (MU_0 * EPS_0));                    %  speed of light in m/s
N = 51;                                          %  number of elements
V = 1;                                           %  voltage of the source
XI = [-0.8611363115940526, -0.3399810435848563, ...
        0.3399810435848563, 0.8611363115940526]; %  integration points
W = [0.3478548451374538, 0.6521451548625461, ...
        0.6521451548625461, 0.3478548451374538]; %  integration weights

% variables
frequencies = [3., 3.5, 6., 10., 15., 20., 30., 40., 60., 80., 100.];
dipole_scalers = 2;

% simulation
output = zeros((N + 1) * length(frequencies) * length(dipole_scalers), 7);
rel_idx = 1;
tic;
for f_idx = 1:length(frequencies)
    f = frequencies(f_idx) * 1e9;
    lambda = C / f;
    for ds_idx = 1:length(dipole_scalers)
        dipole_scaler = dipole_scalers(ds_idx);
        L = lambda / dipole_scaler;
        r = L / N / 10;
        [curr, x] = current(N, f, L, r, V);
        output(rel_idx:rel_idx+N, 1) = repelem(N, N + 1)';
        output(rel_idx:rel_idx+N, 2) = repelem(f, N + 1)';
        output(rel_idx:rel_idx+N, 3) = repelem(L, N + 1)';
        output(rel_idx:rel_idx+N, 4) = repelem(V, N + 1)';
        output(rel_idx:rel_idx+N, 5:7) = [x', real(curr)', imag(curr)'];
        rel_idx = rel_idx + N + 1;
    end
end
disp('Run finalized');
toc;

% save simulation
save('dataset.mat', 'output');