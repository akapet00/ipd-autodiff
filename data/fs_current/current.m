function [curr, x] = current(N, f, L, r, V)
% Finite element Pocklington integro-differential equation solver.
% Returns the current distribution over a dipole
%
% Arguments
%   N : number of finite elements
%   f : frequency of the source [Hz]
%   L : dipole length [m]
%   r : dipole radius [m]
%   V : voltage [V]
%
% Returns
%   g : value of the Green's function
    
    global MU_0;
    global EPS_0;
    global XI;
    global W;
    omega = 2 * pi * f;
    k = omega * sqrt(MU_0 * EPS_0);

    dLi = L / N;
    dLj = L / N;
    x = (0:dLi:L);

    N1 = (1 - XI) / 2;
    N2 = (1 + XI) / 2;
    D1 = -(1 / dLj) * ones(1, length(XI));
    D2 = (1 / dLi) * ones(1, length(XI));

    B = zeros(N + 1, 1);
    B((N+1)/2:(N+1)/2+1) = -1i * (4 * pi) * EPS_0 * omega * V / 2;

    A = zeros(N+1);

    for j=1:N
        xj1 = 0 + dLj * (j-1);
        xj2 = dLj + xj1;
        xj = xj1 * N1 + xj2 * N2;
        for i=1:N
            xi1 = 0 + dLi * (i - 1);
            xi2 = dLi + xi1;
            xi = xi1 * N1 + xi2 * N2;
            G = zeros(length(XI));
            a = zeros(2);
            for n=1:length(XI) 
                for m=1:length(XI)
                    G(n, m) = fs_green(xj(n), xi(m), r, k);
                end 
            end
            a(1, 1) = k ^ 2 * (N1 .* W) * ((N1 .* W * G.').') ...
                - (D1 .* W) * ((D1 .* W * G.').');
            a(1, 2) = k ^ 2 * (N1 .* W) * ((N2 .* W * G.').') ...
                - (D1 .* W) * ((D2 .* W * G.').');
            a(2, 1) = k ^ 2 * (N2 .* W) * ((N1 .* W * G.').') ...
                - (D2 .* W) * ((D1 .* W *G.').');
            a(2, 2) = k^2 * (N2 .* W) * ((N2 .* W *G.').') ...
                - (D2 .* W) * ((D2 .* W * G.').');      
            a = (dLi * dLj) / 4 * a;
            A(j:j+1, i:i+1) = A(j:j+1, i:i+1) + a;
        end
    end

    C = A(2:N, 2:N) \ B(2:N);
    curr = zeros(1, N+1);
    curr(1, 2:N) = C.';
end