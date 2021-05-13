function [g] = green(z_tar, z_src, r, k)
% Green's function for the Pocklington equation.
% Returns the value of the Green's function.
%
% Arguments
%   z_tar : observation point in free space
%   z_src : source point
%   r : dipole radius [m]
%   k : propagation constant [m^-1]
%
% Returns
%   g : value of the Green's function

    R = sqrt((z_tar - z_src) .^ 2 + r .^ 2);
    g = exp(-1i * k .* R) ./ R;
end