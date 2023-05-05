function C = gprcolor(m)
%
% SYNTAX:  C = gprcolor(m)
%
% This function produces a standard seismic red/white/blue colormap.
%
% by James Irving
% October, 2000

if nargin < 1, m = size(get(gcf,'colormap'),1); end
n = max(round(m/2),1);
x = (1:n)'/n;
e = ones(length(x),1);
r = [x; e];
g = [x; flipud(x)];
b = [e; flipud(x)];
C = [r g b];
while size(C,1) > m
   C(1,:) = [];
   if size(C,1) > m, C(size(C,1),:) = []; end
end
