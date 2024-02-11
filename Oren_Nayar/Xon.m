function Xon_im = Xon(s_bar,n,v,sigma)
%INPUT:
%   s_bar is the light direction 3 x 1
%   n is the normal for every pixels 3 x nb_pixels
%   v is the view direction for every pixels 3 x nb_pixels
%   sigma is a term to estimate which define rugosity of the surface
%OUTPUT:
%   Xon_im is the corrective factor for reflection 1 x nb_pixels 
%   (= 1 if sigma = 0 <=> lambertian surface)
%   1 x nb_pixels

A = 1 - (sigma^2 / 2*(sigma^2 + 0.57)); % size 1x1

gamma = dot((s_bar - (s_bar' * n).*n),(v - dot(v,n).*n)); % size 1 x nb_pixels

acos_sn = acos(s_bar'*n); % size 1*nb_pixels
acos_vn = acos(dot(v,n)); % size 1*nb_pixels
B = (0.45*sigma^2*max(0,gamma))/(sigma^2 + 0.09); % size 1 x nb_pixels
C = sin(max(acos_sn,acos_vn)).*tan(min(acos_sn,acos_vn)); % size 1*nb_pixels
Xon_im = (A + B.*C)'; % nb_pixels * 1
end

