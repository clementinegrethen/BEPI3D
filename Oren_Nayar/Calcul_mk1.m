function mk1 = Calcul_mk1(I_vect,S_plus,s_bar,nk,v,sigma)
%INPUT:
%   I_vect is the vectorized vector of the nb_images images nb_pixels*nb_images
%   s is the light direction 3 x nb_images
%   s_bar is the light direction 3 x nb_images
%   n is the normal for every pixels 3 x nb_pixels
%   v is the view direction for every pixels 3 x nb_pixels
%   sigma is a term to estimate which define rugosity of the surface
%OUTPUT:
%   mk1 is mk+1 3 x nb_pixels
[~,nb_images] = size(I_vect);
M = zeros(size(I_vect'));
for i = 1:nb_images
    M(i,:) = I_vect(:,i) ./ Xon(s_bar(:,i),nk,v,sigma);
end
% Attention dim de S+, possiblement transposition
mk1 = S_plus * M; 
end

