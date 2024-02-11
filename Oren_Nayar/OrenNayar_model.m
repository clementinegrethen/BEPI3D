clear;
%% Images et Lights
dossier_images = 'RTI/jpeg-exports';
if ~ exist(dossier_images, 'dir') % Vérifie si le dossier image existe
    disp('Wrong path to images');
end
liste_fichiers = dir(fullfile(dossier_images, '*.JPG')); % Si JPG
% liste_fichiers = dir(fullfile(dossier_images, '*.png')); % Si png

dossier_light = 'Path/lumiere.txt';
if ~ exist(dossier_light, 'file') % Vérifie si le fichier light existe
    disp('Wrong path to light');
end
dossier_mask = '/mask.mat';
if ~exist(dossier_mask, 'file')
    disp('Wrong path to mask')
end

load(dossier_mask, 'mask') % Si mask = .mat
% mask = imread(dossier_mask); % Si mask = image RGB
% mask = mask(:,:,1) > 1; % Si mask = image RGB


fid = fopen(dossier_light, 'r');
index_first_token = 2;

% On récupère la première image pour obtenir les dimensions des images
nom_fichier = fullfile(dossier_images, liste_fichiers(1).name);
info_image = imfinfo(nom_fichier);

% Dimension
% nb_lignes = info_image.Height;
% nb_colonnes = info_image.Width;
nb_lignes = info_image.Height / 4; % Dimension /4 
nb_colonnes = info_image.Width / 4; % Dimension /4
nb_images = length(liste_fichiers);
nb_pixels = nb_lignes * nb_colonnes;

% Initialisation
I_coul = zeros([nb_lignes, nb_colonnes, 3, nb_images]);
S = zeros(nb_images, 3);

% On récupère les direction de light et toutes les images
for i = 1:nb_images
    line = fgetl(fid);
    tokens = strsplit(line, ' ');
    nom_fichier = fullfile(dossier_images, liste_fichiers(i).name);
    imageCourante = imread(nom_fichier);
    I_coul(:, :, :, i) =  double(imresize(imageCourante, [nb_lignes,nb_colonnes]))/255;
    S(i,:) = [str2double(tokens{index_first_token}); -str2double(tokens{index_first_token + 1}); -str2double(tokens{index_first_token + 2})];
end
fclose(fid); 

% On vectorise l'image de façon à avoir nb_pixels x nb_images x 3
I_coul_vect = permute(reshape(I_coul, [nb_pixels 3 nb_images]), [1, 3, 2]); % nb_pixels x nb_images x 3

% Création de l'image en noir et blanc
I_vect = zeros(nb_pixels, nb_images);
for i = 1:nb_images
    I_vect(:, i) = reshape(rgb2gray(I_coul(:, :, :, i)), nb_pixels, 1);
end

%% Paramètres
sigmas = 0:0.05:1;
nb_iterations = 5;

%% Masking
mask_im = imresize(mask, [nb_lignes nb_colonnes]); % Resize the mask to the image size
%mask_im = ones(nb_lignes, nb_colonnes);
mask = mask_im(:); % Vectorize the mask (nb_pixels x 1)
[in_mask, ~] = find(mask); % Find pixels inside the mask
I_vect = I_vect .* mask;

%% Variables
s = S'; 

S_plus = inv(S'*S)*S'; % 3 * nb_images
s_bar = s ./ vecnorm(s); % Attention on load peut-être déjà s normalisé
% v = [0 0 -1];
% v = repmat(v', [1 nb_pixels]);

[x, y] = meshgrid(1:nb_colonnes, 1:nb_lignes);
% Normaliser et centrer les coordonnées x et y
x_normalized = - (x - nb_colonnes/2) / (nb_colonnes/2);
y_normalized = (y - nb_lignes/2) / (nb_lignes/2);
% Concaténer x, y et -1
v = [x_normalized(:), y_normalized(:), -ones(nb_lignes * nb_colonnes, 1)]';
% Normaliser les vecteurs de vue
v = v ./ sqrt(sum(v.^2, 1));

% Initialisation de nb_images avec modèle lambertien -> nb_images = S_plus * I_vect'
m0 = S_plus * I_vect'; % 3 * nb_pixels
rho_0 = vecnorm(m0); % 1 * nb_pixels
n0 = m0 ./ rho_0; % 3*nb_pixels

% Initialisation des matrices à 0
rmse_tot = zeros(length(sigmas),nb_iterations);
I_pred = zeros(size(I_vect)); % Images prédites avec sigma courant
I_pred_vect = zeros(size(I_vect)); % Images finalement prédites
Z = zeros(size(I_vect(:,1)));
rmse_min = rmse(I_vect, (s' * m0)',[1 2]); % Lambertien, sigma = 0
rmse_tot(1,:) = rmse_min;
sigma_pred = 0.0; % Lambertien
n = n0;

% Initialisation des paramètres
rho_k = rho_0;
nk = n0;

for i_sig = 2:length(sigmas)
    sigma = sigmas(i_sig);
    fprintf("sigma : %f \n", sigma)
    for iter = 1:nb_iterations
        fprintf("iter : %d \n", iter)
        mk1 = Calcul_mk1(I_vect,S_plus,s_bar,nk,v,sigma);
        rho_k = vecnorm(mk1);
        nk = mk1 ./ rho_k;
        % Calcul RMSE
        for i=1:nb_images
            I_pred(:,i) = max(Z,Xon(s_bar(:,i),nk,v,sigma) .* (S(i,:) * mk1)'); % taille 1*nb_pixels
        end
        I_pred = I_pred .* mask;
        rmse_sigma = rmse(I_vect,I_pred(:,:),[1 2]);
        rmse_tot(i_sig, iter) = rmse_sigma;
    end
    if rmse_sigma <= rmse_min
        rmse_min = rmse_sigma;
        I_pred_vect = I_pred;
        sigma_pred = sigma;
        n = nk;
    end
end

%% Garder n calculer et recalculer l'albédo pour chaque canal RGB
I_pred_vect_coul = zeros(size(I_coul_vect));
I_lamb_vect_coul = zeros(size(I_coul_vect));
I_coul_vect = I_coul_vect .* mask;
m_coul = zeros(3,nb_pixels,3);
for couleur = 1:3
    m = Calcul_mk1(I_coul_vect(:,:,couleur),S_plus,s_bar,n,v,sigma_pred);
    m_coul(:,:,3) = m;
    m0 = S_plus * I_coul_vect(:,:,couleur)'; % 3 * nb_pixels
    for i=1:nb_images
        I_pred_vect_coul(:,i,couleur) = max(Z,Xon(s_bar(:,i),n,v,sigma_pred) .* (S(i,:) * m)');
        I_lamb_vect_coul(:,i,couleur) = max(Z, (S(i,:) * m0)');
    end
end

% retour aux images de taille originale
imgs_ON = reshape(I_pred_vect, [nb_lignes, nb_colonnes, nb_images]);
imgs_ON_coul = permute(reshape(I_pred_vect_coul, [nb_lignes, nb_colonnes, nb_images, 3]), [1, 2, 4, 3]); % nb_lignes x nb_colonnes x 3 x nb_images
imgs_lamb_coul = permute(reshape(I_lamb_vect_coul, [nb_lignes, nb_colonnes, nb_images, 3]), [1, 2, 4, 3]);

model_ON.m_coul = m_coul;
model_ON.n = n;
model_ON.v = v;
model_ON.sigmas = sigmas;
model_ON.nb_iterations = nb_iterations;
model_ON.rmse = rmse_min;
model_ON.sigma_pred = sigma_pred;
model_ON.rmse_tot = rmse_tot;
model_ON.imgs_ref_coul = I_coul.*mask_im;
model_ON.imgs_ON = imgs_ON;
model_ON.imgs_ON_coul = imgs_ON_coul;
model_ON.imgs_lamb_coul = imgs_lamb_coul;
save('model_ON.mat','-struct','model_ON', '-v7.3')