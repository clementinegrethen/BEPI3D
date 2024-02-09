clear;
close all;

load('model_ON.mat', 'imgs_ref_coul', 'imgs_lamb_coul', 'imgs_ON_coul', 'imgs_ON', 'rmse_tot', 'rmse', 'sigmas', 'sigma_pred', 'nb_iterations');
[nb_lignes, nb_colonnes, nb_images] = size(imgs_ON);
% On récupère s
dossier_light = '/Users/alexisgosselin/Documents/ENSEEIHT/3A/PI3D/BE/RTI/assembly-files/lumiere.txt';
s = zeros(nb_images, 3);
fid = fopen(dossier_light, 'r');
index_first_token = 2;
for i = 1:nb_images
    line = fgetl(fid);
    tokens = strsplit(line, ' ');
    s(i,:) = [str2double(tokens{index_first_token}); -str2double(tokens{index_first_token + 1}); -str2double(tokens{index_first_token + 2})];
end
fclose(fid);

%% Tracer la rmse par image

rmse_values = zeros(nb_images, 1);
% Calcul de la RMSE pour chaque image
for i = 1:nb_images
    ref_image = imgs_ref_coul(:, :, :, i);
    ON_image = imgs_ON_coul(:, :, :, i);
    rmse_values(i) = sqrt(mean((ref_image(:) - ON_image(:)).^2));
end
figure;
bar(rmse_values);
xlabel('Image Index');
ylabel('RMSE');
title('RMSE Values for Each Image');

%% Tracer de la rmse en fonction de sigma
rmse_min_sig = min(rmse_tot,[],2);
figure;
plot(sigmas, rmse_min_sig, '+-');
xlabel('sigma');
ylabel('RMSE');
title(sprintf('RMSE en fonction de sigma'));
grid on;
xticks(sigmas);

%% Calcul de la Rmse en fonction de l'angle par rapport au plan xy
angles_theta = zeros(1, nb_images);
% Calculer les angles theta pour chaque vecteur
for i = 1:nb_images
    vecteur = s(i, :);
    angle_original = atan2(norm(cross([0, 0, 1], vecteur)), dot([0, 0, 1], vecteur));
    angles_theta(i) = min(angle_original, pi - angle_original);
end
angles_theta_degrees = rad2deg(angles_theta);
% Affichage du graphique RMSE en fonction des angles
figure;
scatter(angles_theta_degrees, rmse_values, 'diamond', 'b');
xlabel('Angle theta (degrés)');
ylabel('RMSE');
title('RMSE en fonction de l''angle');
grid on;


% %% Tracer de la rmse en fonction du nombre d'itération
% iterations = 1:nb_iterations;
% for i = 1:length(sigmas)
%     figure;
%     plot(iterations, rmse_tot(i, :), '+-');
%     xlabel('Nombre d''itérations');
%     ylabel('RMSE');
%     title(sprintf('Évolution de la RMSE en fonction du nombre d''itérations - Sigma = %.2f', sigmas(i)));
%     grid on;
%     xticks(iterations);
%     waitforbuttonpress;
%     close;
% end

% %% Tracer des images Ref/Lamb/ON
% figure;
% for i = 1:nb_images
%     subplot(1,3,1);
%     imshow(imgs_ref_coul(:,:,:,i));
%     title('Ground truth');
%     subplot(1,3,2);
%     imshow(imgs_lamb_coul(:,:,:,i));
%     title('Lambertian');
%     subplot(1,3,3);
%     imshow(imgs_ON_coul(:,:,:,i));
%     title('Oren-Nayar');
%     % If character is pressed, go to next image
%     waitforbuttonpress;
% end

% %% Tracer des images Ref/ON
% figure;
% for i = 1:nb_images
%     subplot(1,2,1);
%     imshow(imgs_ref_coul(:,:,:,i));
%     title('Ground truth');
%     subplot(1,2,2);
%     imshow(imgs_ON_coul(:,:,:,i));
%     title('Oren-Nayar');
%     % If character is pressed, go to next image
%     waitforbuttonpress;
% end

%% Tracer des images Ref/ON/Diff Ref-ON
figure;
for i = 1:nb_images
    subplot(1,3,1);
    imshow(imgs_ref_coul(:,:,:,i));
    title('Ground truth', 'FontSize', 20);
    subplot(1,3,2);
    imshow(imgs_ON_coul(:,:,:,i));
    title('Oren-Nayar', 'FontSize', 20);

    diff_rgb = sqrt(sum((double(imgs_ref_coul(:,:,:,i)) - double(imgs_ON_coul(:,:,:,i))).^2, 3));

    % Afficher l'image avec une colormap de bleu à rouge
    subplot(1,3,3);
    imshow(diff_rgb);
    colormap(jet); 
    colorbar;
    title('Différence Référence/ON', 'FontSize', 20);

    % If character is pressed, go to next image
    waitforbuttonpress;
end

% %% Tracer des images Ref/Lamb/ON/Diff Ref-ON
% figure;
% for i = 1:nb_images
%     subplot(2,2,1);
%     imshow(imgs_ref_coul(:,:,:,i));
%     title('Ground truth', 'FontSize', 20);
%     subplot(2,2,2);
%     imshow(imgs_lamb_coul(:,:,:,i));
%     title('Lambertian', 'FontSize', 20);
%     subplot(2,2,3);
%     imshow(imgs_ON_coul(:,:,:,i));
%     title('Oren-Nayar', 'FontSize', 20);
% 
%     diff_rgb = sqrt(sum((double(imgs_ref_coul(:,:,:,i)) - double(imgs_ON_coul(:,:,:,i))).^2, 3));
% 
%     subplot(2,2,4);
%     imshow(diff_rgb);
%     colormap(jet);  % Vous pouvez ajuster la colormap selon vos préférences
%     colorbar;
%     title('Différence Référence/ON', 'FontSize', 20);
%     % If character is pressed, go to next image
%     waitforbuttonpress;
% end

% %% Tracer de la différence entre l'image de référence et le modèle d'Oren
% % Nayar puis entre le modèle lambertien et celui d'Oren Nayar
% figure;
% for i = 1:nb_images
%     diff_lamb = abs(double(imgs_ref_coul(:,:,:,i)) - double(imgs_ON_coul(:,:,:,i)));
%     subplot(1,2,1);
%     imshow(diff_lamb);
%     title('Difference: Ground Truth - Oren-Nayar');
% 
%     diff_ON = abs(double(imgs_lamb_coul(:,:,:,i)) - double(imgs_ON_coul(:,:,:,i)));
%     subplot(1,2,2);
%     imshow(diff_ON);
%     title('Difference: Lambertian - Oren-Nayar');
% 
%     waitforbuttonpress; 
% end