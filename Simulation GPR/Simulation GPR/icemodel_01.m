close all, clear all, clc

%Création d'un modèle de type glace avec une interface courbée entre glace
%et bedrock, avec un PML, trois couches (air, glace et roche) et en
%déterminant dx, dz et dt (et implémantation de canals d'air et autres
%blocs)

% sans topographie, avec un bedrock, et une trajectoire � 5m de la surface
% --> celle-ci ce fait directement dans le FDTD
load icemodel_04.mat rxline rzline x z nx nz gxline

%% Create subsurface model                        

% Propriété électriques de l'air 
ep = ones(nx,nz);                                % [ø] Permitivité diélectrique relative
mu = ones(size(ep));                             % [ø] Perméabilité magnétique relative, toujours constante
sig = zeros(size(ep));                           % [S/m] Conductivité électrique

% Propriété électriques de la glace 
ep(:,z>=0) = 3.5;                               % Permitivité diélectrique de la glace
sig(:,z>=0) = 0.1e-3;                           % Conductivité électrique de la glace 

% set bedrock properties below picked surface
for i=1:nx
    for j=1:nz
        if   z(j) > rzline(i)            % pour la roche 
             ep(i,j)  = 6;
             sig(i,j) = 1e-3;
        end
    end
end


%% Initialisation de la matrice 
% Implémentation d'hétérogénéités
% Poches d'air
ep(x>=10 & x<=10.5,z>=20 & z<=20.5) = 1;
sig(x>=10 & x<=10.5,z>=20 & z<=20.5) = 0;

ep(x>=40 & x<=40.5,z>=3 & z<=3.5) = 1;
sig(x>=40 & x<=40.5,z>=3 & z<=3.5) = 0;

ep(x>=9 & x<=9.5,z>=4 & z<=4.5) = 1;
sig(x>=9 & x<=9.5,z>=4 & z<=4.5) = 0;

ep(x>=3 & x<=3.5,z>=14 & z<=14.5) = 1;
sig(x>=3 & x<=3.5,z>=14 & z<=14.5) = 0;

ep(x>=29 & x<=29.5,z>=18. & z<=18.5) = 1;
sig(x>=29 & x<=29.5,z>=18 & z<=18.5) = 0;

ep(x>=32 & x<=32.5,z>=9 & z<=9.5) = 1;
sig(x>=32 & x<=32.5,z>=9 & z<=9.5) = 0;

ep(x>=46 & x<=46.5,z>=12 & z<=12.5) = 1;
sig(x>=46 & x<=46.5,z>=12 & z<=12.5) = 0;

% Boulders 
ep(x>=16 & x<=16.2,z>=7 & z<=7.2) = 6;
sig(x>=16 & x<=16.2,z>=7 & z<=7.2) = 1e-3;

ep(x>=7 & x<=7.2,z>=5 & z<=5.2) = 6;
sig(x>=7 & x<=7.2,z>=5 & z<=5.2) = 1e-3;

ep(x>=21 & x<=21.2,z>=24 & z<=24.2) = 6;
sig(x>=21 & x<=21.2,z>=24 & z<=24.2) = 1e-3;

ep(x>=43 & x<=43.2,z>=10 & z<=10.2) = 6;
sig(x>=43 & x<=43.2,z>=10 & z<=10.2) = 1e-3;

ep(x>=24 & x<=24.2,z>=6 & z<=6.2) = 6;
sig(x>=24 & x<=24.2,z>=6 & z<=6.2) = 1e-3;

ep(x>=35 & x<=35.2,z>=20 & z<=20.2) = 6;
sig(x>=35 & x<=35.2,z>=20 & z<=20.2) = 1e-3;

ep(x>=25 & x<=25.2,z>=15 & z<=15.2) = 6;
sig(x>=25 & x<=25.2,z>=15 & z<=15.2) = 1e-3;

ep(x>=36 & x<=36.2,z>=4 & z<=4.2) = 6;
sig(x>=36 & x<=36.2,z>=4 & z<=4.2) = 1e-3;

%% trajectoire
txline_01 = (0:0.2:50)';
tzline_01 = -5*ones(size(txline_01));

%% Plot the electrical properties as a check before simulation

%close all

figure(2)
ax1 = subplot(2,1,1);
imagesc(x,z,ep');
colorbar; colormap(ax1,parula);
set(gca,'plotboxaspectratio',[2 1 1]);
title('Relative dielectric permittivity [-]');
xlabel('Position [m]'); ylabel('Depth [m]');
axis image
hold on 
%plot(txline_01,tzline_01,'r')

ax2 = subplot(2,1,2);
h = imagesc(x,z,log10(1000*sig'));
colorbar; colormap(ax2,parula);
set(h,'AlphaData',(~sig'==0));
set(gca,'plotboxaspectratio',[2 1 1]);
title('log10(Electrical conductivity [mS/m])');
xlabel('Position [m]'); ylabel('Depth [m]'); 
axis image
hold on 
%plot(txline_01,tzline_01,'r')
return

%% enregistrer le mod�le et ses variables pour le FDTD
save('icemodel_01.mat')
figurefile = ['model01_text',datestr(now,'mm-dd-HHMM'),'.jpg'];
print('-djpeg','-r300',figurefile);
