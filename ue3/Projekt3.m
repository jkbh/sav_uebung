clear all
close all

filename = "./audio_file.wav";
[x, fs1] = audioread(filename, "native");
x = double(x);

step_sizes = 1:300;
errors = zeros(size(step_sizes));
entropies = zeros(size(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    x_hat = round(x / step_size) * step_size;

    % Histogramm erstellen
    histogram = histcounts(x_hat, 'Normalization', 'probability');

    % Entropie berechnen
    entropies(i) = -sum(histogram .* log2(histogram), 'omitnan');
    errors(i) = norm(x - x_hat, 2);
end

figure;
plot(entropies, errors, 'o');
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;




%% Aufgabe 2

[x, fs] = audioread(filename, "native");
x = double(x);

%Analyse
M=8; %Mehr Filter -> Metallischerer Klang
channelizer = dsp.Channelizer(M); % DSP Toolbox
synthesizer = dsp.ChannelSynthesizer();

y = channelizer(x);

%Quantisierung

entropies2 = zeros(M, length(step_sizes));
errors2 = zeros(M, length(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    y_hat = round(y / step_size) * step_size;
    
    %Entropie und Error pro Band
    for band=1:M
        % Histogramm erstellen
        histogram = histcounts(real(y_hat(:,band)), 'Normalization', 'probability');
    
        % Entropie berechnen
        %entropies2(band, i) = -sum(histogram .* log2(histogram), 'omitnan');
        R= -sum(histogram .* log2(histogram), 'omitnan');
        %errors2(band, i) = norm(real(y(:,band)) - real(y_hat(:,band)), 2);
        D = norm(real(y(:,band)) - real(y_hat(:,band)), 2);
        DR{band}.D(i,1)=D;
        DR{band}.R(i,1)=R;
    end
end

figure;
for band=1:M
    plot(DR{band}.R(:,1), DR{band}.D(:,1), 'o');
    hold on;
end
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;

% Synthese
x_hat = double(synthesizer(y_hat));

% Normalize
%x_hat = (2*(double(x_hat) - double(min(x_hat)))/(double(max(x_hat)) - double(min(x_hat)))) - 1;
%sound(x_hat, fs);
    
%% Aufgabe 3
% Lagrange-Optimierung:
lambda(1)=0;
lambda(3)=1e4;
lambda(2)=(lambda(1)+lambda(3))/2;

% Für jedes lambda D, R und Z bestimmen:
for j=1:3
    R(j)=0;
    D(j)=0;
    Z(j)=0;
    for i=1:M
        [minZ,imin]=min(DR{i}.D+lambda(j)*DR{i}.R);
        Z(j)=Z(j)+minZ;
        R(j)=R(j)+DR{i}.R(imin);
        D(j)=D(j)+DR{i}.D(imin);
    end;
end;

Rbudget=4.5;

% Intervallschachtelung:
count=0;
while (abs(Z(3)-Z(1))/abs(Z(3)+Z(1))>2e-4)&(count<50)
    count=count+1;
    if R(2)>Rbudget
        lambda(1)=lambda(2);
        R(1)=R(2);
        D(1)=D(2);
        Z(1)=Z(2);
    else
        lambda(3)=lambda(2);
        R(3)=R(2);
        D(3)=D(2);
        Z(3)=Z(2);
    end;
    lambda(2)=(lambda(1)+lambda(3))/2;
    j=2;
    D(j)=0;
    R(j)=0;
    Z(j)=0;
    for i=1:M
        [minZ,imin]=min(DR{i}.D+lambda(j)*DR{i}.R);
        Z(j)=Z(j)+minZ;
        R(j)=R(j)+DR{i}.R(imin);
        D(j)=D(j)+DR{i}.D(imin);
    end;
end;

% Für lambda(3) die Qualisiererindizes ermitteln:
    j=3;
    D(j)=0;
    R(j)=0;
    for i=1:M
        [minZ,imin]=min(DR{i}.D+lambda(j)*DR{i}.R);
        Z(j)=Z(j)+minZ;
        R(j)=R(j)+DR{i}.R(imin);
        D(j)=D(j)+DR{i}.D(imin);
        iminZ(i)=imin;
    end;
    Rate=R(3)
    Quantisiererindizes=iminZ

for band=1:M
    QQ=step_sizes(iminZ(band));
    M=y(:,band);
    %round(y / step_size) * step_size
    result(:, band) = round(M/QQ)*QQ;
end;

s = double(synthesizer(result));
soundsc(s, fs);