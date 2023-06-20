% Laden der Audiodatei
filename = './data/female/00002.wav';
[x, fs] = audioread(filename); % Einlesen der Audiodatei
x = x(:, 1); % Falls das Audiosignal mehrere Kanäle hat, wählen Sie den gewünschten Kanal aus

% Parameter für den Vocoder
frameLength = 0.03; % Länge eines Frames in Sekunden
overlap = 0.5; % Überlappungsfaktor der Frames
order = 20; % Ordnung des LPC-Modells

% Fensterlänge und Überlappung in Abtastwerten berechnen
frameLengthSamples = round(frameLength * fs);
overlapSamples = round(frameLengthSamples * overlap);

% Teilen der Audiodaten in Frames
frames = buffer(x, frameLengthSamples, overlapSamples, 'nodelay');
numFrames = size(frames, 2);

% Initialisierung der Ausgabe
y = zeros((numFrames - 1) * overlapSamples + frameLengthSamples, 1);

% LPC-Berechnung und Synthese für jeden Frame
for i = 1:numFrames
    frame = frames(:, i);
    
    % Fensterung
    window = hamming(frameLengthSamples);
    windowedFrame = frame .* window;
    
    % Berechnung der LPC-Koeffizienten
    lpcCoeffs = lpc(windowedFrame, order);
    
    % Synthese des vokalisierten Signals
    %if mod(i, 2) == 0 % Gerade Frames: stimmlos (Rauschen)
    %    excitation = randn(frameLengthSamples, 1);
    %else % Ungerade Frames: stimmhaft (Impuls)
    %    excitation = [1; zeros(frameLengthSamples - 1, 1)];
    %end
    excitation = [1; zeros(frameLengthSamples - 1, 1)];
    voicedFrame = filter(1, lpcCoeffs, excitation);
    
    % Überlappung und Addition der Frames
    startIdx = (i - 1) * overlapSamples + 1;
    endIdx = startIdx + frameLengthSamples - 1;
    
    if endIdx > length(y)
        voicedFrame = voicedFrame(1:length(y)-startIdx+1);
        y(startIdx:end) = y(startIdx:end) + voicedFrame;
    else
        y(startIdx:endIdx) = y(startIdx:endIdx) + voicedFrame;
    end
end

% Normalisierung der Ausgabe auf den Bereich [-1, 1]
y = y / max(abs(y));

% Abspielen des vokalisierten Signals
sound(y, fs);
