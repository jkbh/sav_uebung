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
%sound(y, fs);


%% Schritt 2

% Stimme laden
voice = load("./data/femalevoice.mat").female;

% Dictionary für lpc coeffs
lpcsPhonemDict = dictionary(string([]), cell([]));

for i = 1:10
    [x, fs] = audioread(['./data/female/', num2str(i, '%0.5d'), '.wav']);
    
    % Intervalle und dazugehörige phoneme laden
    voiceData = voice{2,i};
    numIntervals = size(voiceData, 1);    

    for j = 1:numIntervals
        interval = [voiceData{j, 1:2}];
        intervalSample = round(interval ./ 1e7 .* fs); % 100 nanoseconds to seconds to samples
        phonem = voiceData{j,3};
        
        lpcCoeffsPhonem = lpc(x(intervalSample(1) + 1:intervalSample(2)), order);
    
        % Build up dict with lpc coefficients for all phonem samples
        if isKey(lpcsPhonemDict, phonem)
            lpcsPhonemDict{phonem} =  cat(1, lpcsPhonemDict{phonem}, lpcCoeffsPhonem);
        else
            lpcsPhonemDict{phonem} =  lpcCoeffsPhonem;
        end        
    end
end

numPhonems = numEntries(lpcsPhonemDict);

% average over lpc coefficients
for i = 1:numPhonems
    phonems = keys(lpcsPhonemDict);
    phonem = phonems(i);
    lpcsPhonemDict{phonem} = mean(lpcsPhonemDict{phonem}, 1);
end

%% Schritt 3

phonems = ["jh", "ey", "k", "ah", "b"];
%phonems = ["m", "ae", "el", "f"];

outputLength = (length(phonems) - 1) * overlapSamples + frameLengthSamples;
output = zeros(outputLength, 1);

for i = 1:length(phonems)
    excitation = [1; zeros(frameLengthSamples - 1, 1)];
    voicedFrame = filter(1, lpcsPhonemDict{phonems(i)}, excitation);
    
    startIdx = (i - 1) * overlapSamples + 1;
    endIdx = startIdx + frameLengthSamples - 1;
    
    if endIdx > outputLength
        voicedFrame = voicedFrame(1:outputLength - startIdx + 1);
        output(startIdx:end) = output(startIdx:end) + voicedFrame;
    else
        output(startIdx:endIdx) = output(startIdx:endIdx) + voicedFrame;
    end
end

% Normalisierung des vokalisierten Signals auf den Bereich [-1, 1]
output = output / max(abs(output));

% Abspielen des vokalisierten Signals
sound(output, fs);