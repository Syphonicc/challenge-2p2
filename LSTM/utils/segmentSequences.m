function segmentedCell = segmentSequences(dataCell, windowSize, overlap)
    % Validate inputs
    if overlap >= windowSize
        error('Overlap must be smaller than window size.');
    end

    step = windowSize - overlap;
    segmentedCell = {};  % initialize output cell array
    idx = 1;

    for ii = 1:numel(dataCell)
        sequence = dataCell{ii}; % matrix of size [numTimeSteps x numFeatures]
        [seqLength, numFeatures] = size(sequence);

        % Slide window
        for startIdx = 1:step:(seqLength - windowSize + 1)
            endIdx = startIdx + windowSize - 1;
            segmentedCell{idx} = sequence(startIdx:endIdx, :); % [windowSize x numFeatures]
            idx = idx + 1;
        end
    end
end
