function dataStruct = load_variable_struct_from_h5(filename, variableName)
% Helper: loads all a/f0p combinations for a given variable from an HDF5 file
info = h5info(filename);
dataStruct = struct();

for i = 1:numel(info.Groups)
    aName = erase(info.Groups(i).Name, '/');
    subGroups = info.Groups(i).Groups;

    for j = 1:numel(subGroups)
        fName = strsplit(subGroups(j).Name, '/');
        fName = fName{end};
        datasetPath = sprintf('/%s/%s/%s', aName, fName, variableName);
        try
            dataStruct.(aName).(fName) = h5read(filename, datasetPath);
        catch
            warning('Missing dataset %s', datasetPath);
            dataStruct.(aName).(fName) = [];
        end
    end
end
end
