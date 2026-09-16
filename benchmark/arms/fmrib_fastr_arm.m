function fmrib_fastr_arm(request_json)
%FMRIB_FASTR_ARM Correct one whole recording with the original FMRIB FASTR.
%
%   Every decision this harness could make has already been made in Python and
%   handed over in the request: where every volume starts, which channels are
%   not EEG, and every FASTR parameter. The volume starts are the same array
%   the Python arm derives its geometry from, so a difference between the two
%   arms cannot be a difference in where they think the scanner fired.
%
%   The triggers are one per volume, which is what this implementation is built
%   for. Handed one trigger per acquisition group instead it suppresses nothing
%   measurable -- 1x against 284x on the same recording -- because it has no
%   notion of an acquisition slot, so averaging a group against its neighbours
%   in a multiband sequence averages different slice sets together. Driving it
%   per group to look like the Python arm would report that gap as a bad
%   implementation rather than as the missing capability it is.
%
%   No low-pass and no decimation are applied here. Each tool ships a different
%   output filter, and the benchmark applies one shared filter to every arm
%   afterwards so that the comparison is of corrections rather than of filters.

request = jsondecode(fileread(request_json));

if isempty(which('eeglab'))
    error('fmrib_fastr_arm:noEeglab', 'EEGLAB is not on the MATLAB path.');
end
eeglab('nogui');
if isempty(which('fmrib_fastr'))
    error('fmrib_fastr_arm:noPlugin', 'The FMRIB plug-in is not installed.');
end

[directory, stem, extension] = fileparts(request.raw_vhdr);
EEG = pop_loadbv(directory, [stem extension], [], [], false);

triggers = double(request.volume_triggers(:)') + 1;  % Python counts from zero
if any(diff(triggers) <= 0)
    error('fmrib_fastr_arm:triggers', 'Volume triggers must increase.');
end
if triggers(end) > EEG.pnts
    error('fmrib_fastr_arm:triggers', 'A volume trigger is past the recording.');
end

excluded = excluded_channel_indices(EEG, request.non_eeg_channels);

corrected = fmrib_fastr( ...
    EEG, ...
    0, ...                                  % no low-pass; the benchmark filters
    request.interpolation_factor, ...
    request.neighbor_count, ...
    triggers, ...
    0, ...                                  % volume triggers: one per repetition
    0, ...                                  % no adaptive noise cancellation
    0, ...                                  % no trigger correction
    0, ...
    0, ...
    request.pre_trigger_fraction, ...
    excluded, ...
    0);                                     % no residual PCA

write_binary(request.output_data, corrected.data);
metadata = struct( ...
    'sampling_rate_hz', double(EEG.srate), ...
    'channel_names', {{EEG.chanlocs.labels}}, ...
    'sample_count', size(corrected.data, 2), ...
    'first_volume_sample', triggers(1) - 1, ...
    'matlab_version', version, ...
    'eeglab_version', eeg_getversion());
write_json(request.output_metadata, metadata);
end


function indices = excluded_channel_indices(EEG, names)
%EXCLUDED_CHANNEL_INDICES Resolve channel names to indices, refusing unknowns.
labels = {EEG.chanlocs.labels};
names = cellstr(names);
indices = zeros(1, numel(names));
for k = 1:numel(names)
    found = find(strcmp(labels, names{k}));
    if numel(found) ~= 1
        error('fmrib_fastr_arm:channel', ...
            'Channel %s does not name exactly one channel.', names{k});
    end
    indices(k) = found;
end
end


function write_binary(path, data)
%WRITE_BINARY Store the corrected samples as channel-major float32.
handle = fopen(path, 'w');
if handle < 0
    error('fmrib_fastr_arm:write', 'Cannot write %s.', path);
end
closer = onCleanup(@() fclose(handle));
count = fwrite(handle, single(data'), 'single');
if count ~= numel(data)
    error('fmrib_fastr_arm:write', 'Wrote %d of %d samples.', count, numel(data));
end
end


function write_json(path, value)
%WRITE_JSON Store run metadata beside the samples it describes.
handle = fopen(path, 'w');
if handle < 0
    error('fmrib_fastr_arm:write', 'Cannot write %s.', path);
end
closer = onCleanup(@() fclose(handle));
fprintf(handle, '%s', jsonencode(value));
end
