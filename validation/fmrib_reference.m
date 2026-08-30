function fmrib_reference(input_vhdr, output_mat, channels, first_volume, ...
    volume_count, lowpass_hz, interpolation_factor, window, anc_enabled, ...
    pre_trigger_fraction, excluded_channels, obs_rank)
%FMRIB_REFERENCE Run the original FMRIB FASTR on a bounded BrainVision span.

validate_inputs(input_vhdr, output_mat, channels, first_volume, ...
    volume_count, lowpass_hz, interpolation_factor, window, anc_enabled, ...
    pre_trigger_fraction, excluded_channels, obs_rank);
require_function('eeglab');
eeglab('nogui');
require_function('pop_loadbv');
require_function('fmrib_fastr');

[input_directory, input_stem, input_extension] = fileparts(input_vhdr);
input_name = [input_stem input_extension];
metadata = pop_loadbv(input_directory, input_name, [], channels, true);
volume_positions = extract_volume_positions(metadata.event);
last_volume = first_volume + volume_count - 1;
if last_volume > numel(volume_positions)
    error('Requested volume range exceeds the available volume markers.');
end

selected_positions = volume_positions(first_volume:last_volume);
samples_per_volume = round(median(diff(selected_positions)));
if any(abs(diff(selected_positions) - samples_per_volume) > 1)
    error('Selected volume markers are not contiguous.');
end
sample_start = max(1, selected_positions(1) - 2 * samples_per_volume);
sample_stop = min(metadata.pnts, selected_positions(end) + 3 * samples_per_volume);

EEG = pop_loadbv( ...
    input_directory, ...
    input_name, ...
    [sample_start sample_stop], ...
    channels, ...
    false);
triggers = selected_positions - sample_start + 1;
raw_data = double(EEG.data);
corrected = fmrib_fastr( ...
    EEG, ...
    lowpass_hz, ...
    interpolation_factor, ...
    window, ...
    triggers, ...
    0, ...
    anc_enabled, ...
    0, ...
    0, ...
    0, ...
    pre_trigger_fraction, ...
    excluded_channels, ...
    obs_rank);

corrected_data = double(corrected.data);
sampling_rate = double(EEG.srate);
channel_names = {EEG.chanlocs.labels};
sample_start_zero_based = sample_start - 1;
parameters = struct( ...
    'first_volume', first_volume, ...
    'volume_count', volume_count, ...
    'lowpass_hz', lowpass_hz, ...
    'interpolation_factor', interpolation_factor, ...
    'window', window, ...
    'anc_enabled', logical(anc_enabled), ...
    'pre_trigger_fraction', pre_trigger_fraction, ...
    'excluded_channels', excluded_channels, ...
    'obs_rank', obs_rank);
matlab_version = version;
fmrib_reference_commit = '2aa522bc5ec4215f42b3ba8efdb2b84d2a312935';
data_unit = 'microvolts';
save(output_mat, 'raw_data', 'corrected_data', 'triggers', ...
    'sampling_rate', 'channel_names', 'sample_start_zero_based', ...
    'parameters', 'matlab_version', 'fmrib_reference_commit', ...
    'data_unit', '-v7');
end


function positions = extract_volume_positions(events)
if isempty(events)
    error('The recording has no events.');
end
volume_events = strcmp({events.type}, 'V  1');
positions = round([events(volume_events).latency]);
if numel(positions) < 2 || any(diff(positions) <= 0)
    error('At least two ordered BrainVision V  1 events are required.');
end
end


function require_function(name)
if isempty(which(name))
    error('Required MATLAB function is not on the path: %s', name);
end
end


function validate_inputs(input_vhdr, output_mat, channels, first_volume, ...
    volume_count, lowpass_hz, interpolation_factor, window, anc_enabled, ...
    pre_trigger_fraction, excluded_channels, obs_rank)
if ~isfile(input_vhdr)
    error('Input BrainVision header does not exist.');
end
if isfile(output_mat)
    error('Output MAT file already exists.');
end
if isempty(channels) || any(channels < 1) || any(mod(channels, 1) ~= 0) ...
        || numel(unique(channels)) ~= numel(channels)
    error('Channels must be unique positive integers.');
end
validate_positive_integer(first_volume, 'First volume');
validate_positive_integer(volume_count, 'Volume count');
validate_nonnegative_number(lowpass_hz, 'Low-pass frequency');
validate_positive_integer(interpolation_factor, 'Interpolation factor');
validate_positive_integer(window, 'Window');
if mod(window, 2) ~= 0
    error('Window must be even.');
end
if ~isscalar(anc_enabled) || ~ismember(anc_enabled, [0 1])
    error('ANC enabled must be zero or one.');
end
if ~isscalar(pre_trigger_fraction) || ~isfinite(pre_trigger_fraction) ...
        || pre_trigger_fraction < 0 || pre_trigger_fraction > 1
    error('Pre-trigger fraction must lie between zero and one.');
end
if any(excluded_channels < 1) || any(excluded_channels > numel(channels)) ...
        || any(mod(excluded_channels, 1) ~= 0)
    error('Excluded channels must index the selected channel list.');
end
if ischar(obs_rank) || isstring(obs_rank)
    if ~strcmpi(obs_rank, 'auto')
        error('OBS rank string must be auto.');
    end
elseif ~isscalar(obs_rank) || ~isfinite(obs_rank) || obs_rank < 0 ...
        || mod(obs_rank, 1) ~= 0
    error('OBS rank must be auto or a nonnegative integer.');
end
end


function validate_positive_integer(value, name)
if ~isscalar(value) || ~isfinite(value) || value < 1 || mod(value, 1) ~= 0
    error('%s must be a positive integer.', name);
end
end


function validate_nonnegative_number(value, name)
if ~isscalar(value) || ~isfinite(value) || value < 0
    error('%s must be a finite nonnegative number.', name);
end
end
