import json
import random
import argparse
from enum import Enum

import torch
from torch import Tensor
import pretty_midi
from tqdm import tqdm
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import librosa

from shared import *
from music import PIANO_RANGE
from midi_synth_wav import midiSynthWave, SynthAnomalyChecker
from my_musicgen import myMusicGen, EncodecModel

(DENSITY_MU, DENSITY_SIGMA) = (2.520, 0.672)
(DURATION_MU, DURATION_SIGMA) = (-1.754, 1.077)
(VELOCITY_MU, VELOCITY_SIGMA) = (84.174, 25.561)
(PITCH_MU, PITCH_SIGMA) = (60.229, 13.938)

SONG_LEN = float(SEC_PER_DATAPOINT)

class WhichSet(Enum):
    MONKEY = 'monkey'
    ORACLE = 'oracle'

class Stage(Enum):
    CPU = 'cpu'
    GPU = 'gpu'

class BadMidi(Exception): pass
class MidiTooShort(BadMidi): pass
class NoNotesInMidi(BadMidi): pass

def generateMidi():
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0, is_drum=False, name='Piano')
    midi.instruments.append(piano)
    density = torch.randn((1, )).mul(DENSITY_SIGMA).add(DENSITY_MU).exp().item()
    n_notes = round(SONG_LEN * density)
    onsets = torch.rand((n_notes, )).mul(SONG_LEN).numpy()
    durations = torch.randn((n_notes, )).mul(DURATION_SIGMA).add(DURATION_MU).exp().numpy()
    velocities = torch.randn((n_notes, )).mul(VELOCITY_SIGMA).add(VELOCITY_MU).clamp(1, 127).round().numpy()
    pitches = torch.randn((n_notes, )).mul(PITCH_SIGMA).add(PITCH_MU).round().numpy()
    pitches[np.logical_or(
        pitches < PIANO_RANGE[0], 
        pitches >= PIANO_RANGE[1], 
    )] = 60
    for onset, duration, velocity, pitch in zip(
        onsets, durations, velocities, pitches, 
    ):
        piano.notes.append(pretty_midi.Note(
            velocity=int(velocity), pitch=int(pitch), 
            start=onset, end=min(onset + duration, SONG_LEN),
        ))
    return midi

def legalizeMidi(src_path: str):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0, is_drum=False, name='Piano')
    midi.instruments.append(piano)

    srcMidi = pretty_midi.PrettyMIDI(src_path)
    srcPiano, = srcMidi.instruments
    srcPiano: pretty_midi.Instrument
    sortByNoteOn(srcPiano)
    total_len = srcMidi.get_end_time()
    if total_len <= SONG_LEN:
        raise MidiTooShort(src_path)
    leeway = total_len - SONG_LEN
    offset = random.uniform(0, leeway)
    for srcNote in srcPiano.notes:
        srcNote: pretty_midi.Note
        start = srcNote.start - offset
        end   = srcNote.end   - offset
        if start < 0.0:
            continue
        if start >= SONG_LEN:
            break   # because we know the notes are sorted by start time
        piano.notes.append(pretty_midi.Note(
            velocity=srcNote.velocity, pitch=srcNote.pitch, 
            start=start, end=min(end, SONG_LEN),
        ))
    if not piano.notes:
        raise NoNotesInMidi(src_path)
    return midi

def prepareOneDatapoint(
    encodec: EncodecModel, 
    idx: int, dest_dir: str, midi_source: Optional[str], 
    checker: SynthAnomalyChecker,
    verbose: bool, is_fluidsynth_nyush: bool, 
):
    def printProfiling(*a, **kw):
        if verbose:
            print(*a, **kw, flush=True)
    
    synth_temp_path = path.join(dest_dir, f'temp')
    midi_path = path.join(dest_dir, f'{idx}.mid')

    if midi_source is None:
        printProfiling('Generating MIDI')
        midi = generateMidi()
    else:
        printProfiling('Legalizing MIDI')
        midi = legalizeMidi(midi_source)
    piano, = midi.instruments
    piano: pretty_midi.Instrument

    printProfiling('Writing MIDI')
    midi.write(midi_path)
    
    printProfiling('Synthesizing audio')
    wave_np = midiSynthWave(midi_path, synth_temp_path, checker, verbose, is_fluidsynth_nyush)
    wave = torch.tensor(wave_np, device=DEVICE)

    if idx < 8:
        printProfiling('Writing wav')
        wavfile.write(path.join(dest_dir, f'{idx}.wav'), ENCODEC_SR, wave_np)
    
    # printProfiling('Loading midi')
    # midi = pretty_midi.PrettyMIDI(midi_path)
    # piano, = midi.instruments
    # piano: pretty_midi.Instrument
    # n_notes = len(piano.notes)

    with torch.no_grad():
        printProfiling('Encodec.encode')
        codes, _ = encodec.encode(wave.unsqueeze(0).unsqueeze(0))
        # printProfiling('Encodec.decode')
        # recon: Tensor = encodec.decode(codes)[0, 0, :]   # type: ignore
    
    # printProfiling('Writing recon')
    # recon_path = path.join(dest_dir, f'{idx}_encodec_recon.wav')
    # wavfile.write(recon_path, ENCODEC_SR, recon.cpu().numpy())

    # printProfiling('STFT')
    # stft, griffinLim, n_bins = fftTools()
    # spectrogram: Tensor = stft(wave)
    # freq, t = spectrogram.shape
    # assert freq == n_bins
    # if t == N_FRAMES_PER_DATAPOINT:
    #     pass
    # elif t == N_FRAMES_PER_DATAPOINT + 1:
    #     spectrogram = spectrogram[:, :-1]
    # elif t == N_FRAMES_PER_DATAPOINT - 1:
    #     spectrogram = torch.nn.functional.pad(
    #         spectrogram, (0, 1), value=1e-5, 
    #     )
    #     raise ValueError(t)
    # else:
    #     raise ValueError(t)
    # log_spectrogram = spectrogram.log()

    # printProfiling('Writing Griffin-Lim')
    # griffin_lim_path = path.join(dest_dir, f'{idx}_griffin_lim.wav')
    # griffin_lim: Tensor = griffinLim(log_spectrogram.exp())
    # wavfile.write(griffin_lim_path, ENCODEC_SR, griffin_lim.cpu().numpy())
    
    printProfiling('Formatting datapoint')
    score = torch.zeros((
        2, 
        PIANO_RANGE[1] - PIANO_RANGE[0],
        N_FRAMES_PER_DATAPOINT, 
    ))
    for note in piano.notes:
        note: pretty_midi.Note
        duration = note.end - note.start
        t_slice = slice(
            round(note.start * ENCODEC_FPS), 
            round(note.end   * ENCODEC_FPS), 
        )
        pitch_index = note.pitch - PIANO_RANGE[0]
        score[0, pitch_index, t_slice] = note.velocity / 127.0
        score[1, pitch_index, t_slice] = torch.linspace(
            0, -1.0 * duration, t_slice.stop - t_slice.start,
        ).exp()
    encodec_tokens = codes[0, :, :].to(torch.int16).cpu()
    assert encodec_tokens.shape == (ENCODEC_N_BOOKS, N_FRAMES_PER_DATAPOINT), encodec_tokens.shape

    printProfiling('Writing datapoint')
    torch.save(score, path.join(
        dest_dir, f'{idx}_score.pt',
    ))
    torch.save(encodec_tokens, path.join(
        dest_dir, f'{idx}_encodec_tokens.pt',
    ))
    # log_spectrogram_ = log_spectrogram.cpu().to(torch.float16)
    # torch.save(log_spectrogram_, path.join(
    #     dest_dir, f'{idx}_log_spectrogram.pt',
    # ))

    return (
        score, encodec_tokens, 
        # log_spectrogram_, 
        None, 
    )

def prepareOneSet(
    which_set: WhichSet,
    select_dir: str, 
    n_datapoints: int, 
    synthAnomalyChecker: SynthAnomalyChecker,
    verbose: bool, 
    is_fluidsynth_nyush: bool, 
    only_plot_no_write_disk: bool = False,
):
    encodec = myMusicGen.encodec.to(DEVICE)
    encodec.eval()

    if which_set == WhichSet.MONKEY:
        dest_set_dir = PIANO_MONKEY_DATASET_DIR
    elif which_set == WhichSet.ORACLE:
        dest_set_dir = PIANO_ORACLE_DATASET_DIR
        with open(path.join(
            PIANO_LA_DATASET_DIR, select_dir, 'index.json', 
        ), 'r', encoding='utf-8') as f:
            midi_filenames: List = json.load(f)
        random.shuffle(midi_filenames)
    dest_dir = path.join(dest_set_dir, select_dir)
    os.makedirs(dest_dir, exist_ok=True)

    data_ids = []
    try:
        for datapoint_i in tqdm(range(n_datapoints)):
            if verbose:
                print()
            while True:
                if which_set == WhichSet.ORACLE:
                    midi_src = path.join(
                        PIANO_LA_DATASET_DIR, 
                        select_dir, 
                        midi_filenames.pop(0), 
                    )
                elif which_set == WhichSet.MONKEY:
                    midi_src = None
                try:
                    score, encodec_tokens, log_spectrogram = prepareOneDatapoint(
                        encodec, datapoint_i, dest_dir, 
                        midi_src, synthAnomalyChecker, 
                        verbose, is_fluidsynth_nyush, 
                    )
                except BadMidi:
                    if verbose:
                        print('BadMidi')
                else:
                    break
            data_ids.append(str(datapoint_i))
            if only_plot_no_write_disk:
                fig, axes = plt.subplots(2, 1, sharex=True)
                axes: List[Axes]
                im0 = plotScore(score, axes[0])
                colorBar(fig, axes[0], im0)

                assert isinstance(log_spectrogram, Tensor)
                spectrogram: np.ndarray = log_spectrogram.exp().clamp(1e-6, 100.0).numpy()
                # D = librosa.amplitude_to_db(spectrogram)
                # im1 = axes[1].imshow(
                #     D, aspect='auto', interpolation='nearest', 
                #     origin='lower', 
                # )
                # colorBar(fig, axes[1], im1)
                _, _, n_bins = fftTools()
                axes[1].pcolormesh(
                    # np.linspace(0, SONG_LEN, N_FRAMES_PER_DATAPOINT),
                    np.arange(N_FRAMES_PER_DATAPOINT),
                    np.linspace(0, ENCODEC_SR / 2, n_bins), 
                    spectrogram ** 0.5, 
                )
                plt.show()
    finally:
        if not only_plot_no_write_disk:
            with open(path.join(
                dest_dir, 'index.json',
            ), 'w', encoding='utf-8') as f:
                json.dump(data_ids, f)

def laptop():
    TO_PREPARE: List[Tuple[WhichSet, int]] = [
        (WhichSet.ORACLE, 32 // 16),
        (WhichSet.MONKEY, 32 // 16), 
    ]

    for which_set, n_datapoints in TO_PREPARE:
        with SynthAnomalyChecker().context() as checker:
            for select_dir in tqdm(LA_DATASET_DIRS, desc=which_set.value):
                prepareOneSet(
                    which_set, 
                    select_dir, 
                    n_datapoints, 
                    checker, 
                    verbose=False, 
                    is_fluidsynth_nyush=False, 
                    # only_plot_no_write_disk=True,
                )

if __name__ == '__main__':
    initMainProcess()

    # laptop()
    # import sys; sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--which_set', type=WhichSet, required=True, choices=[*WhichSet],
    )
    parser.add_argument(
        '--select_dir', type=str, required=True, 
    )
    parser.add_argument(
        '--n_datapoints', type=int, required=True, 
    )
    parser.add_argument(
        '--verbose', action='store_true',
    )
    parser.add_argument(
        '--is_fluidsynth_nyush', action='store_true',
    )
    args = parser.parse_args()
    with SynthAnomalyChecker().context() as checker:
        prepareOneSet(
            args.which_set, 
            args.select_dir, 
            args.n_datapoints, 
            checker, 
            args.verbose, 
            args.is_fluidsynth_nyush, 
        )
    print('OK')
