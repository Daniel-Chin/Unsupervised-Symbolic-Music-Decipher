import json
import random
import argparse
from io import BytesIO
from enum import Enum

import torch
from torch import Tensor
import pretty_midi
import audioread
from audioread.rawread import RawAudioFile
from tqdm import tqdm
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt

from shared import *
from music import PIANO_RANGE
from midi_synth_wav import midiSynthWav
from my_encodec import getEncodec, HFEncodecCompressionModel

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
    for srcNote in srcPiano.notes:
        srcNote: pretty_midi.Note
        if srcNote.start >= SONG_LEN:
            break   # because we know the notes are sorted by start time
        piano.notes.append(pretty_midi.Note(
            velocity=srcNote.velocity, pitch=srcNote.pitch, 
            start=srcNote.start, end=min(srcNote.end, SONG_LEN),
        ))
    return midi

def prepareOneDatapoint(
    stage: Stage, encodec: Optional[HFEncodecCompressionModel], 
    idx: int, dest_dir: str, midi_source: Optional[str], 
    verbose: bool, do_fluidsynth_write_pcm: bool, 
):
    def printProfiling(*a, **kw):
        if verbose:
            print(*a, **kw, flush=True)
    
    wav_path = path.join(dest_dir, f'{idx}_synthed.wav')
    midi_path = path.join(dest_dir, f'{idx}.mid')

    if stage == Stage.CPU:
        if midi_source is None:
            printProfiling('Generating MIDI')
            midi = generateMidi()
        else:
            printProfiling('Legalizing MIDI')
            midi = legalizeMidi(midi_source)
        piano, = midi.instruments
        piano: pretty_midi.Instrument
        n_notes = len(piano.notes)

        printProfiling('Writing MIDI')
        midi.write(midi_path)

        printProfiling('Synthesizing audio')
        midiSynthWav(midi_path, wav_path, verbose, do_fluidsynth_write_pcm)
    
    elif stage == Stage.GPU:
        assert encodec is not None

        printProfiling('Loading midi')
        midi = pretty_midi.PrettyMIDI(midi_path)
        piano, = midi.instruments
        piano: pretty_midi.Instrument
        n_notes = len(piano.notes)
        
        printProfiling('Loading audio')
        buf = BytesIO()
        with audioread.audio_open(wav_path) as f:
            f: RawAudioFile
            assert f.samplerate == ENCODEC_SR
            n_channels = f.channels
            for chunk in f.read_data():
                buf.write(chunk)
        buf.seek(0)
        dtype = np.dtype(np.int16).newbyteorder('<')
        wave_int = torch.tensor(np.frombuffer(buf.read(), dtype=dtype)).to(DEVICE)
        format_factor: int = 2 ** (dtype.itemsize * 8 - 1)  # needs type hint because type checker doesn't know dtype.itemsize > 0
        wave_float = wave_int.float() / format_factor
        wave_mono = wave_float.view((-1, n_channels)).mean(dim=1)
        n_samples = int(np.ceil(SONG_LEN * ENCODEC_SR))
        wave_trunc = wave_mono[:n_samples]
        wave_pad = torch.nn.functional.pad(
            wave_trunc, (0, n_samples - len(wave_trunc)),
        )
        wave = wave_pad

        with torch.no_grad():
            printProfiling('Encodec.encode')
            codes, _ = encodec.encode(wave.unsqueeze(0).unsqueeze(0))
            printProfiling('Encodec.decode')
            recon: Tensor = encodec.decode(codes)[0, 0, :]   # type: ignore
        
        printProfiling('Writing recon')
        recon_path = path.join(dest_dir, f'{idx}_encodec_recon.wav')
        wavfile.write(recon_path, ENCODEC_SR, recon.cpu().numpy())
        
        printProfiling('Formatting datapoint')
        x = (-torch.randn((
            2, 
            PIANO_RANGE[1] - PIANO_RANGE[0],
            SEC_PER_DATAPOINT * ENCODEC_FPS, 
        )).square()).exp()
        x[0, :, :] = 0.0
        for note in piano.notes:
            note: pretty_midi.Note
            duration = note.end - note.start
            t_slice = slice(
                round(note.start * ENCODEC_FPS), 
                round(note.end   * ENCODEC_FPS), 
            )
            pitch_index = note.pitch - PIANO_RANGE[0]
            x[0, pitch_index, t_slice] = note.velocity / 127.0
            x[1, pitch_index, t_slice] = torch.linspace(
                0, -1.0 * duration, t_slice.stop - t_slice.start,
            ).exp()
        y = codes[0, :, :].to(torch.int16).cpu()
        assert y.shape == (ENCODEC_N_BOOKS, N_TOKENS_PER_DATAPOINT)

        printProfiling('Writing datapoint')
        torch.save(x, path.join(
            dest_dir, f'{idx}_x.pt',
        ))
        torch.save(y, path.join(
            dest_dir, f'{idx}_y.pt',
        ))

        return x, y

def prepareOneSet(
    which_set: WhichSet,
    stage: Stage,
    select_dir: str, 
    n_datapoints: int, 
    verbose: bool, 
    do_fluidsynth_write_pcm: bool, 
    plot_x: bool = False,
):
    if stage == Stage.GPU:
        encodec = getEncodec().to(DEVICE)
    else:
        encodec = None

    if which_set == WhichSet.MONKEY:
        dest_set_dir = TRANSFORMER_PIANO_MONKEY_DATASET_DIR
        midi_basenames = [None] * n_datapoints
    elif which_set == WhichSet.ORACLE:
        dest_set_dir = TRANSFORMER_PIANO_ORACLE_DATASET_DIR
        with open(path.join(
            PIANO_LA_DATASET_DIR, select_dir, 'index.json', 
        ), 'r', encoding='utf-8') as f:
            filenames = json.load(f)
        midi_basenames = random.choices(filenames, k=n_datapoints)
    dest_dir = path.join(dest_set_dir, select_dir)
    os.makedirs(dest_dir, exist_ok=True)

    data_ids = []
    try:
        for datapoint_i, midi_basename in enumerate(tqdm(midi_basenames)):
            if verbose:
                print()
            out = prepareOneDatapoint(
                stage, encodec, datapoint_i, dest_dir, 
                midi_basename and path.join(PIANO_LA_DATASET_DIR, select_dir, midi_basename), 
                verbose, do_fluidsynth_write_pcm, 
            )
            data_ids.append(str(datapoint_i))
            if plot_x and out is not None:
                x, _ = out
                plt.imshow(x[0, :, :].T, aspect='auto')
                plt.colorbar()
                plt.show()
                plt.imshow(x[1, :, :].T, aspect='auto')
                plt.colorbar()
                plt.show()
    finally:
        if stage == Stage.GPU:
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
        for select_dir in tqdm(LA_DATASET_DIRS, desc=which_set.value):
            for stage in (Stage.CPU, Stage.GPU):
                prepareOneSet(
                    which_set, 
                    stage, 
                    select_dir, 
                    n_datapoints, 
                    verbose=False, 
                    do_fluidsynth_write_pcm=False, 
                    # plot_x=True,
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
        '--stage', type=Stage, required=True, choices=[*Stage],
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
        '--do_fluidsynth_write_pcm', action='store_true',
    )
    args = parser.parse_args()
    prepareOneSet(
        args.which_set, 
        args.stage, 
        args.select_dir, 
        args.n_datapoints, 
        args.verbose, 
        args.do_fluidsynth_write_pcm, 
    )
