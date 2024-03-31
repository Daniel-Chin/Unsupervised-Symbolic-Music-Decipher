import json
from subprocess import Popen
import random
import argparse
from io import BytesIO

import torch
from torch import Tensor
import pretty_midi
import audiocraft
from audiocraft.models.encodec import HFEncodecCompressionModel
import audioread
from audioread.rawread import RawAudioFile
import tqdm
import scipy.io.wavfile as wavfile

from shared import *
from music import PIANO_RANGE

(DENSITY_MU, DENSITY_SIGMA) = (2.520, 0.672)
(DURATION_MU, DURATION_SIGMA) = (-1.754, 1.077)
(VELOCITY_MU, VELOCITY_SIGMA) = (84.174, 25.561)
(PITCH_MU, PITCH_SIGMA) = (60.229, 13.938)

SONG_LEN = float(SEC_PER_DATAPOINT)

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
        pitches > PIANO_RANGE[1], 
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
    encodec: audiocraft.models.encodec.CompressionModel, 
    idx: int, dest_dir: str, midi_source: Optional[str], 
    fluid_synth_out: str,
):
    if midi_source is None:
        print('Generating MIDI')
        midi = generateMidi()
    else:
        print('Legalizing MIDI')
        midi = legalizeMidi(midi_source)
    piano, = midi.instruments
    piano: pretty_midi.Instrument
    n_notes = len(piano.notes)

    print('Writing MIDI')
    midi_path = path.join(dest_dir, f'{idx}.mid')
    midi.write(midi_path)

    print('Synthesizing audio')
    wav_path = path.join(dest_dir, 'temp.wav')
    with open(fluid_synth_out, 'w') as pOut:
        with Popen([
            'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
            '-F', wav_path, '-r', str(ENCODEC_SR), 
        ], stdout=pOut) as p:
            p.wait()
    
    print('Loading audio')
    buf = BytesIO()
    with audioread.audio_open(wav_path) as f:
        f: RawAudioFile
        assert f.samplerate == ENCODEC_SR
        assert f.channels == 1
        for chunk in f.read_data():
            buf.write(chunk)
    buf.seek(0)
    dtype = np.dtype(np.int16).newbyteorder('<')
    wave_int = np.frombuffer(buf.read(), dtype=dtype)
    wave = wave_int.astype(np.float32, copy=False) / 2 ** (dtype.itemsize * 8 - 1)
    n_samples = int(np.ceil(SONG_LEN * ENCODEC_SR))
    wave_trunc = torch.Tensor(wave)[:n_samples]
    wave_pad = torch.nn.functional.pad(
        wave_trunc, (0, n_samples - len(wave_trunc)),
    )
    wave_gpu = wave_pad.to(DEVICE)

    with torch.no_grad():
        print('Encodec.encode')
        codes, _ = encodec.encode(wave_gpu.unsqueeze(0).unsqueeze(0))
        print('Encodec.decode')
        recon: Tensor = encodec.decode(codes)[0, 0, :]   # type: ignore
    
    print('Writing recon')
    # write `recon` wave to file
    recon_path = path.join(dest_dir, f'{idx}_encodec_recon.wav')
    wavfile.write(recon_path, ENCODEC_SR, recon.cpu().numpy())
    
    print('Formatting datapoint')
    x = torch.zeros((
        n_notes, 
        # 1 + 1 + 88, 
        1 + 1 + 1, 
    ))
    for i, note in enumerate(piano.notes):
        note: pretty_midi.Note
        x[i, 0] = note.start / float(SEC_PER_DATAPOINT)
        x[i, 1] = note.velocity / 127.0
        # x[note_i, 2 + note.pitch - PIANO_RANGE[0]] = 1.0
        x[i, 2] = float(note.pitch)
    y = codes[0, :, :].to(torch.int16).cpu()
    assert y.shape == (4, N_TOKENS_PER_DATAPOINT)

    print('Writing datapoint')
    torch.save(x, path.join(
        dest_dir, f'{idx}_x.pt',
    ))
    torch.save(y, path.join(
        dest_dir, f'{idx}_y.pt',
    ))

def main(
    monkey_dataset_size: int, 
    oracle_dataset_size: int,
    fluid_synth_out: str,
):
    encodec = audiocraft.models.encodec.CompressionModel.get_pretrained(
        'facebook/encodec_32khz', DEVICE, 
    )
    assert isinstance(encodec, HFEncodecCompressionModel)
    encodec.eval()

    def oneSet(dest_dir: str, midi_sources: List, desc: str):
        data_ids = []
        try:
            for datapoint_i, midi_source in enumerate(tqdm.tqdm(midi_sources, desc)):
                print()
                prepareOneDatapoint(
                    encodec, datapoint_i, dest_dir, midi_source, fluid_synth_out, 
                )
                data_ids.append(str(datapoint_i))
        finally:
            with open(path.join(
                dest_dir, 'index.json',
            ), 'w', encoding='utf-8') as f:
                json.dump(data_ids, f)
    
    oneSet(TRANSFORMER_PIANO_MONKEY_DATASET_DIR, [None] * monkey_dataset_size, 'monkey')

    midi_sources = []
    for i, dir_ in enumerate(LA_DATASET_DIRS):
        with open(path.join(
            PIANO_LA_DATASET_DIR, dir_, 'index.json', 
        ), 'r', encoding='utf-8') as f:
            filenames = json.load(f)
        still_need = oracle_dataset_size - len(midi_sources)
        lets_take = still_need // (len(LA_DATASET_DIRS) - i)
        midi_sources.extend([
            path.join(PIANO_LA_DATASET_DIR, dir_, x) 
            for x in random.choices(filenames, k=lets_take)
        ])
    oneSet(TRANSFORMER_PIANO_ORACLE_DATASET_DIR, midi_sources, 'oracle')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--monkey_dataset_size', type=int, required=True, 
    )
    parser.add_argument(
        '--oracle_dataset_size', type=int, required=True, 
    )
    parser.add_argument(
        '--fluidsynth_out', type=str, required=True,
    )
    args = parser.parse_args()
    main(args.monkey_dataset_size, args.oracle_dataset_size, args.fluidsynth_out)
