import json
from subprocess import Popen, DEVNULL

import librosa
import torch
import pretty_midi
import audiocraft
from audiocraft.models.encodec import HFEncodecCompressionModel
import tqdm

from shared import *
from music import PIANO_RANGE

(DENSITY_MU, DENSITY_SIGMA) = (2.520, 0.672)
(DURATION_MU, DURATION_SIGMA) = (-1.754, 1.077)
(VELOCITY_MU, VELOCITY_SIGMA) = (84.174, 25.561)
(PITCH_MU, PITCH_SIGMA) = (60.229, 13.938)

def main(dataset_size: int):
    encodec = audiocraft.models.encodec.CompressionModel.get_pretrained(
        'facebook/encodec_32khz', DEVICE, 
    )
    assert isinstance(encodec, HFEncodecCompressionModel)
    encodec.eval()
    stems = []
    try:
        for i in tqdm.trange(dataset_size):
            print()
            print('Generating MIDI')
            midi = pretty_midi.PrettyMIDI()
            piano = pretty_midi.Instrument(0, is_drum=False, name='Piano')
            midi.instruments.append(piano)
            song_len = float(SEC_PER_DATAPOINT)
            density = torch.randn((1, )).mul(DENSITY_SIGMA).add(DENSITY_MU).exp().item()
            n_notes = round(song_len * density)
            onsets = torch.rand((n_notes, )).mul(song_len).numpy()
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
                    start=onset, end=min(onset + duration, song_len),
                ))
            
            print('Writing MIDI file')
            midi_path = path.join(
                TRANSFORMER_PIANO_DATASET_DIR, f'{i}.mid',
            )
            midi.write(midi_path)

            print('Synthesizing audio')
            wav_path = path.join(
                TRANSFORMER_PIANO_DATASET_DIR, 'temp.wav',
            )
            with Popen([
                'fluidsynth', '-ni', SOUNDFONT_FILE, midi_path,
                '-F', wav_path, '-r', str(ENCODEC_SR), 
            ], stdout=DEVNULL) as p:
                p.wait()
            
            print('Loading audio')
            wave, _ = librosa.load(wav_path, sr=ENCODEC_SR, mono=True)
            wave_trunc = wave[:int(np.ceil(song_len * ENCODEC_SR))]
            wave_gpu = torch.Tensor(wave_trunc).to(DEVICE)

            print('Encodec')
            with torch.no_grad():
                codes, _ = encodec.encode(wave_gpu.unsqueeze(0).unsqueeze(0))
            
            print('Formatting datapoint')
            x = torch.zeros((
                n_notes, 
                # 1 + 1 + 88, 
                1 + 1 + 1, 
            ))
            for note_i, note in enumerate(piano.notes):
                note: pretty_midi.Note
                x[note_i, 0] = note.start / float(SEC_PER_DATAPOINT)
                x[note_i, 1] = note.velocity / 127.0
                # x[note_i, 2 + note.pitch - PIANO_RANGE[0]] = 1.0
                x[note_i, 2] = float(note.pitch)
            y = codes[0, :, :].to(torch.int16).cpu()
            assert y.shape == (4, N_TOKENS_PER_DATAPOINT)

            print('Writing datapoint')
            stem = str(i)
            torch.save(x, path.join(
                TRANSFORMER_PIANO_DATASET_DIR, f'{stem}_x.pt',
            ))
            torch.save(y, path.join(
                TRANSFORMER_PIANO_DATASET_DIR, f'{stem}_y.pt',
            ))

            stems.append(stem)
    finally:
        with open(path.join(
            TRANSFORMER_PIANO_DATASET_DIR, 'index.json',
        ), 'w', encoding='utf-8') as f:
            json.dump(stems, f)

if __name__ == '__main__':
    main(64)
