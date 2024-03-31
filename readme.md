## Prepare Repo
- Use conda to install "./environment.yaml"  
- Download the [LA Midi Dataset](https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset) and set `LA_MIDI_PATH` in "./active.env" to the midi dir (ending with `.../Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs`)  
- Install `fluidsynth`.  
- Obtain "MuseScore_Basic.sf2" and put it in "./assets".  
  - Instructions are in "./assets/acknowledge.md".  
- Run `conda activate env_neural_avh`
- Run `main()` in "./midi_process.py"  
- Run `python ./tf_piano_dataset_prepare.py --monkey_dataset_size 1024 --oracle_dataset_size 64`
