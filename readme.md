## Prepare Repo
- Use conda to install "./environment.yaml"  
  - Fix CUDA version to your native.
- Copy "./example.env" to "active.env".  
- Download the [LA Midi Dataset](https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset) and set `LA_MIDI_PATH` in "./active.env" to the midi dir (ending with `.../Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs`)  
- Install `fluidsynth`.  
- Obtain "MuseScore_Basic.sf2" and put it in "./assets".  
  - Instructions are in "./assets/acknowledge.md".  
- Run `cd ./hpc`
- Run `python3 ./midi_process_parallel.py` and wait for completion. 
- Run `python3 ./prepare_datasets_parallel.py --stage cpu` and wait for completion. 
- Run `python3 ./prepare_datasets_parallel.py --stage gpu` and wait for completion. 

## Train
From the outermost dir,  
- Run `cd ./hpc`
- Run `python3 ./sched_piano.py` to submit a slurm job. 
  - It uses `train_piano_template.sbatch` which has some NYUSH HPC-specific setups. You may want to review it and make edits. 
  - Under the hood it runs `../main_train_piano.py`. Feel free to run it directly. 
