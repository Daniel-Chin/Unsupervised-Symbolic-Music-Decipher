@lru_cache(maxsize=1)
def soundfontFilePath():
    out = StringIO()
    with Popen(['fluidsynth', '--version'], stdout=out) as p:
        p.wait()
    out.seek(0)
    text = out.read()
    version = text.split('FluidSynth runtime version ', 1)[1].split('\n', 1)[0].strip()
