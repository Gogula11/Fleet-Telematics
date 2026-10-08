import os, runpy


def run():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for s in ["ingest", "cleaning", "features", "train", "explain", "visualise"]:
        runpy.run_path(f"src/{s}.py", run_name="__main__")
