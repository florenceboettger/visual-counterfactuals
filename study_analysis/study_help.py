from study import Study

from pathlib import Path

def load_studies(path: Path) -> list[Study]:
    study_dirs = [str(path / f) for f in path.iterdir() if (path / f).is_dir()]

    studies = []
    for dir in study_dirs:
        studies.append(Study.from_directory(dir))
        
    return studies