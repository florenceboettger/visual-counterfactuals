from study import Study

def load_studies(path):
    study_dirs = [str(path / f) for f in path.iterdir() if (path / f).is_dir()]

    studies = []
    for dir in study_dirs:
        studies.append(Study.from_directory(dir))
        
    return studies