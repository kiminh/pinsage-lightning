from pathlib import Path


def get_project_dir():
    """Get the path to the project home."""
    path = Path(__file__).parent.parent
    project_dir = path.parent
    return project_dir
