from dataclasses import dataclass
from pathlib import Path
import subprocess


@dataclass(frozen=True, slots=True)
class RepositoryState:
    commit: str
    tags: tuple[str, ...]

    @property
    def tag(self) -> str | None:
        return self.tags[0] if self.tags else None


def get_git_revision(repo_path: str | Path = ".") -> RepositoryState:
    repo = str(repo_path)

    commit = subprocess.check_output(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        text=True,
    ).strip()

    tags = subprocess.check_output(
        ["git", "-C", repo, "tag", "--points-at", "HEAD"],
        text=True,
    ).splitlines()

    return RepositoryState(commit=commit, tags=tuple(tags))
