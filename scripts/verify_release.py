import json
import re
from pathlib import Path

ROOT_FILES = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "CITATION.cff",
              "pyproject.toml", "requirements.txt", ".gitignore", "RELEASE_MANIFEST.json"}
ROOT_DIRS = {"src", "tests", "configs", "scripts", ".github", "licenses"}
IGNORED = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "build", "dist"}
EXTENSIONS = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".jsonl", ".toml", ".cff", ".sh"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "provider_token": re.compile(r"\b(?:sk-|ghp_|github_pat_|hf_)[A-Za-z0-9_-]{24,}"),
    "home_directory": re.compile(r"/(?:home\d*|Users)/[A-Za-z0-9_.-]+/"),
    "credential_assignment": re.compile(r'''(?i)(?:api_key|api_secret|access_token|password)\s*[:=]\s*["'][^"'\n]{16,}["']'''),
}


def release_files(root):
    files, issues = [], []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in IGNORED or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append({"path": str(relative), "reason": "symlink"})
            continue
        if not path.is_file():
            continue
        allowed = (len(relative.parts) == 1 and relative.name in ROOT_FILES) or (
            len(relative.parts) > 1 and relative.parts[0] in ROOT_DIRS and path.suffix in EXTENSIONS)
        if not allowed:
            issues.append({"path": str(relative), "reason": "outside source allowlist"})
            continue
        if path.suffix == ".jsonl":
            issues.append({"path": str(relative), "reason": "unexpected data records"})
        if path.stat().st_size > 1024 * 1024:
            issues.append({"path": str(relative), "reason": "unexpected large file"})
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            issues.append({"path": str(relative), "reason": "non-text file"})
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(content):
                issues.append({"path": str(relative), "line": content.count("\n", 0, match.start()) + 1,
                               "reason": name})
        files.append(path)
    return files, issues


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    files, issues = release_files(root)
    print(json.dumps({"status": "failed" if issues else "passed", "files": len(files),
                      "bytes": sum(path.stat().st_size for path in files), "issues": issues}, indent=2))
    raise SystemExit(1 if issues else 0)
