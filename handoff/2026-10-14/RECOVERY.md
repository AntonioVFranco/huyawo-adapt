# Huyawo End-of-Day Project Backup

Backup date: 2026-10-14. Source repository: AntonioVFranco/huyawo-adapt.

Each ordered part is verified by SHA-256 in manifest.json. Concatenate parts in numeric order into project-state.tar.gz and verify its SHA-256 against archive_sha256. Extract ONLY inside an isolated /workspace/huyawo-agentic-adaptation restoration directory after checking archive member paths and symlinks. The archive already contains all nonsecret persistent project files, including prior experiment evidence and the pinned six tokenizer files. The exact installed TRL template bytes are under frozen-installed-trl/. Recreate .venv independently using the tracked dependency definitions. Canonical RFC-HAA-0037 ACCEPTED and the project CONTEXT.md remain available separately in the Huyawo Google Drive.
