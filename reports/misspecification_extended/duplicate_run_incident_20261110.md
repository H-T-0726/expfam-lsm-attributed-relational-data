# MX2 duplicate-run incident (2026-11-10)

During the first sequential full-run attempt, Ctrl+C ended the shell session but left the Python child process alive. A second resume process then wrote ten duplicate fit keys to the same CSV. This was a process-control error, not a scientific or numerical failure.

The remaining Python process (PID 18324, start 03:09 local time) was stopped before any further fit execution. The exact 212-row file was preserved as `fit_results_duplicate_incident_raw.csv`. The scripted `--repair` step kept the first recorded row for each key, leaving 202 unique fit rows, and wrote `duplicate_incident.csv` with all excluded keys and changed fields. For every duplicate, the recorded scientific metrics and diagnostic fields were identical; only wall runtime differed. No failed fit was replaced, no seed was changed, and no output row was invented. The repair script refuses to run when no duplicates exist or when the raw archive already exists.

This means there were ten extra recorded MCEM executions beyond the 1,120 unique new fit cells in the design. An in-flight execution may also have been interrupted when the process was stopped; it has no completed result row and cannot be counted precisely. The final report therefore distinguishes unique analyzed cells from physical execution attempts and does not claim that every attempted process is represented by a unique result.

The runner now uses a single parent process to append completed results when running multiple workers. The manifest is derived from the unique result keys; resuming checks these keys before submitting new work. The raw incident CSV and duplicate table remain part of the provenance.
