# Codebase Report: .claude/ Directory Analysis
Generated: 2026-01-16 09:53

## Summary

The `.claude/` directory in `/home/danny/Documents/personal/extension/training` appears **healthy and minimal** with no obvious corruption or bloat. The hang during `claude resume` is likely NOT caused by issues in this local directory, but rather by:

1. **Active Claude sessions** - 5 claude processes are running simultaneously
2. **Large global cache** - `~/.claude/` is 1.3GB with 187 session environments
3. **Session state tracking** - A SQLite database tracks instance sessions

## Project .claude/ Structure

```
.claude/                                    (44K total)
├── cache/                                  (44K)
│   ├── agents/
│   │   └── oracle/
│   │       └── latest-output.md           (13K - RF-DETR research)
│   └── artifact-index/
│       └── context.db                     (12K - SQLite database)
└── settings.local.json                    (3.9K - permissions config)
```

## Key Findings

### ✓ VERIFIED: No Session Bloat
- **Local .claude/ size:** 44K (very small)
- **Global ~/.claude/ size:** 1.3GB (contains 187 session environments)
- Only 7 files total in local .claude/ directory
- No large files, no session locks, no corrupted state

### ✓ VERIFIED: Minimal Cache Contents

**Agent Cache:**
- Only `oracle` agent has cached output (13K RF-DETR research report from Jan 15)
- No scout, kraken, architect, or other agent caches present

**Artifact Index Database:**
- SQLite database: `context.db` (12K)
- Schema: Single table `instance_sessions` 
- Contents: 1 entry tracking terminal PID 2322993 for "validation-output" session
- Last updated: 2026-01-16 14:38:03

### ✓ VERIFIED: Settings Configuration

**Permissions in settings.local.json:**
- 101 pre-approved bash commands and tool invocations
- MCP plugin permissions for Serena (code navigation tool)
- WebSearch and WebFetch allowed for multiple domains (docs sites, arxiv, github, etc.)
- Python, pip, uv, git, and training scripts authorized
- No unusual or malformed permissions

### ? INFERRED: Potential Hang Causes

**Active Claude Processes:**
```
PID     CPU%  MEM%   STATUS
65419   150%  0.5%   Rl+ (running)
69817   43.8% 0.1%   Sl+ (sleeping)
96010   151%  0.3%   Rl+ (running)
118552  16.6% 0.3%   Rl+ (running - THIS SESSION)
176335  11.5% 0.1%   Sl+ (sleeping)
```

5 simultaneous claude sessions are running, with 2 consuming 150% CPU each. This suggests:
- Heavy concurrent workload
- Possible resource contention
- Multiple terminals trying to access shared state

**Hook Processes:**
- 2 Node.js hook processes running:
  - `edit-context-inject.mjs`
  - `signature-helper.mjs`

### ✓ VERIFIED: No File Locks
- `lsof` check found no file locks in `.claude/` directory
- SQLite database not locked
- No stale lock files present

## Comparison to Normal .claude/ Directory

| Component | This Directory | Expected | Status |
|-----------|----------------|----------|--------|
| Size | 44K | 10K-100K | ✓ Normal |
| Agent caches | 1 (oracle) | 0-5 | ✓ Normal |
| Session state | 1 entry | 0-3 | ✓ Normal |
| Settings file | 3.9K | 2K-10K | ✓ Normal |
| Lock files | 0 | 0 | ✓ Normal |
| Database | 12K SQLite | N/A | ✓ Healthy |

## Diagnosis: Why `claude resume` Hangs

### Likely Root Causes (in order of probability):

1. **Session ID Resolution Failure**
   - The `instance_sessions` table tracks terminal PIDs
   - Current PID (118552) may not match expected session
   - `claude resume` might be waiting for user input to disambiguate sessions

2. **Global State Contention**
   - 187 session environments in `~/.claude/session-env/`
   - Multiple active claude processes (5 concurrent)
   - Possible lock contention in global state directory

3. **Database Query Blocking**
   - SQLite database might be waiting on another process
   - Though `lsof` shows no locks currently

4. **Hook Initialization**
   - TypeScript hooks need to compile/load
   - `edit-context-inject.mjs` and `signature-helper.mjs` may be slow

### NOT the cause:
- ✗ Corrupted local cache (cache is minimal and healthy)
- ✗ Bloated local directory (only 44K)
- ✗ File locks (none detected)
- ✗ Large agent outputs (only 13K oracle report)

## Recommended Actions

1. **Check what `claude resume` is waiting for:**
   ```bash
   strace -p <claude-resume-pid> 2>&1 | head -50
   ```

2. **List available sessions:**
   ```bash
   # Check if there are multiple resumable sessions
   ls -la ~/.claude/session-env/ | wc -l
   ```

3. **Clean old session state:**
   ```bash
   # Remove stale session environments older than 7 days
   find ~/.claude/session-env/ -mtime +7 -type d -exec rm -rf {} +
   ```

4. **Check hook performance:**
   ```bash
   time node ~/.claude/hooks/dist/edit-context-inject.mjs --help
   time node ~/.claude/hooks/dist/signature-helper.mjs --help
   ```

5. **Inspect database directly:**
   ```bash
   sqlite3 ~/.claude/cache/artifact-index/context.db "SELECT * FROM instance_sessions WHERE updated_at > datetime('now', '-1 day');"
   ```

## Architecture Notes

### Session Tracking System
- Each terminal gets a unique PID entry in `instance_sessions` table
- Sessions are named (e.g., "validation-output")
- Updated timestamp tracks last activity
- Used for resuming interrupted sessions and preventing conflicts

### Agent Cache Strategy
- Agents write markdown reports to `.claude/cache/agents/<agent-name>/`
- Only `oracle` has cached output (research task from previous session)
- No accumulation of large cache files

### Permission System
- `settings.local.json` stores pre-approved command patterns
- 101 entries covering bash commands, MCP tools, WebSearch/WebFetch
- Allows training scripts, Python, git, and data processing tools

## Open Questions

- ? UNCERTAIN: Why is PID 2322993 in database but not in `ps aux` output? (stale entry?)
- ? UNCERTAIN: Are there session resume prompts being hidden/buffered?
- ? UNCERTAIN: Is the global `~/.claude/session-env/` directory causing slowdown with 187 entries?

## File Citations

- **Local .claude/:** `/home/danny/Documents/personal/extension/training/.claude/`
- **Settings:** `/home/danny/Documents/personal/extension/training/.claude/settings.local.json`
- **Database:** `/home/danny/Documents/personal/extension/training/.claude/cache/artifact-index/context.db`
- **Oracle cache:** `/home/danny/Documents/personal/extension/training/.claude/cache/agents/oracle/latest-output.md`
- **Global cache:** `~/.claude/` (1.3GB, 187 session environments)
