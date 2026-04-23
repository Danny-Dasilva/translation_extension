# Research Report: Claude Code CLI `resume` Command Hanging/Not Loading
Generated: 2026-01-16

## Summary

The `claude --resume` and `claude --continue` commands have well-documented hanging/freezing issues across multiple platforms (Windows, macOS, Linux/WSL). These issues stem from session file corruption, platform-specific terminal handling bugs, and multi-instance conflicts. Several workarounds exist ranging from simple cache clearing to complete reinstallation.

## Questions Answered

### Q1: Known issues with `claude resume` command
**Answer:** Multiple GitHub issues document resume command hanging across all platforms. Windows is most affected with terminal freezes that even Ctrl+C cannot interrupt. The issue affects versions from 1.0.9 through current 2.x releases.
**Source:** [GitHub Issue #9844](https://github.com/anthropics/claude-code/issues/9844), [GitHub Issue #7455](https://github.com/anthropics/claude-code/issues/7455)
**Confidence:** High

### Q2: Common causes of Claude Code CLI hanging
**Answer:** Multiple causes identified:
- Session file corruption (especially after Ctrl+C interruption)
- Running multiple Claude Code instances in same folder
- Large file reads corrupting session state
- Platform-specific terminal handling issues (especially Windows)
- Context overload from accumulated session history
**Source:** [GitHub Issue #6823](https://github.com/anthropics/claude-code/issues/6823), [GitHub Issue #13224](https://github.com/anthropics/claude-code/issues/13224)
**Confidence:** High

### Q3: How Claude Code stores session state and what can corrupt it
**Answer:** Sessions stored in `~/.claude/projects/<path-hash>/session-id.jsonl`. Corruption can occur from:
- Reading files >5MB (exceeds API limit, corrupts session permanently)
- Interrupted operations (Ctrl+C during writes)
- File-history feedback loops (can consume 300GB+)
- Corrupt JSON in `.claude.json` config
**Source:** [GitHub Issue #6780](https://github.com/anthropics/claude-code/issues/6780), [GitHub Issue #10107](https://github.com/anthropics/claude-code/issues/10107)
**Confidence:** High

### Q4: Solutions others have found
**Answer:** Multiple solutions documented below in recommendations section.
**Source:** [ClaudeLog Troubleshooting](https://claudelog.com/troubleshooting/), [Luiz Tanure Blog](https://www.letanure.dev/blog/2025-08-09--claude-code-part-11-troubleshooting-recovery)
**Confidence:** High

## Detailed Findings

### Finding 1: Platform-Specific Issues

**Source:** Multiple GitHub issues

**Windows:**
- Terminal freeze is most severe on Windows
- Standard interrupt signals (Ctrl+C, Ctrl+Z) cannot terminate process
- Requires force-kill via Task Manager
- Issue persists across reinstalls indicating system-level state corruption
- Last known stable version: 1.0.85

**macOS:**
- `--resume` often doesn't show most recent session
- Need to press Esc, retry, then session appears
- Multiple instances cause freeze and CPU exhaustion

**WSL:**
- Hangs forever on Bash/shell commands
- Gets stuck at "Envisioning..." or "Musing..."
- Affects even native Linux paths (not just /mnt/c)

### Finding 2: Session Storage Architecture

**Source:** [Vincent Schmalbach Blog](https://www.vincentschmalbach.com/migrate-claude-code-sessions-to-a-new-computer/)

**Key Points:**
- Sessions stored in `~/.claude/projects/` organized by project path
- Format: `~/.claude/projects/-home-user-project/session-id.jsonl`
- Additional files in `~/.claude/sessions/`
- Configuration in `~/.claude.json`

**Corruption Detection:**
```bash
# Find sessions with excessive snapshots (corruption indicator)
find ~/.claude/projects/ -name "*.jsonl" -exec sh -c \
  'grep -c "\"type\":\"file-history-snapshot\"" "$1"' sh {} \; | sort -n | tail -5
```

### Finding 3: Known Regression - Large File Corruption

**Source:** [GitHub Issue #6780](https://github.com/anthropics/claude-code/issues/6780)

**Key Points:**
- Read tool allows files >5MB (exceeds Claude API limit)
- Causes irreversible session corruption
- All subsequent sessions in project inherit corrupted data
- Requires manual JSON editing to recover
- Introduced with cloud context retrieval changes

## Comparison Matrix: Solutions

| Solution | Invasiveness | Preserves Sessions | Effectiveness | Use When |
|----------|--------------|-------------------|---------------|----------|
| Clear cache (`--clear-cache`) | Low | Yes | Medium | First attempt |
| Use `--continue` instead | Low | Yes | Medium | `--resume` specific issues |
| Delete specific session | Medium | Partial | High | Known corrupted session |
| Clear sessions folder | High | No | High | Persistent issues |
| Full reinstall | Highest | No | Highest | Nothing else works |

## Recommendations

### For This Issue - Try These In Order

**1. First: Clear Cache**
```bash
claude --clear-cache
```

**2. Use Alternative Command**
```bash
# Instead of --resume, try:
claude --continue

# Or start fresh and use /resume in REPL:
claude
> /resume
```

**3. Delete Session Cache**
```bash
rm ~/.claude/sessions/*.json
```

**4. Check for Corrupted Sessions**
```bash
# Look for suspiciously large session files
ls -la ~/.claude/projects/*/
find ~/.claude/projects/ -name "*.jsonl" -size +10M
```

**5. Nuclear Option - Full Reset**
```bash
npm uninstall -g @anthropic-ai/claude-code
rm ~/.claude.json
rm -rf ~/.claude/
npm cache clean --force
npm install -g @anthropic-ai/claude-code
```

### Implementation Notes

- **Backup first**: Before deleting anything, backup `~/.claude/` if you have important session history
- **Windows users**: May need to downgrade to version 1.0.85 if issues persist
- **WSL users**: Try running from native Linux paths, not /mnt/c mounted paths
- **Multi-instance**: Avoid running multiple Claude Code instances in same folder
- **Context management**: Use `/compact` when context >70% to prevent corruption

### Prevention

1. Use `/compact` regularly to manage context size
2. Avoid reading very large files (>5MB)
3. Don't Ctrl+C during file operations
4. Run only one Claude Code instance per project folder
5. Regular backups of `~/.claude/` directory

## Sources

1. [GitHub Issue #9844 - "claude --resume" hangs terminal window](https://github.com/anthropics/claude-code/issues/9844)
2. [GitHub Issue #7455 - Freezing Issues at Session Resuming](https://github.com/anthropics/claude-code/issues/7455)
3. [GitHub Issue #6823 - Claude CLI 1.0.96 randomly freezes on Windows](https://github.com/anthropics/claude-code/issues/6823)
4. [GitHub Issue #13224 - Claude Code hangs or freezes during execution](https://github.com/anthropics/claude-code/issues/13224)
5. [GitHub Issue #6780 - Irreversible Session Corruption with Large Files](https://github.com/anthropics/claude-code/issues/6780)
6. [GitHub Issue #10107 - File-history causes catastrophic disk exhaustion](https://github.com/anthropics/claude-code/issues/10107)
7. [ClaudeLog Troubleshooting Guide](https://claudelog.com/troubleshooting/)
8. [Claude Code Troubleshooting and Recovery - Luiz Tanure](https://www.letanure.dev/blog/2025-08-09--claude-code-part-11-troubleshooting-recovery)
9. [Vincent Schmalbach - Migrate Claude Code Sessions](https://www.vincentschmalbach.com/migrate-claude-code-sessions-to-a-new-computer/)
10. [Steve Kinney - Claude Code Session Management](https://stevekinney.com/courses/ai-development/claude-code-session-management)
11. [Official Claude Code Troubleshooting Docs](https://code.claude.com/docs/en/troubleshooting)

## Open Questions

- No official fix timeline from Anthropic for the Windows freezing issues
- Whether the large file corruption issue (#6780) has been fully patched
- Root cause of WSL-specific hanging behavior not fully diagnosed
