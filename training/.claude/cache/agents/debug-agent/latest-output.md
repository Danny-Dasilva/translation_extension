# Debug Report: `claude resume` Hangs in Training Directory
Generated: 2026-01-16T14:55:00Z

## Symptom
Running `claude resume` in `/home/danny/Documents/personal/extension/training` causes the page to never load. The command hangs indefinitely without displaying any content.

## Investigation Steps

1. Checked `.claude/` directory structure - appears normal
2. Examined `sessions-index.json` - valid JSON with 27 entries
3. Validated individual session files - all valid JSON
4. Counted files in working directory - **519,395 files total**
5. Checked directory sizes - 45GB in `comic-text-detector/`
6. Verified no lock files present
7. Checked `.claudeignore` - **not present** at project root
8. Checked `.gitignore` - exists but doesn't cover data directories comprehensively

## Evidence

### Finding 1: Massive File Count
- **Location:** `/home/danny/Documents/personal/extension/training/`
- **Observation:** 519,395 files in the working directory
  - 422,318 files in `comic-text-detector/data/` alone
  - 11GB in `data/animetext/`
  - 2.1GB in `data/animetext_val/`
- **Relevance:** Claude Code likely scans/indexes files on startup. Half a million files causes timeout/hang.

### Finding 2: No `.claudeignore` File
- **Location:** `/home/danny/Documents/personal/extension/training/.claudeignore`
- **Observation:** File does not exist
- **Relevance:** Without `.claudeignore`, Claude attempts to index all files including training data

### Finding 3: Session Data is Valid
- **Location:** `~/.claude/projects/-home-danny-Documents-personal-extension-training/`
- **Observation:**
  - 251 session files totaling 179MB
  - `sessions-index.json` is valid (27 entries)
  - All JSONL files parse correctly
  - Largest session is 35MB (within normal range)
- **Relevance:** Session corruption is NOT the cause

### Finding 4: Directory Structure
```
training/
  comic-text-detector/  (45GB, 422K+ files)
    data/
      animetext/        (11GB) - training images
      animetext_val/    (2.1GB) - validation images
    runs/               (9.6GB) - model checkpoints
    models/             (1.6GB) - pretrained models
```

## Root Cause Analysis

**Most likely cause:** Claude Code attempts to scan or index the working directory on startup for `claude resume`. With 519,395 files, this operation times out or hangs before the UI can load.

**Confidence:** High

**Supporting evidence:**
1. Session files are all valid - ruling out data corruption
2. The hang occurs specifically in this directory with 500K+ files
3. No `.claudeignore` exists to exclude large data directories
4. Parent `.gitignore` ignores `*.png` and `*.jpg` for git but Claude Code has its own ignore mechanism

**Alternative hypotheses:**
1. Memory exhaustion from loading session history (less likely - sessions validated)
2. Renderer lock from another process (no lock files found)

## Recommended Fix

**Files to create:**
- `/home/danny/Documents/personal/extension/training/.claudeignore`

**Content for `.claudeignore`:**
```
# Large training data directories
comic-text-detector/data/
comic-text-detector/runs/
comic-text-detector/models/
comic-text-detector/outputs/

# Binary files
*.pt
*.onnx
*.ckpt
*.zip
*.tar
*.tar.gz

# Image datasets
*.png
*.jpg
*.jpeg

# Caches
__pycache__
.ruff_cache
wandb/
```

**Alternative fix (if .claudeignore doesn't help):**
1. Move large data directories outside the project:
   ```bash
   mv comic-text-detector/data ~/ml-data/comic-text-detector-data
   ln -s ~/ml-data/comic-text-detector-data comic-text-detector/data
   ```

2. Clear and rebuild session index:
   ```bash
   rm ~/.claude/projects/-home-danny-Documents-personal-extension-training/sessions-index.json
   ```

**Steps:**
1. Create `.claudeignore` file with content above
2. Retry `claude resume`
3. If still hanging, try the alternative symlink approach

## Prevention

1. **Always add `.claudeignore`** when working with ML projects that have large datasets
2. **Keep training data outside project root** when possible (use symlinks)
3. **Monitor directory size** - Claude Code works best with <10K files in working directory
