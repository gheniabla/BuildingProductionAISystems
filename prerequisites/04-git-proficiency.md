# Prerequisite 4: Git Proficiency

Version control is non-negotiable in production AI work. This guide covers the Git skills you will use daily throughout this course. It is not a comprehensive Git reference -- it is the practical subset you actually need.

---

## Table of Contents

1. [Setup](#1-setup)
2. [Core Workflow](#2-core-workflow)
3. [Branching](#3-branching)
4. [Pull Requests](#4-pull-requests)
5. [Useful Commands](#5-useful-commands)
6. [Common Mistakes and Fixes](#6-common-mistakes-and-fixes)
7. [Cheat Sheet](#7-cheat-sheet)

---

## 1. Setup

### Install Git

Confirm Git is installed:

```bash
git --version
```

If not installed, get it from [git-scm.com](https://git-scm.com/) or via your package manager:

```bash
# macOS
brew install git

# Ubuntu/Debian
sudo apt-get install git
```

### Configure Your Identity

Every commit is tagged with your name and email. Set these once:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

A few other useful defaults:

```bash
# Set default branch name to main
git config --global init.defaultBranch main

# Set default editor (pick one)
git config --global core.editor "code --wait"   # VS Code
git config --global core.editor "nano"           # Nano

# Verify your config
git config --list
```

### SSH Keys (Brief)

SSH keys let you push and pull without typing your password every time.

```bash
# Generate a key (accept defaults, optionally set a passphrase)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Copy the public key to your clipboard
# macOS:
pbcopy < ~/.ssh/id_ed25519.pub
# Linux:
cat ~/.ssh/id_ed25519.pub
```

Then add the key in GitHub: **Settings > SSH and GPG keys > New SSH key**, paste, and save.

Test the connection:

```bash
ssh -T git@github.com
# Expected: "Hi username! You've successfully authenticated..."
```

### .gitignore for Python/AI Projects

Create a `.gitignore` in the root of every repository. For this course, you will want at minimum:

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/

# Virtual environments
.venv/
venv/
env/

# Environment variables and secrets
.env
.env.local
*.pem
*credentials*.json

# Model weights and large data files
*.pt
*.pth
*.onnx
*.h5
*.pkl
*.bin
models/
data/

# Jupyter notebook checkpoints
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

> **Rule of thumb:** If a file is generated, secret, or larger than a few MB, it should be in `.gitignore`.

---

## 2. Core Workflow

This is the loop you will repeat dozens of times per week.

### Starting a Project

```bash
# Clone an existing repository (you will do this for course assignments)
git clone git@github.com:org/repo-name.git
cd repo-name

# Or initialize a new repository from scratch
mkdir my-project && cd my-project
git init
```

### The Daily Loop

```
Edit files --> Stage changes --> Commit --> Push
```

**Step 1: Check what changed**

```bash
git status
```

This is your most-used command. Run it often.

**Step 2: Stage the files you want to commit**

```bash
# Stage specific files (preferred)
git add src/train.py src/config.yaml

# Stage everything in the current directory
git add .
```

**Step 3: Commit with a meaningful message**

```bash
git commit -m "Add learning rate scheduler to training loop"
```

Good commit messages are short, imperative, and describe *what* the change does:

| Good | Bad |
|------|-----|
| `Add data validation to pipeline input` | `updated stuff` |
| `Fix off-by-one error in batch indexing` | `fix bug` |
| `Remove unused model checkpoint loader` | `cleanup` |

For longer messages, omit `-m` and Git will open your editor:

```bash
git commit
```

Then write:

```
Add retry logic to API client

The OpenAI API occasionally returns 429 errors under load.
This adds exponential backoff with a maximum of 3 retries.
```

**Step 4: Push to the remote**

```bash
git push origin main
```

### Pulling Changes

Before you start working each day, pull the latest changes:

```bash
git pull origin main
```

If you are working on a branch (which you usually should be):

```bash
git pull origin main        # update your local main
git checkout your-branch
git merge main              # bring main's changes into your branch
```

### Full Daily Example

```bash
# Start of day
git pull origin main

# Work on your files...

# End of session
git status
git add src/evaluate.py tests/test_evaluate.py
git commit -m "Add BLEU score evaluation for translation model"
git push origin feature/evaluation-metrics
```

---

## 3. Branching

Never commit directly to `main` in a team project. Use branches.

### Creating and Switching Branches

```bash
# Create a new branch and switch to it (preferred)
git checkout -b feature/data-pipeline

# Modern alternative (Git 2.23+)
git switch -c feature/data-pipeline

# List all branches (* marks the current one)
git branch

# Switch to an existing branch
git checkout main
git switch main
```

### Branch Naming Conventions

Use prefixes to communicate the purpose of the branch:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `feature/` | New functionality | `feature/add-embeddings-cache` |
| `bugfix/` | Fix a bug | `bugfix/tokenizer-overflow` |
| `experiment/` | Try something (may be thrown away) | `experiment/llama-finetuning` |
| `docs/` | Documentation only | `docs/api-reference` |
| `refactor/` | Code restructuring | `refactor/split-training-module` |

Keep names lowercase, use hyphens, and keep them short but descriptive.

### Feature Branch Workflow

This is the workflow you will use for every assignment and project:

```bash
# 1. Start from an up-to-date main
git checkout main
git pull origin main

# 2. Create your feature branch
git checkout -b feature/prompt-caching

# 3. Do your work, committing as you go
git add src/cache.py
git commit -m "Add LRU cache for prompt templates"

git add src/cache.py tests/test_cache.py
git commit -m "Add cache invalidation and unit tests"

# 4. Push your branch to the remote
git push -u origin feature/prompt-caching
# The -u flag sets up tracking so future pushes are just: git push

# 5. Open a pull request (see Section 4)

# 6. After the PR is merged, clean up
git checkout main
git pull origin main
git branch -d feature/prompt-caching
```

### Merging

```bash
# Merge another branch into your current branch
git checkout main
git merge feature/prompt-caching
```

### Handling Merge Conflicts

Conflicts happen when two branches modify the same lines. Git marks the conflicts in the file:

```
<<<<<<< HEAD
learning_rate = 0.001
=======
learning_rate = 0.0005
>>>>>>> feature/tune-hyperparams
```

To resolve:

1. Open the file and decide which version to keep (or combine them).
2. Remove the conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
3. Stage and commit the resolution.

```bash
# After editing the file to resolve the conflict
git add src/config.py
git commit -m "Resolve merge conflict in learning rate config"
```

> **Tip:** VS Code highlights conflicts and provides one-click resolution buttons. Use them.

---

## 4. Pull Requests

Pull requests (PRs) are how changes get reviewed and merged into `main` in team projects.

### Creating a PR with the GitHub CLI

Install the GitHub CLI if you have not already:

```bash
# macOS
brew install gh

# Authenticate
gh auth login
```

Create a PR from your current branch:

```bash
gh pr create --title "Add prompt caching layer" --body "## Summary
- Implements LRU cache for prompt templates
- Reduces redundant API calls by ~40%

## Testing
- Unit tests in tests/test_cache.py
- Manual test against staging endpoint"
```

Or open the interactive flow:

```bash
gh pr create
```

### Creating a PR on the GitHub Web UI

1. Push your branch.
2. Go to the repository on GitHub.
3. Click **"Compare & pull request"** on the banner that appears.
4. Fill in the title and description.
5. Click **"Create pull request"**.

### Writing Good PR Descriptions

A good PR description answers three questions:

- **What** does this change do?
- **Why** is this change needed?
- **How** can someone test or verify it?

```markdown
## What
Adds a caching layer for LLM prompt templates using an LRU cache.

## Why
Repeated identical prompts were causing unnecessary API calls,
increasing latency and cost.

## How to Test
Run `pytest tests/test_cache.py`. Verify cache hits in logs
when running the pipeline twice with the same input.
```

### Code Review Basics

When reviewing a teammate's PR:

- **Read the description first** to understand the intent.
- **Check for correctness**, not just style.
- **Be specific** in feedback: point to exact lines, suggest fixes.
- **Approve** when you are satisfied; **request changes** when something must be fixed.

Using the CLI to review:

```bash
# List open PRs
gh pr list

# View a specific PR
gh pr view 42

# Check out a PR locally to test it
gh pr checkout 42

# Approve
gh pr review 42 --approve

# Request changes
gh pr review 42 --request-changes --body "The cache has no TTL -- stale prompts will never expire."
```

### Merging Strategies

When merging a PR, GitHub offers three options:

| Strategy | What It Does | When to Use |
|----------|-------------|-------------|
| **Merge commit** | Creates a merge commit preserving all branch commits | Default; preserves full history |
| **Squash and merge** | Combines all branch commits into a single commit | Clean history; many small WIP commits on branch |
| **Rebase and merge** | Replays commits on top of main (no merge commit) | Linear history; clean, well-structured commits |

For this course, **squash and merge** is recommended for most PRs. It keeps `main` clean.

```bash
# Merge a PR via CLI
gh pr merge 42 --squash
```

---

## 5. Useful Commands

### Viewing History

```bash
# Compact one-line log
git log --oneline

# Log with branch graph
git log --oneline --graph --all

# Show last 5 commits
git log --oneline -5

# Show commits by a specific author
git log --author="Your Name" --oneline
```

### Viewing Changes

```bash
# See unstaged changes
git diff

# See staged changes (what will be committed)
git diff --staged

# Compare two branches
git diff main..feature/data-pipeline

# Show changes in a specific file
git diff src/train.py
```

### Stashing Work

Stash lets you save uncommitted changes temporarily without committing them.

```bash
# Stash your current changes
git stash

# List stashes
git stash list

# Restore the most recent stash
git stash pop

# Restore a specific stash
git stash apply stash@{2}

# Drop a stash
git stash drop stash@{0}
```

Common use case: you are mid-work on a branch and need to switch to `main` quickly.

```bash
git stash
git checkout main
# ... do something on main ...
git checkout feature/my-branch
git stash pop
```

### Undoing Things

**`git reset --soft`** -- Undo commits but keep changes staged:

```bash
# Undo the last commit, keep changes staged
git reset --soft HEAD~1
```

**`git reset --hard`** -- Undo commits and discard all changes:

```bash
# DANGER: This permanently deletes uncommitted work
git reset --hard HEAD~1
```

> **Warning:** `git reset --hard` is destructive. There is no undo. Only use it when you are certain you want to throw away changes.

**`git revert`** -- Create a new commit that undoes a previous commit (safe for shared branches):

```bash
# Revert a specific commit (by hash)
git revert a1b2c3d
```

Use `revert` instead of `reset` on branches that others are working on.

### .gitignore Patterns

| Pattern | Matches |
|---------|---------|
| `*.log` | All `.log` files in any directory |
| `data/` | The `data` directory and everything inside it |
| `!data/README.md` | Exception: track `README.md` inside `data/` |
| `*.pt` | All PyTorch model weight files |
| `secret_*` | Any file starting with `secret_` |
| `**/__pycache__/` | `__pycache__` directories at any depth |

If a file is already tracked, adding it to `.gitignore` will not stop tracking it. You need to untrack it first:

```bash
git rm --cached path/to/file
git commit -m "Stop tracking file that should be ignored"
```

---

## 6. Common Mistakes and Fixes

### Committed Secrets (API Keys, .env Files)

This is the most serious mistake. Once a secret is pushed to a remote, assume it is compromised.

**Immediate steps:**

1. **Rotate the secret.** Generate a new API key or password. Do this *first*.
2. Remove the file and add it to `.gitignore`:

```bash
echo ".env" >> .gitignore
git rm --cached .env
git commit -m "Remove .env from tracking and add to gitignore"
git push
```

3. If the repository is public and the secret was in the commit history, you need to rewrite history using [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) or `git filter-repo`:

```bash
# Using git-filter-repo (install: pip install git-filter-repo)
git filter-repo --path .env --invert-paths
```

> **Prevention:** Always create your `.gitignore` before your first commit. Always use environment variables for secrets.

### Committed Large Files (Model Weights, Datasets)

Git is not designed for large binary files. If you accidentally committed a 2GB model file:

```bash
# Remove from tracking (keeps the local file)
git rm --cached models/large_model.pt
echo "models/" >> .gitignore
git commit -m "Remove large model file from tracking"
```

If it is already in the history and bloating the repo:

```bash
git filter-repo --path models/large_model.pt --invert-paths
```

For legitimate large file needs, use [Git LFS](https://git-lfs.github.com/):

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

### Committed to the Wrong Branch

You made commits on `main` that should have been on a feature branch:

```bash
# Create a new branch at the current position (saves your commits)
git branch feature/my-work

# Move main back to where it should be
git reset --hard origin/main

# Switch to your feature branch with your work intact
git checkout feature/my-work
```

### Undo the Last Commit

```bash
# Keep the changes (just undo the commit)
git reset --soft HEAD~1

# Discard the changes entirely
git reset --hard HEAD~1
```

### Accidentally Deleted a File

If you have not committed the deletion:

```bash
git checkout -- path/to/file
```

If you committed the deletion:

```bash
git revert HEAD
```

---

## 7. Cheat Sheet

### Setup

| Command | Description |
|---------|-------------|
| `git config --global user.name "Name"` | Set your name |
| `git config --global user.email "email"` | Set your email |
| `git init` | Initialize a new repository |
| `git clone <url>` | Clone a remote repository |

### Daily Workflow

| Command | Description |
|---------|-------------|
| `git status` | Show changed files |
| `git add <file>` | Stage a file for commit |
| `git add .` | Stage all changes |
| `git commit -m "message"` | Commit staged changes |
| `git push` | Push commits to remote |
| `git pull` | Pull latest changes from remote |

### Branching

| Command | Description |
|---------|-------------|
| `git branch` | List branches |
| `git checkout -b <name>` | Create and switch to a new branch |
| `git switch -c <name>` | Create and switch to a new branch (modern) |
| `git checkout <name>` | Switch to existing branch |
| `git merge <branch>` | Merge branch into current branch |
| `git branch -d <name>` | Delete a branch (after merge) |

### Pull Requests (GitHub CLI)

| Command | Description |
|---------|-------------|
| `gh pr create` | Create a pull request (interactive) |
| `gh pr list` | List open pull requests |
| `gh pr view <number>` | View PR details |
| `gh pr checkout <number>` | Check out a PR locally |
| `gh pr merge <number> --squash` | Squash and merge a PR |

### Inspecting and Comparing

| Command | Description |
|---------|-------------|
| `git log --oneline` | Compact commit log |
| `git log --oneline --graph --all` | Visual branch graph |
| `git diff` | Show unstaged changes |
| `git diff --staged` | Show staged changes |
| `git diff main..<branch>` | Compare branch to main |

### Undoing Things

| Command | Description |
|---------|-------------|
| `git stash` | Temporarily save uncommitted changes |
| `git stash pop` | Restore stashed changes |
| `git reset --soft HEAD~1` | Undo last commit, keep changes staged |
| `git reset --hard HEAD~1` | Undo last commit, **discard changes** |
| `git revert <hash>` | Create a new commit that undoes a commit |
| `git rm --cached <file>` | Stop tracking a file without deleting it |

### Safety Rules for This Course

1. **Never commit secrets.** Set up `.gitignore` before your first commit.
2. **Never commit model weights or large data files.** Use `.gitignore` or Git LFS.
3. **Never push directly to `main`.** Use feature branches and pull requests.
4. **Pull before you push.** Run `git pull` at the start of each session.
5. **Commit often, push often.** Small, frequent commits are easier to review and revert.

---

**Next:** [Prerequisite 5 >>](./05-docker-basics.md)
