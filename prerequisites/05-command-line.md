# Prerequisite 5: Command Line Basics

A practical reference for the terminal commands you will use throughout **Building Production AI Systems**. This is not a comprehensive shell scripting guide -- it covers exactly what you need to run Python, manage environments, work with Docker, and navigate your projects.

---

## 1. Getting Started

### What is a Terminal?

A **terminal** (or terminal emulator) is a program that gives you a text-based interface to your operating system. Inside the terminal runs a **shell** -- a program that interprets the commands you type.

| Term | What it is |
|------|-----------|
| **Terminal** | The window/application (e.g., Terminal.app, iTerm2, Windows Terminal) |
| **Shell** | The command interpreter running inside the terminal |
| **bash** | The most common shell on Linux; default on older macOS |
| **zsh** | Default shell on macOS since Catalina (2019). Very similar to bash for daily use |

For this course, bash and zsh are interchangeable. Everything below works in both.

### Opening a Terminal

- **macOS**: Spotlight (`Cmd + Space`) and type "Terminal", or open iTerm2 if installed.
- **Linux**: `Ctrl + Alt + T` on most distributions.
- **Windows**: Use WSL2 (Windows Subsystem for Linux). Open "Ubuntu" from the Start menu after installing it.

### The Prompt

When you open a terminal you see something like this:

```
username@hostname ~ %
```

The `%` (zsh) or `$` (bash) is your **prompt** -- it means the shell is waiting for input. Throughout this guide, we use `$` to indicate the prompt. Do not type it.

### Running a Command

Type a command and press **Enter**:

```bash
$ echo "Hello, world!"
Hello, world!
```

---

## 2. Navigation

Your file system is a tree. The terminal always has a **current working directory** -- the folder you are "in" right now.

```bash
# Print the current working directory
$ pwd
/Users/yourname/projects

# List files and folders in the current directory
$ ls
data   models   app.py   requirements.txt

# List with details (permissions, size, date)
$ ls -la
total 24
drwxr-xr-x  6 you  staff   192 Jan 15 10:00 .
drwxr-xr-x  4 you  staff   128 Jan 10 09:00 ..
-rw-r--r--  1 you  staff   245 Jan 15 10:00 .env
drwxr-xr-x  3 you  staff    96 Jan 14 08:00 data
-rw-r--r--  1 you  staff  1024 Jan 15 10:00 app.py

# List with human-readable file sizes (KB, MB, GB)
$ ls -lh

# Change directory
$ cd projects
$ cd ..          # Go up one level (parent directory)
$ cd ~           # Go to your home directory (/Users/yourname)
$ cd ~/projects  # Go to a specific path from home
$ cd -           # Go back to the previous directory
```

**Tips:**
- Press **Tab** to autocomplete file and directory names. Press it twice to see all options.
- Type `clear` (or press `Ctrl + L`) to clear the screen.

---

## 3. File Operations

```bash
# Create a directory
$ mkdir my_project
$ mkdir -p my_project/src/utils   # Create nested directories in one go

# Create an empty file
$ touch README.md

# Copy a file
$ cp config.yaml config_backup.yaml

# Copy a directory (must use -r for recursive)
$ cp -r src/ src_backup/

# Move or rename a file
$ mv old_name.py new_name.py
$ mv file.py ../other_folder/     # Move to another directory

# Delete a file
$ rm unwanted_file.py

# Delete a directory and everything inside it
$ rm -r old_directory/
```

> **Warning:** `rm` is permanent. There is no trash can, no undo. Double-check before running `rm -r` on anything. Never run `rm -rf /` or `rm -rf ~`.

### Reading File Contents

```bash
# Print the entire file to the terminal
$ cat app.py

# Show only the first or last 10 lines
$ head app.py
$ tail app.py
$ head -n 20 app.py   # First 20 lines
$ tail -n 5 app.py    # Last 5 lines

# Scroll through a file interactively (press q to quit)
$ less app.py
```

---

## 4. Environment Variables

Environment variables are key-value pairs available to all programs in your shell session. They are how you pass configuration -- especially secrets -- to your applications.

```bash
# Set a variable for the current session
$ export OPENAI_API_KEY="sk-abc123..."

# Read a variable
$ echo $OPENAI_API_KEY
sk-abc123...

# See all environment variables
$ env
```

### The `.env` File

For this course, you will store API keys in a `.env` file rather than typing them every time:

```
# .env
OPENAI_API_KEY=sk-abc123...
ANTHROPIC_API_KEY=sk-ant-abc123...
DATABASE_URL=postgresql://user:pass@localhost:5432/mydb
```

Load it into your current shell session:

```bash
# "source" runs each line in the file as a shell command
$ source .env

# Some projects use "set -a" to auto-export all variables
$ set -a && source .env && set +a
```

> **Important:** Always add `.env` to your `.gitignore` file. Never commit secrets to Git.

### PATH

`PATH` is a special variable that tells the shell where to find programs:

```bash
$ echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin

# Add a directory to PATH for the current session
$ export PATH="$HOME/.local/bin:$PATH"
```

When you type a command like `python3`, the shell searches each directory in `PATH` (left to right) until it finds a matching executable.

---

## 5. Python Environment Management

### Running Python

```bash
# Check your Python version
$ python3 --version
Python 3.11.6

# Start an interactive Python session
$ python3

# Run a script
$ python3 app.py
```

> On some systems, `python` points to Python 2. Always use `python3` explicitly until you are inside a virtual environment.

### Virtual Environments

A virtual environment is an isolated Python installation. Each project should have its own to avoid dependency conflicts.

```bash
# Create a virtual environment in a folder called .venv
$ python3 -m venv .venv

# Activate it
# macOS / Linux:
$ source .venv/bin/activate

# Windows (PowerShell):
$ .venv\Scripts\Activate.ps1

# Windows (cmd):
$ .venv\Scripts\activate.bat
```

Once activated, your prompt changes:

```
(.venv) $ python --version   # Now "python" points to Python 3 inside the venv
Python 3.11.6
```

### Installing Packages

```bash
# Install a single package
(.venv) $ pip install fastapi

# Install all packages listed in a requirements file
(.venv) $ pip install -r requirements.txt

# Save your current packages to a file
(.venv) $ pip freeze > requirements.txt

# Deactivate when you're done
(.venv) $ deactivate
```

---

## 6. Process Management

When you run a command, it occupies your terminal until it finishes. For long-running processes (servers, workers), you need to manage them.

```bash
# Start a server (blocks the terminal)
$ uvicorn app:app --reload

# Stop it with Ctrl+C
^C

# Run something in the background (add &)
$ redis-server &
[1] 12345              # Job number and process ID (PID)

# List background jobs in this terminal session
$ jobs
[1]+  Running    redis-server &

# Bring a background job to the foreground
$ fg %1

# List all running processes
$ ps aux | grep python

# Kill a process by PID
$ kill 12345

# Force kill if it won't stop
$ kill -9 12345
```

Common long-running processes in this course:

```bash
# FastAPI dev server
$ uvicorn app.main:app --reload --port 8000

# Redis (used for caching / queues)
$ redis-server

# Celery worker
$ celery -A worker worker --loglevel=info
```

Open a **new terminal tab** (`Cmd + T` on macOS) to keep a server running while you work in another shell.

---

## 7. Pipes and Redirection

Pipes and redirection let you chain commands together and control where output goes.

### Redirection

```bash
# Write output to a file (overwrites)
$ echo "hello" > output.txt

# Append to a file
$ echo "world" >> output.txt

# Redirect errors to a file
$ python3 app.py 2> errors.log

# Redirect both stdout and stderr
$ python3 app.py > output.log 2>&1
```

### Pipes

The pipe `|` sends the output of one command as input to the next:

```bash
# Search for a pattern in output
$ cat requirements.txt | grep fastapi
fastapi==0.104.1

# Count lines in a file
$ wc -l app.py
     142 app.py

# Count how many Python files are in a directory tree
$ find . -name "*.py" | wc -l
23

# Search for "TODO" in all Python files
$ grep -r "TODO" --include="*.py" .

# Show running Python processes
$ ps aux | grep python
```

These are the pipe/redirection patterns you will actually use. That is all you need.

---

## 8. Docker Basics

Docker packages applications into **containers** -- isolated environments that run the same way everywhere. You will use Docker to run databases, Redis, and to deploy your AI systems.

### Essential Commands

```bash
# Run a container (pulls the image if you don't have it)
$ docker run -d --name my-redis -p 6379:6379 redis
#  -d          run in background (detached)
#  --name      give the container a name
#  -p 6379:6379   map host port to container port

# List running containers
$ docker ps

# List ALL containers (including stopped)
$ docker ps -a

# Stop a container
$ docker stop my-redis

# Remove a container
$ docker rm my-redis

# View container logs
$ docker logs my-redis
$ docker logs -f my-redis   # Follow (stream) logs in real time

# Run a command inside a running container
$ docker exec -it my-redis bash
```

### Docker Compose

Most projects in this course use `docker-compose.yml` to define multi-container setups (e.g., app + database + Redis).

```bash
# Start all services defined in docker-compose.yml
$ docker compose up

# Start in background
$ docker compose up -d

# Stop all services
$ docker compose down

# Rebuild images and start
$ docker compose up --build

# View logs for all services
$ docker compose logs -f
```

> Older installations use `docker-compose` (with a hyphen) instead of `docker compose` (with a space). Both work the same way.

---

## 9. Cheat Sheet

### Navigation

| Command | Description |
|---------|-------------|
| `pwd` | Print current directory |
| `ls` | List files |
| `ls -la` | List all files with details |
| `ls -lh` | List files with human-readable sizes |
| `cd <dir>` | Change directory |
| `cd ..` | Go up one level |
| `cd ~` | Go to home directory |
| `cd -` | Go to previous directory |
| `clear` | Clear the screen |

### Files and Directories

| Command | Description |
|---------|-------------|
| `mkdir <dir>` | Create a directory |
| `mkdir -p <path>` | Create nested directories |
| `touch <file>` | Create an empty file |
| `cp <src> <dst>` | Copy a file |
| `cp -r <src> <dst>` | Copy a directory |
| `mv <src> <dst>` | Move or rename |
| `rm <file>` | Delete a file |
| `rm -r <dir>` | Delete a directory (careful!) |
| `cat <file>` | Print file contents |
| `head -n N <file>` | First N lines |
| `tail -n N <file>` | Last N lines |
| `less <file>` | Scroll through a file |

### Environment Variables

| Command | Description |
|---------|-------------|
| `export KEY="value"` | Set a variable |
| `echo $KEY` | Print a variable |
| `source .env` | Load variables from a file |
| `env` | List all variables |
| `echo $PATH` | Show executable search path |

### Python

| Command | Description |
|---------|-------------|
| `python3 --version` | Check Python version |
| `python3 script.py` | Run a script |
| `python3 -m venv .venv` | Create virtual environment |
| `source .venv/bin/activate` | Activate venv (macOS/Linux) |
| `deactivate` | Deactivate venv |
| `pip install <pkg>` | Install a package |
| `pip install -r requirements.txt` | Install from file |
| `pip freeze > requirements.txt` | Save installed packages |

### Processes

| Command | Description |
|---------|-------------|
| `Ctrl + C` | Stop the current process |
| `command &` | Run in background |
| `jobs` | List background jobs |
| `fg %1` | Bring job 1 to foreground |
| `ps aux \| grep <name>` | Find a process |
| `kill <PID>` | Stop a process |
| `kill -9 <PID>` | Force stop a process |

### Pipes and Redirection

| Command | Description |
|---------|-------------|
| `cmd > file` | Write output to file |
| `cmd >> file` | Append output to file |
| `cmd1 \| cmd2` | Pipe output of cmd1 into cmd2 |
| `grep <pattern> <file>` | Search for a pattern |
| `grep -r <pattern> .` | Search recursively |
| `wc -l <file>` | Count lines |

### Docker

| Command | Description |
|---------|-------------|
| `docker run -d --name <n> <image>` | Run a container |
| `docker ps` | List running containers |
| `docker ps -a` | List all containers |
| `docker stop <name>` | Stop a container |
| `docker rm <name>` | Remove a container |
| `docker logs -f <name>` | Follow container logs |
| `docker exec -it <name> bash` | Shell into a container |
| `docker compose up -d` | Start all services |
| `docker compose down` | Stop all services |
| `docker compose up --build` | Rebuild and start |
| `docker compose logs -f` | Follow all service logs |

---

**You do not need to memorize any of this.** Bookmark this page. Use it as a reference. The commands will become second nature after the first few weeks of the course.
