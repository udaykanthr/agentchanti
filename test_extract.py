import re
def _is_file_path(text: str) -> bool:
    text = text.strip()
    if ' ' in text: return False
    has_sep = '/' in text or '\\' in text
    has_ext = bool(re.search(r'\.\w{1,5}$', text))
    return has_sep or has_ext

def _looks_like_command(text: str) -> bool:
    text = text.strip()
    if not text: return False
    if _is_file_path(text): return False
    if re.match(r"^(source|gem|group|require)\s+['\"]", text): return False
    first_token = re.split(r'[\s>|&;<]', text)[0].lower()
    if first_token.endswith('.exe'): first_token = first_token[:-4]
    first_token = first_token.rstrip('.')
    known_commands = {
        'pip', 'pip3', 'python', 'python3', 'py',
        'npm', 'npx', 'node', 'yarn', 'pnpm',
        'go', 'cargo', 'rustc', 'mvn', 'gradle', 'javac', 'java',
        'ruby', 'bundle', 'gem', 'rspec',
        'git', 'docker', 'make', 'cmake',
        'mkdir', 'rmdir', 'del', 'copy', 'move', 'ren', 'type', 'dir',
        'ls', 'cat', 'cp', 'mv', 'rm', 'find', 'grep', 'chmod', 'chown',
        'cd', 'echo', 'set', 'export', 'source', 'touch',
        'curl', 'wget', 'ssh', 'scp',
        'apt', 'apt-get', 'brew', 'choco', 'yum', 'dnf', 'pacman',
        'powershell', 'pwsh', 'cmd',
        'pytest', 'jest', 'tox', 'mypy', 'flake8', 'black', 'ruff',
    }
    return first_token in known_commands

def _extract_commands_from_text(text: str) -> list[str]:
    commands = []
    seen = set()
    for m in re.finditer(r"```(?:\w*)\n(.*?)```", text, re.DOTALL):
        block = m.group(1).strip()
        if re.search(r'<<-?\s*[\'"]?(\w+)[\'"]?', block):
            if block not in seen:
                commands.append(block)
                seen.add(block)
        else:
            for line in block.splitlines():
                line = line.strip()
                if line and _looks_like_command(line) and line not in seen:
                    commands.append(line)
                    seen.add(line)
    for m in re.finditer(r"(?<!`)`([^`\n]+)`(?!`)", text):
        cmd = m.group(1).strip()
        if cmd and _looks_like_command(cmd) and cmd not in seen:
            commands.append(cmd)
            seen.add(cmd)
    return commands

print("Testing extract...")
print(_extract_commands_from_text("We need to output only the command: `pip install pygame pillow`. Let's produce that.pip install pygame pillow"))
print(_looks_like_command("We need to output only the command: `pip install pygame pillow`. Let's produce that.pip install pygame pillow"))
print(_looks_like_command("pip install pygame pillow"))

