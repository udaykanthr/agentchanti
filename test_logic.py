from multi_agent_coder.orchestrator.classification import _extract_commands_from_text, _looks_like_command

def parse_cmd(cmd_response):
    extracted = _extract_commands_from_text(cmd_response)
    if extracted:
        cmd = extracted[0]
    else:
        cmd = cmd_response.strip('`').strip()
        if not _looks_like_command(cmd):
            for line in cmd.splitlines():
                line = line.strip('`').strip()
                if _looks_like_command(line):
                    cmd = line
                    break
    return cmd

print(parse_cmd("We need to output only the command: `pip install pygame pillow`. Let's produce that.pip install pygame pillow"))
print(parse_cmd("pip install pygame pillow"))
print(parse_cmd("Just run this:\npip install pygame pillow"))
print(parse_cmd("`pytest`"))
