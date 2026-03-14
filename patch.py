import os

filepath = r"c:\Users\udayk\gitrepos\agentchanti\multi_agent_coder\orchestrator\step_handlers.py"
with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Insert _parse_echo_commands
insertion = '''
def _parse_echo_commands(text: str) -> dict[str, str]:
    """Parse shell-style file creation commands (echo "..." >> file)."""
    files: dict[str, str] = {}
    import re
    
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('>'):
            line = line[1:].strip()
        elif line.startswith('$'):
            line = line[1:].strip()
            
        parts = line.split(' && ')
        for part in parts:
            part = part.strip()
            
            m_nul = re.match(r'^(?:type\\s+nul|touch)\\s+>\\s+(.+)$', part)
            if not m_nul:
                m_nul = re.match(r'^touch\\s+(.+)$', part)
            if m_nul:
                fpath = m_nul.group(1).strip().strip('"\\'')
                fpath = fpath.replace('src.', 'src/')
                if fpath not in files:
                    files[fpath] = ""
                continue
            
            m_redir = re.search(r'\\s+(>>?>)\\s+([^>]+)$', part)
            if m_redir and part.startswith('echo '):
                operator = m_redir.group(1)
                fpath = m_redir.group(2).strip().strip('"\\'')
                fpath = fpath.replace('src.', 'src/')
                
                content_echo = part[5:m_redir.start()].strip()
                if content_echo.startswith('"') and content_echo.endswith('"'):
                    content_echo = content_echo[1:-1]
                    content_echo = content_echo.replace('\\\\"', '"').replace('\\\\n', '\\n')
                elif content_echo.startswith("'") and content_echo.endswith("'"):
                    content_echo = content_echo[1:-1]
                    content_echo = content_echo.replace("\\\\'", "'").replace('\\\\n', '\\n')
                else:
                    content_echo = content_echo.replace('\\\\n', '\\n')
                
                if operator == '>':
                    files[fpath] = content_echo + "\\n"
                else:
                    if fpath not in files:
                        files[fpath] = ""
                    files[fpath] += content_echo + "\\n"
                    
    return files


def _handle_code_step(step_text'''

if "_parse_echo_commands" not in content:
    content = content.replace("def _handle_code_step(step_text", insertion)

# 2. Modify _handle_code_step loop
start_str_code = "    for attempt in range(1, MAX_STEP_RETRIES + 1):\n"
end_str_code = "        if not files:\n            feedback = \"No file markers"

if ("inline_files =" not in content) and (start_str_code in content) and (end_str_code in content):
    loop_start = content.find(start_str_code)
    parse_end = content.find(end_str_code, loop_start)
    block = content[loop_start + len(start_str_code):parse_end]
    
    indented_block = ""
    for line in block.splitlines(True):
        if line.strip():
            indented_block += "    " + line
        else:
            indented_block += line
            
    # Insert _parse_echo_commands as fallback inside the block
    indented_block = indented_block.replace(
        "            files = executor.parse_code_blocks_fuzzy(response)\n",
        "            files = executor.parse_code_blocks_fuzzy(response)\n        if not files:\n            files = _parse_echo_commands(response)\n"
    )
    
    new_loop = f"""    inline_files = _parse_echo_commands(step_text)

    for attempt in range(1, MAX_STEP_RETRIES + 1):
        if attempt == 1 and inline_files:
            response = ""
            display.step_info(step_idx, "Extracted files directly from step instructions.")
            log.info(f"Step {{step_idx+1}}: Using inline files from step text.")
            explanation = ""
            files = inline_files
        else:
{indented_block}"""
    content = content[:loop_start] + new_loop + content[parse_end:]

# 3. Modify _handle_test_step loop
start_str_test = "    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):\n"
end_str_test = "        if not test_files:\n            feedback = \"No test files found"

if ("inline_test_files =" not in content) and (start_str_test in content) and (end_str_test in content):
    t_loop_start = content.find(start_str_test)
    t_parse_end = content.find(end_str_test, t_loop_start)
    t_block = content[t_loop_start + len(start_str_test):t_parse_end]
    
    t_indented = ""
    for line in t_block.splitlines(True):
        if line.strip():
            t_indented += "    " + line
        else:
            t_indented += line
            
    t_indented = t_indented.replace(
        "            test_files = executor.parse_code_blocks_fuzzy(test_response)\n",
        "            test_files = executor.parse_code_blocks_fuzzy(test_response)\n        if not test_files:\n            test_files = _parse_echo_commands(test_response)\n"
    )
    
    t_new_loop = f"""    inline_test_files = _parse_echo_commands(step_text)

    for gen_attempt in range(1, MAX_TEST_GEN_RETRIES + 1):
        if gen_attempt == 1 and inline_test_files:
            test_response = ""
            display.step_info(step_idx, "Extracted test files directly from step instructions.")
            log.info(f"Step {{step_idx+1}}: Using inline test files from step text.")
            explanation = ""
            test_files = inline_test_files
        else:
{t_indented}"""
    content = content[:t_loop_start] + t_new_loop + content[t_parse_end:]

with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)

print("Patch applied successfully.")
