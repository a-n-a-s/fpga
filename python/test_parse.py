import subprocess
import re

def run(cmd, cwd):
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    return p.returncode, p.stdout, p.stderr

def parse_rtl_class(sim_out: str):
    m = re.search(r"Predicted Class:\s*(\d+)", sim_out)
    if not m:
        return None
    return int(m.group(1))

# Test
rc, out, err = run(['vvp', 'simv', '+INPUT_FILE=temp_window.mem'], cwd='D:\\serious_done')
print(f"Return code: {rc}")
print(f"Stdout length: {len(out)}")
print(f"Stderr length: {len(err)}")

rtl_cls = parse_rtl_class(out)
print(f"Parsed class: {rtl_cls}")

if rtl_cls is None:
    print("\n=== Could not find 'Predicted Class' in output ===")
    print("\nLast 500 chars of stdout:")
    print(out[-500:])
