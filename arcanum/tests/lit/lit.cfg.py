# -*- Python -*-

import os
import lit.formats
import lit.util

config.name = "Arcanum"
config.suffixes = [".cpp"]
config.test_format = lit.formats.ShTest(bool(lit.util.which("bash")))
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.arcanum_obj_root

config.substitutions.append(("%arcanum", config.arcanum_path))
config.substitutions.append(("%FileCheck", config.filecheck_path + " --match-full-lines"))
config.substitutions.append(("%not", config.not_path))

if lit.util.which("bash"):
    config.available_features.add("shell")

import subprocess
if lit.util.which("why3"):
    # Check that Why3 is properly configured with z3 prover
    try:
        result = subprocess.run(
            ["why3", "config", "list-provers"],
            capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and "Z3" in result.stdout:
            config.available_features.add("why3")
    except (subprocess.TimeoutExpired, OSError):
        pass

config.environment["PATH"] = os.pathsep.join(
    [os.path.dirname(config.arcanum_path), config.environment.get("PATH", "")]
)
