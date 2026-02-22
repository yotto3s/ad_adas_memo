"""Lit configuration for Arcanum integration tests."""

import os
import lit.formats

config.name = "Arcanum"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".cpp"]
config.test_source_root = os.path.dirname(__file__)

# The arcanum binary is expected on PATH (set via --path in the lit invocation)
# or in the build directory.
config.substitutions.append(("%arcanum", "arcanum"))
config.substitutions.append(
    ("%FileCheck", "FileCheck")
)

# Environment passthrough for tools
config.environment["PATH"] = os.environ.get("PATH", "")
