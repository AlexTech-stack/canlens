# Copyright 2026 Alexander Günther
# SPDX-License-Identifier: MIT

"""Allow `python3 -m canlens` without an install."""
import sys

from .cli import main

sys.exit(main())
