"""Guard: Task 4 computes no digests anywhere (project rule, AGENTS.md).

Scans the task4 sources for hashlib usage and digest-field names, mirroring
the task2 no-hash assertions.
"""

import re
import unittest
from pathlib import Path

TASK4_DIR = Path(__file__).resolve().parents[1] / "tasks" / "task4"
DIGEST_FIELD = re.compile(r"sha256|md5|sha1|hashlib", re.IGNORECASE)


class TestNoHashing(unittest.TestCase):
    def test_sources_have_no_digest_references(self):
        offenders = []
        for path in sorted(TASK4_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for line_number, line in enumerate(text.splitlines(), 1):
                if DIGEST_FIELD.search(line):
                    offenders.append(f"{path.name}:{line_number}: {line.strip()}")
        self.assertEqual(offenders, [])

    def test_no_hashlib_import_anywhere(self):
        for path in sorted(TASK4_DIR.glob("*.py")):
            self.assertNotIn("hashlib", path.read_text(encoding="utf-8"), path.name)


if __name__ == "__main__":
    unittest.main()
