"""Regression tests for the Strategy 2 AST-based Python sandbox policy."""

import os
import asyncio
import unittest
from unittest.mock import patch

from tool_sandbox import PythonSandbox


class PythonSandboxSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sandbox = PythonSandbox()

    def check(self, code: str) -> tuple[bool, str]:
        with patch.dict(os.environ, {"SANDBOX_OFF": ""}, clear=False):
            return self.sandbox._check_code_safety(code)

    def test_allows_safe_imports_and_comments_containing_from(self) -> None:
        safe, message = self.check(
            "# The result comes from the inclusion-exclusion formula.\n"
            "from math import gcd\n"
            "from itertools import combinations\n"
            "print(gcd(12, 18), len(list(combinations(range(3), 2))))\n"
        )
        self.assertTrue(safe, message)

    def test_allows_imported_symbol_name_not_in_module_allowlist(self) -> None:
        safe, message = self.check("from itertools import combinations\nprint(combinations)\n")
        self.assertTrue(safe, message)

    def test_rejects_non_allowlisted_import(self) -> None:
        safe, message = self.check("import os\n")
        self.assertFalse(safe)
        self.assertIn("Import of 'os' is not allowed", message)

    def test_rejects_dangerous_builtin_calls(self) -> None:
        for source, name in [("open('x')\n", "open"), ("eval('1 + 1')\n", "eval"), ("__import__('os')\n", "__import__")]:
            with self.subTest(source=source):
                safe, message = self.check(source)
                self.assertFalse(safe)
                self.assertIn(name, message)

    def test_rejects_dunder_attribute_access(self) -> None:
        safe, message = self.check("(1).__class__\n")
        self.assertFalse(safe)
        self.assertIn("Dunder attribute", message)

    def test_rejects_markdown_fence_as_invalid_python(self) -> None:
        safe, message = self.check("```py\nprint(1)\n```\n")
        self.assertFalse(safe)
        self.assertIn("Invalid Python syntax", message)

    def test_echoes_final_expression_result(self) -> None:
        result = asyncio.run(self.sandbox.execute_code("from math import comb\ncomb(6, 3)\n"))
        self.assertEqual(result, "Output:\n20")

    def test_does_not_echo_none_from_explicit_print(self) -> None:
        result = asyncio.run(self.sandbox.execute_code("print(20)\n"))
        self.assertEqual(result, "Output:\n20")


if __name__ == "__main__":
    unittest.main()
