"""Streamlit smoke tests: mock login behaviour and sidebar controls. Runs in a temp working directory so the real vector store is untouched."""

import os
import tempfile
import unittest
from streamlit.testing.v1 import AppTest
from tests import PROJECT_ROOT

APP = str(PROJECT_ROOT / "streamlit_app.py")


class StreamlitAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._old_cwd = os.getcwd()
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)  # Windows keeps Chroma's sqlite file open
        os.chdir(cls._tmp.name)  # relative "./vector_store" and "data" now point at an empty scratch dir

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls._old_cwd)
        cls._tmp.cleanup()

    def fresh(self):
        return AppTest.from_file(APP, default_timeout=120).run()

    def test_login_page_shows_prefilled_demo_credentials(self):
        at = self.fresh()
        self.assertEqual(len(at.exception), 0)
        values = {i.label: i.value for i in at.text_input}
        self.assertEqual(values, {"Username": "admin@tesla.demo", "Password": "demo123"})
        self.assertFalse(at.session_state["authenticated"])

    def test_any_nonempty_username_logs_in_and_password_is_not_checked(self):
        at = self.fresh()
        at.text_input[0].set_value("someone@example.com")
        at.text_input[1].set_value("totally-wrong-password")
        at.button[0].click().run()
        self.assertTrue(at.session_state["authenticated"])
        self.assertEqual(at.session_state["username"], "someone@example.com")

    def test_blank_username_is_rejected(self):
        at = self.fresh()
        at.text_input[0].set_value("   ")
        at.button[0].click().run()
        self.assertFalse(at.session_state["authenticated"])
        self.assertIn("Please enter a username.", [e.value for e in at.error])

    def test_sidebar_controls_and_defaults_after_login(self):
        at = self.fresh()
        at.button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        sliders = {s.label: (s.value, s.min, s.max, s.step) for s in at.slider}
        self.assertEqual(
            sliders,
            {
                "Top K Chunks": (3, 1, 5, 1),
                "Temperature": (0.0, 0.0, 1.0, 0.1),
                "Top P": (0.1, 0.0, 1.0, 0.1),
            },
        )

    def test_logout_returns_to_login(self):
        at = self.fresh()
        at.button[0].click().run()
        logout = next(b for b in at.button if "Logout" in b.label)
        logout.click().run()
        self.assertFalse(at.session_state["authenticated"])


if __name__ == "__main__":
    unittest.main()
