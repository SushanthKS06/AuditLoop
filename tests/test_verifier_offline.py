"""The submission gate must stay offline and runtime-bounded.

Static regression guards on scripts/verify_submission.py:
- the demo smoke step must pass --no-llm (never the real provider path),
- every subprocess call must carry an explicit timeout,
- provider keys must be stripped from child environments.

The live-provider path lives separately in scripts/verify_live_llm.py.
"""

import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERIFIER = os.path.join(REPO, "scripts", "verify_submission.py")
LIVE_SCRIPT = os.path.join(REPO, "scripts", "verify_live_llm.py")


def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TestVerifierOffline:
    def test_demo_step_is_no_llm(self):
        src = _read(VERIFIER)
        assert "--no-llm" in src, (
            "verify_submission.py demo step must use --no-llm for offline determinism."
        )

    def test_no_unbounded_subprocess(self):
        src = _read(VERIFIER)
        runs = re.findall(r"subprocess\.run\((.*?)\)", src, re.DOTALL)
        assert runs, "Expected subprocess calls in the verifier."
        for call in runs:
            assert "timeout" in call, (
                f"Every subprocess.run in the verifier needs a timeout. Got: {call[:120]}"
            )

    def test_provider_keys_stripped_from_children(self):
        src = _read(VERIFIER)
        assert "GROQ_API_KEY" in src, (
            "Verifier must explicitly handle GROQ_API_KEY for child processes."
        )
        assert "OPENAI_API_KEY" in src

    def test_no_datetime_utcnow(self):
        src = _read(VERIFIER)
        assert "utcnow()" not in src, "Use timezone-aware datetime.now(timezone.utc)."

    def test_live_script_is_separate_and_optional(self):
        assert os.path.exists(LIVE_SCRIPT), "scripts/verify_live_llm.py must exist."
        src = _read(LIVE_SCRIPT)
        assert "NOT part of the submission gate" in src
        assert "GROQ_API_KEY" in src
