import os

def test_basic_layout():
    required = ["configs", "figures", "scripts", "src", "requirements.txt", "README.md"]
    for p in required:
        assert os.path.exists(p), f"Missing required path: {p}"
