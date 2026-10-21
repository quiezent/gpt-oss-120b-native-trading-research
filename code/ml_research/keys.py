"""API keys from the environment, falling back to the Windows user environment (a key set after this process started).
Never print or log a key."""
import os


def get(name):
    v = os.environ.get(name)
    if not v:
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as h:
                v = winreg.QueryValueEx(h, name)[0]
        except OSError:
            v = None
    return v
