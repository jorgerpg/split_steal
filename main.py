import subprocess


def run_headless_scripts():
  subprocess.run(["python3", "-m", "NSH.newSplitHeadless"])
  subprocess.run(["python3", "-m", "NSHL.newSplitHeadlessLogic"])


if __name__ == "__main__":
  run_headless_scripts()
