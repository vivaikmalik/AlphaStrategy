import os

def build_main():
    """Compiles src/ into a single MAIN.py[cite: 4]."""
    modules = ["config.py", "data.py", "gru.py", "text.py", "kalman.py", "ranker.py", "optimizer.py", "metrics.py"]
    with open("MAIN.py", "w") as outfile:
        outfile.write("# MAIN.py - Auto-compiled AlphaStrategy Submission\n\n")
        for module in modules:
            path = os.path.join("src", module)
            if os.path.exists(path):
                with open(path, "r") as infile:
                    # Strip local relative imports for single-file structure
                    lines = [line for line in infile.readlines() if not line.startswith("from src.")]
                    outfile.writelines(lines)
                    outfile.write("\n\n")
        outfile.write("if __name__ == '__main__':\n    print('Running end-to-end pipeline...')\n")
    print("✅ MAIN.py successfully built.")

if __name__ == "__main__":
    build_main()