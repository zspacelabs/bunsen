"""mdBook preprocessor: builds The Modelopedia into the book's src/modelopedia/ before the renderers run.

mdBook copies every non-Markdown file under src/ into the built book unchanged, so the generated HTML is published
beside the book at <site>/modelopedia/, and a chapter links to it. The book's chapters pass through untouched.

Wired up in book.toml:

    [preprocessor.modelopedia]
    command = "python3 modelopedia/preprocess.py"

It needs only Python 3's standard library.
"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        return 0  # every renderer: it only writes static files
    context, book = json.load(sys.stdin)
    root = Path(context["root"])
    src = context["config"]["book"].get("src", "src")
    import build
    build.build(root / src / "modelopedia")
    json.dump(book, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
