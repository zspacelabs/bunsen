# The Modelopedia

An architecture book of bunsen's image, language and speech model families, read as one modular algebra. It is
hand-built HTML and SVG, not mdBook Markdown, and is published beside the Bunsen Book at `modelopedia/`.

- `pages/`: one HTML fragment per page, with placeholders: `{{fig:name}}`, `{{circuit:key|note}}`,
  `{{cite:key}}`, `{{tree}}`, `{{matrix:…}}`, `{{refs}}`, `{{regnet:…}}`.
- `build.py`: the page list, the merged bibliography, the lineage tree and overview, and the page shell.
- `img/`: conv nets and vision transformers: figures (`figs.py`, `figs_tx.py`), variant circuits
  (`circuits.py`), the tree and slot matrices (`build.py`), and references (`refs.py`).
- `lng/`: language and speech models, laid out the same way, with the circuit boards in `board.py` and
  `board_sp.py`.
- `assets/`: the stylesheet and the theme/tree script.
- `preprocess.py`: an mdBook preprocessor that runs `build.py` on every `mdbook build`.

The output goes to `book/src/modelopedia/` (gitignored), which mdBook copies into the built book unchanged.
`python3 build.py` builds it alone; open `../src/modelopedia/index.html`. Only Python 3's standard library is
needed.
