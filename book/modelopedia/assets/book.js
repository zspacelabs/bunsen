// The Modelopedia: theme toggle and lineage-tree tools.
(function () {
  var root = document.documentElement;
  var btn = document.getElementById('theme');
  if (btn) btn.addEventListener('click', function () {
    var dark = root.dataset.theme
      ? root.dataset.theme === 'dark'
      : window.matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('mp-theme', root.dataset.theme); } catch (e) {}
  });

  var tree = document.getElementById('lineage');
  if (!tree) return;
  var tools = document.querySelector('.tree-tools');
  tools.addEventListener('click', function (ev) {
    var act = ev.target.getAttribute && ev.target.getAttribute('data-act');
    if (!act) return;
    tree.querySelectorAll('details').forEach(function (d) { d.open = act === 'open'; });
  });
  var input = tools.querySelector('input');
  input.addEventListener('input', function () {
    var q = input.value.trim().toLowerCase();
    var lis = tree.querySelectorAll('li');
    lis.forEach(function (li) { li.classList.remove('match', 'miss'); });
    if (!q) return;
    lis.forEach(function (li) {
      var hit = (li.getAttribute('data-s') || '').indexOf(q) >= 0;
      li.classList.add(hit ? 'match' : 'miss');
      if (hit) {
        var p = li.parentElement;
        while (p && p !== tree) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
      }
    });
  });
})();
