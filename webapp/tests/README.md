# Dashboard tests

No test framework and no browser: a minimal DOM shim (`domshim.mjs`) is enough to
catch the failures that matter here — a view throwing on real data leaves a blank
page, and that is invisible until someone opens the tab during a demo.

    # 1. start the server, capture a snapshot, copy the modules
    python webapp/serve.py --port 8140 &
    curl -s localhost:8140/api/snapshot > /tmp/snap.json
    for f in svg store sse views app; do cp webapp/static/js/$f.js /tmp/$f.mjs; done
    sed -i "s|\.js'|.mjs'|g" /tmp/views.mjs /tmp/sse.mjs /tmp/app.mjs
    cp webapp/tests/*.mjs /tmp/

    # 2. every view renders, with data and in the loading state
    node /tmp/viewtest.mjs

    # 3. live reducer: replay is idempotent, and a live run has the SAME key set
    #    as a stored one -- the invariant that keeps views to one render path
    cp fl_rnaseq/results/live/*.jsonl /tmp/ev.jsonl
    node /tmp/livetest.mjs

`test_normalize.py` runs against the real result files with the system Python and
no venv, which also proves the dashboard needs nothing from `.venv-fl`.
