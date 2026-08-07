# Deploy — 10 minutes

Run these from `C:\DataAnalysis\fpl-agent\gameweek-desk`.

## 1. Sanity check locally (2 min)

```powershell
pip install -r requirements.txt
python test_smoke.py
streamlit run app.py
```

`test_smoke.py` should end with `ALL CHECKS PASSED`. Click through the app once:
Run analysis → check the escalations appear → publish one → check the Audit Log page.

## 2. Push to GitHub (3 min)

First create the repo: [github.com/new](https://github.com/new) → name `gameweek-desk` →
**public** → do **not** tick "Add a README" (one already exists here) → Create.

Then, replacing `YOUR_USERNAME` with your actual GitHub handle:

```powershell
git init
git add .
git commit -m "Gameweek Desk: availability verification and editorial review console"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gameweek-desk.git
git push -u origin main
```

A browser window opens for GitHub auth on the first push. If you already added a remote
with the wrong URL, run `git remote remove origin` before re-adding it.

`warning: LF will be replaced by CRLF` on Windows is cosmetic — ignore it.

`.gitignore` already excludes `secrets.toml`, the audit log, and the old virtualenvs.

> Note: the two old `venv/` and `myenv/` folders sit one level up in `fpl-agent`, outside
> this directory, so they will not be committed. Initialise git *here*, not in the parent.

## 3. Deploy (3 min)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. **New app** → **Deploy a public app from a repo**.
3. Repository `<you>/gameweek-desk`, branch `main`, main file path `app.py`.
4. **Deploy**. First build takes 2–3 minutes.

Optionally, **Settings → Secrets**, paste:

```toml
GROQ_API_KEY = "gsk_..."
SERPAPI_KEY  = "..."
```

The app works without them — it runs on recorded verdicts. Adding them switches the agent
to live search. **For tomorrow I'd leave them off**: the recorded path is deterministic,
so the demo shows the same queue you rehearsed, and live search in pre-season returns thin
results for a lot of players.

## 4. Generate the submission PDF with your real links (1 min)

```powershell
python make_submission_pdf.py "https://your-app.streamlit.app" "https://your-video-link"
```

Overwrites `Gameweek_Desk_Submission.pdf` with the links embedded.

## 5. Record the video

Follow `VIDEO_SCRIPT.md`. Before recording: set **Verification scope → Full squad** and
**Free transfers → 2**, run once to warm the cache, reload, and clear the audit log.

---

## If the build fails

**`ModuleNotFoundError`** — check `requirements.txt` was committed. It needs only
streamlit, pandas, numpy, requests. There is deliberately no LightGBM or scikit-learn;
that is why the build is fast.

**App loads but the page is blank** — check the main file path is `app.py`, not
`gameweek-desk/app.py`, if you pointed the repo root at this folder.

**"No transfer clears the threshold"** — not an error. Raise Free transfers to 2 or lower
the fixture horizon. The demo squad is tuned to produce recommendations at
`free_transfers = 2`.
