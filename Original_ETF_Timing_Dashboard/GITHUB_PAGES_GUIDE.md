# GitHub Pages configuration guide

Goal: publish the dashboard as a public web page that end users open with a
single link. No Python, terminal, local server, or browser debugging required.

## What to upload

Upload **only `index.html`** to the repository root.

It is a self-contained 1.7 MB file: all 6,758 historical observations, the
scoring engine, all tabs, diagnostics, and the embedded dataset are inside it.
GitHub Pages does not run `worker.js`; that is deployed separately to Cloudflare
(see `CLOUDFLARE_WORKER_GUIDE.md`).

Do **not** upload for publishing:

- `ETF_Timing_Dashboard.html` (duplicate of `index.html`)
- `worker.js` (belongs in Cloudflare, not GitHub Pages)
- `Launch Dashboard.cmd`, `serve_dashboard.ps1` (local Windows only)
- `dashboard_template.html`, `dashboard_data.json`, `build_*.py`, `test_*.js`
- `data_raw/`, `research/`, CSV files, audit HTML files

It is fine to keep those files in the same repository for source control; just
make sure `index.html` is at the root.

---

## Step 1 — create the repository

1. Go to <https://github.com/new>.
2. Repository name: anything, for example `etf-timing-dashboard`.
3. Visibility: **Public** (GitHub Pages on a free account requires public).
4. Do **not** initialise with a README if you already have files; it is harmless either way.
5. Click **Create repository**.

## Step 2 — upload `index.html`

1. On the repository page click **Add file → Upload files**.
2. Drag in `index.html` from:

   ```
   C:\Users\User\WorkBuddy\2026-11-16-20-56-14\ETF_Timing_Dashboard\index.html
   ```

3. Click **Commit changes**.

If you prefer the Web Editor: create a new file named exactly `index.html`,
paste the contents of the local file, then commit.

## Step 3 — enable Pages

1. Repository **Settings**.
2. Sidebar: **Code, planning, and automation → Pages**.
3. **Build and deployment → Source** = **Deploy from a branch**.
4. Branch = `main` (or `master`, whichever contains `index.html`).
5. Folder = **/(root)**.
6. Click **Save**.

After 1–2 minutes, the top of the Pages settings page shows:

```
Your site is live at https://<username>.github.io/<repository-name>/
```

That is the final link. It also appears under the repository's
**Environments → github-pages** or **Deployments** section.

### URL rules

| Repository name | Public URL |
|---|---|
| `etf-timing-dashboard` | `https://<username>.github.io/etf-timing-dashboard/` |
| `<username>.github.io` | `https://<username>.github.io/` |

## Step 4 — connect live data (site owner only)

1. Deploy `worker.js` to Cloudflare (see the Worker guide).
2. Edit `index.html` on GitHub and set:

   ```js
   workerBase: 'https://etf-market-data.your-subdomain.workers.dev',
   ```

3. Commit. Pages rebuilds automatically.
4. Open **Diagnostics → Data Provider Status** and confirm Worker = **PASS**.

End users never do any of this. They only open the final link.

---

## Updating the hosted site

Whenever the dashboard is rebuilt locally:

1. Run `python build_dashboard.py` (maintainer only) — it refreshes both HTML files.
2. If you set `workerBase` by editing `index.html` on GitHub, re-apply that
   single-line change after uploading the new file.
3. Commit. The public URL stays the same.

Tip: set `DEPLOY_CONFIG.workerBase` in `dashboard_template.html` before building
so every rebuild keeps the Worker URL automatically.

---

## If the link does not appear

| Check | Fix |
|---|---|
| File not committed | Press **Commit changes** after uploading |
| File in a subfolder | Move `index.html` to the repository root |
| Wrong filename | Must be exactly `index.html` |
| Private repository on free plan | Make the repository **Public** |
| Source still `None` | Re-select branch + `/(root)`, then **Save** |
| Wrong branch selected | Choose the branch that contains `index.html` |
| Build failed | Open the **Actions** tab and inspect the red workflow run |

## Verification checklist

- [ ] `https://<username>.github.io/<repo>/` loads the dashboard
- [ ] **Help & Setup** tab is present
- [ ] **Diagnostics → Data Provider Status** shows Worker PASS (after Worker setup)
- [ ] **Test Live Market Data** shows six green ticks with Source = Worker
- [ ] **Generate Latest Score** shows "Current provider: Worker" and 100% confidence
- [ ] If the Worker is down, the message says live quotes are unavailable and embedded data is used — no action required
