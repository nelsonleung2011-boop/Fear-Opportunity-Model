# End-user instructions

You only need one thing:

```
https://<username>.github.io/<repository-name>/
```

Open that link in a browser. That is all.

You do **not** need:

- Python
- Command Prompt / Terminal
- A local web server
- Cloudflare
- GitHub
- Browser Developer Tools
- Any installation or configuration

---

## Using the dashboard

1. Open the hosted link.
2. On **Date Explorer**, choose a date (or press **Generate Latest Score**).
3. Press **Generate Score**.
4. Read the 0–100 Opportunity Score and its label:

   | Score | Meaning |
   |---|---|
   | 0–48 | Wait |
   | 49–60 | Neutral |
   | 61–74 | Buy |
   | 75–100 | Strong Buy |

5. Use the tabs to explore:

   - **Compare Dates** — two dates side by side
   - **Backtest Explorer** — historical signal performance
   - **Market Data Audit** — where each number came from
   - **Diagnostics** — provider status and technical logs
   - **Buy Score Trend** — last 12 months of score history and momentum
   - **Manual Score Calculator** — enter your own values
   - **Help & Setup** — plain-language guidance

---

## What the messages mean

### "Current provider: Worker"

Live quotes are being fetched server-side through Cloudflare, then cached for
10 minutes. This is the normal hosted experience.

### "Live market quotes are currently unavailable. The dashboard remains fully functional using the embedded historical dataset dated 2026-11-13. No action is required."

The Worker could not be reached (temporary network issue or the site owner has
not connected it yet). Every feature still works using the built-in dataset.
You do not need to do anything.

### "Chart library unavailable"

The optional chart CDN could not load. Scores, tables, signals and audits still
work; only the graphs are blank. Reconnect to the internet and reload.

---

## If you opened the HTML file from your computer

If the page says it is running directly from your computer, browser security
rules may block live data. Press **Continue with embedded data** — everything
still works offline. For live data, use the hosted link instead.

---

## Privacy and cost

- The dashboard has no login, no tracking, and no user accounts.
- It sends only six symbol requests to the site's own Cloudflare Worker.
- There is no cost to the end user.

For research/education only; not investment advice.
