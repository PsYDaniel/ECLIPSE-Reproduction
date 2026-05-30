# Publishing this project to GitHub (step by step)

This repository is ready to upload. Follow the three parts below: **fill in your details**, **push to
GitHub**, then **turn on the website (GitHub Pages)**. It takes about 10 minutes.

---

## Part 1 — Fill in your details

Search the project for these placeholders and replace them. (In most editors: *Edit → Find in
files*.)

| Placeholder | Appears in | Replace with |
|-------------|------------|--------------|
| `[YOUR NAME]` | `README.md`, `docs/documentation.md`, `LICENSE`, `index.html` | your name |
| `[COURSE NAME / CODE]` | `README.md`, `docs/documentation.md`, `index.html` | your course |
| `[INSTRUCTOR]` | `README.md` | your instructor's name |
| `USERNAME` | `index.html` (the `REPO_URL` line), `README.md` site link | your GitHub username |
| `VIDEO_LINK_HERE` / `YOUTUBE_URL` | `README.md`, `index.html` | your video URL |

Two of these are the most important, because the website's links depend on them. Open
**`index.html`**, scroll to the `<script>` at the very bottom, and edit:

```js
const REPO_URL    = "https://github.com/USERNAME/ECLIPSE-reproduction"; // ← your repo
const YOUTUBE_URL = "";  // ← paste your video link here, e.g. "https://youtu.be/XXXX"
```

> **The Hebrew `takeaways.pdf`** has your name as `[שם הסטודנט]` baked into the PDF. To change it,
> ask me to regenerate the PDF with your name, or re-export it yourself from the source.

## Part 2 — Push to GitHub

**Option A — no command line (easiest).**
1. Go to <https://github.com/new>, name the repo **`ECLIPSE-reproduction`**, keep it Public, and
   click *Create repository*.
2. On the new repo page, click *uploading an existing file*.
3. Drag in **everything inside this folder** (including the `docs/`, `plan/`, `notebook/`,
   `results/` folders and the hidden `.nojekyll` file). Commit.

**Option B — git command line.**
```bash
cd ECLIPSE-reproduction
git init
git add -A
git commit -m "ECLIPSE reproduction — project submission"
git branch -M main
git remote add origin https://github.com/USERNAME/ECLIPSE-reproduction.git
git push -u origin main
```

> Make sure the hidden **`.nojekyll`** file is included — it tells GitHub Pages to serve the site
> as-is. (`git add -A` includes it automatically; the web uploader needs you to drag it in.)

## Part 3 — Turn on the website (GitHub Pages)

1. In your repo, open **Settings → Pages**.
2. Under *Build and deployment → Source*, choose **Deploy from a branch**.
3. Set the branch to **`main`** and the folder to **`/ (root)`**. Save.
4. Wait ~1 minute, then refresh. Your site is live at:

   ```
   https://USERNAME.github.io/ECLIPSE-reproduction/
   ```

5. Put that URL back into `README.md` (the "Project website" line) so the repo and the site point at
   each other.

## Final checklist before you submit

- [ ] All placeholders replaced (table in Part 1).
- [ ] `REPO_URL` and `YOUTUBE_URL` set in `index.html`.
- [ ] Video recorded, uploaded, and linked in `README.md` **and** `index.html`.
- [ ] GitHub Pages is live and the deliverable cards open the right files.
- [ ] `takeaways.pdf` shows your name.
- [ ] You can open the site and click through all eight deliverables.

That's everything the course spec asks for. Good luck with the defense!
