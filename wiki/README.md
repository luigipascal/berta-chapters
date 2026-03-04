# Wiki content for GitHub

This folder holds the **source content** for the [Berta Chapters GitHub Wiki](https://github.com/luigipascal/berta-chapters/wiki).

## How to publish to the GitHub wiki

**Option A — Edit in the browser**

1. Open https://github.com/luigipascal/berta-chapters/wiki
2. Click **Create the first page** (or edit **Home** if it already exists)
3. Copy the contents of `Home.md` into the editor and save

**Option B — Clone the wiki repo and push**

```bash
git clone https://github.com/luigipascal/berta-chapters.wiki.git
cd berta-chapters.wiki
# Copy Home.md from this wiki/ folder into the clone, then:
git add .
git commit -m "Update wiki home page"
git push origin main
```

Only the default branch of the wiki repo is shown on GitHub.
