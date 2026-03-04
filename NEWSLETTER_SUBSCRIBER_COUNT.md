# How to See Newsletter Subscriber Count

The Berta Chapters newsletter uses **Netlify Forms**. Every subscription is a form submission. Here’s how to see how many people have subscribed.

---

## Option 1: Netlify Dashboard (easiest)

1. Log in at [app.netlify.com](https://app.netlify.com).
2. Open the site that hosts Berta Chapters (e.g. the site for `berta-chapters` or `chapters.berta.one`).
3. In the left sidebar, go to **Forms**.
4. Click the **newsletter** form.
5. You’ll see the list of submissions. The **total number of submissions** is your subscriber count (one submission = one subscription).

You can also export submissions (e.g. CSV) from that same form page for your records.

**Note:** Netlify may separate “Verified” and “Spam” submissions. Use the toggle above the list if you want to include or exclude spam. Your real subscriber count is the number of **verified** submissions.

---

## Option 2: Netlify API (for automation or display)

If you want to show the count on the site or use it in a script:

- Use the [Netlify API](https://docs.netlify.com/api/get-started/): **List form submissions** for the form named `newsletter`.
- You’ll need a **Netlify API token** (Personal Access Token) from: **User settings → Applications → Personal access tokens**.
- Endpoint (replace `site_id` and use your token):  
  `GET https://api.netlify.com/api/v1/sites/{site_id}/forms/newsletter/submissions`  
  The response length (or the `total_count` if provided) is the number of subscribers.

To display the count on the site you’d typically call this from a serverless function (e.g. Netlify Function) so the API token is not exposed in the browser, then have the front end call your function.

---

## Summary

| Goal | Where to go |
|------|-------------|
| **See how many people subscribed** | Netlify Dashboard → Your site → **Forms** → **newsletter** → count (and list) of submissions |
| **Export emails** | Same form page → use Netlify’s export option |
| **Use count in code** | Netlify API: list submissions for form `newsletter` (with API token) |

Your current roadmap target is **10 newsletter subscribers** to unlock weekly chapter releases.
