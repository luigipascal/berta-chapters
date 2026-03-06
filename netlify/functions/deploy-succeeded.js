/**
 * Netlify Function: deploy-succeeded (Background Event)
 *
 * Automatically triggered after every successful Netlify deploy.
 * Reads chapter_notification.json to determine what to notify about,
 * then checks LAST_NOTIFIED_CHAPTER to avoid re-sending.
 *
 * How to send a newsletter for a new chapter:
 * 1. Update chapter_notification.json with the new chapter details.
 * 2. Commit, push, and deploy. That's it.
 *
 * The function compares the chapter number in the JSON file against
 * LAST_NOTIFIED_CHAPTER (stored as a Netlify env var). If they differ,
 * it fetches all subscribers from Netlify Forms and emails them via
 * Resend, then updates LAST_NOTIFIED_CHAPTER so subsequent deploys
 * won't re-send.
 *
 * One-time setup (Netlify Dashboard > Site settings > Environment variables):
 *   RESEND_API_KEY       - Resend API key
 *   CONFIRM_FROM_EMAIL   - Verified sender email in Resend
 *   SITE_URL             - https://chapters.berta.one
 *   NETLIFY_API_TOKEN    - Personal access token (app.netlify.com/user/applications)
 *   NETLIFY_SITE_ID      - Site ID (Site settings > General)
 *
 * No per-release changes needed in the dashboard.
 */

var config = require("./chapter_notification.json");

exports.handler = async function () {
  var chapterNumber = String(config.chapter_number);
  var chapterTitle = config.chapter_title;
  var chapterDescription = config.chapter_description;

  var lastNotified = process.env.LAST_NOTIFIED_CHAPTER || "";
  if (lastNotified === chapterNumber) {
    console.log("Chapter " + chapterNumber + " already notified — skipping");
    return { statusCode: 200, body: "Already notified for chapter " + chapterNumber };
  }

  var apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    console.log("RESEND_API_KEY not set — cannot send emails");
    return { statusCode: 200, body: "No email service configured" };
  }

  var netlifyToken = process.env.NETLIFY_API_TOKEN;
  var siteId = process.env.NETLIFY_SITE_ID;
  if (!netlifyToken || !siteId) {
    console.log("NETLIFY_API_TOKEN or NETLIFY_SITE_ID not set");
    return { statusCode: 200, body: "Netlify API not configured" };
  }

  var fromEmail = process.env.CONFIRM_FROM_EMAIL || "onboarding@resend.dev";
  var siteUrl = process.env.SITE_URL || "https://chapters.berta.one";

  // ── Fetch subscribers from Netlify Forms ──
  var subscribers = [];
  try {
    var formsRes = await fetch(
      "https://api.netlify.com/api/v1/sites/" + siteId + "/forms",
      { headers: { "Authorization": "Bearer " + netlifyToken } }
    );
    if (!formsRes.ok) {
      console.log("Failed to fetch forms: HTTP " + formsRes.status);
      return { statusCode: 200, body: "Failed to fetch forms" };
    }
    var forms = await formsRes.json();
    var newsletterForm = forms.find(function (f) { return f.name === "newsletter"; });

    if (!newsletterForm) {
      console.log("Newsletter form not found");
      return { statusCode: 200, body: "Newsletter form not found" };
    }

    var page = 1;
    var perPage = 100;
    while (true) {
      var subsRes = await fetch(
        "https://api.netlify.com/api/v1/forms/" + newsletterForm.id +
        "/submissions?per_page=" + perPage + "&page=" + page,
        { headers: { "Authorization": "Bearer " + netlifyToken } }
      );
      if (!subsRes.ok) break;
      var subs = await subsRes.json();
      if (!subs || subs.length === 0) break;
      for (var i = 0; i < subs.length; i++) {
        var email = subs[i].data && subs[i].data.email;
        if (email && subscribers.indexOf(email) === -1) {
          subscribers.push(email);
        }
      }
      if (subs.length < perPage) break;
      page++;
    }
  } catch (err) {
    console.log("Error fetching subscribers: " + err.message);
    return { statusCode: 500, body: "Failed to fetch subscribers" };
  }

  if (subscribers.length === 0) {
    console.log("No subscribers found");
    return { statusCode: 200, body: "No subscribers" };
  }

  console.log("Sending Chapter " + chapterNumber + ": " + chapterTitle +
    " notification to " + subscribers.length + " subscriber(s)");

  // ── Build email ──
  var chapterUrl = siteUrl + "/chapters/chapter-" +
    (parseInt(chapterNumber) < 10 ? "0" : "") + chapterNumber + "/";

  var htmlBody = [
    "<div style='font-family: Times New Roman, serif; max-width: 600px; margin: 0 auto;'>",
    "  <h2 style='color: #003366;'>New Chapter Published!</h2>",
    "  <h3 style='color: #4a90d9;'>Chapter " + chapterNumber + ": " + chapterTitle + "</h3>",
    "  <p>" + chapterDescription + "</p>",
    "  <p style='margin: 20px 0;'>",
    "    <a href='" + chapterUrl + "' style='background: #003366; color: #ffffff; " +
    "padding: 10px 24px; text-decoration: none; font-weight: bold;'>",
    "      Read Chapter " + chapterNumber + " Now</a>",
    "  </p>",
    "  <hr style='border-top: 2px solid #808080;'>",
    "  <p><strong>What's included:</strong></p>",
    "  <ul>",
    "    <li>3 interactive notebooks (Introduction, Intermediate, Advanced)</li>",
    "    <li>5 exercises with complete solutions</li>",
    "    <li>Production-ready toolkit scripts</li>",
    "    <li>3 SVG diagrams</li>",
    "    <li>Synthetic datasets for hands-on practice</li>",
    "  </ul>",
    "  <p><a href='" + siteUrl + "/chapters/'>Browse all chapters</a> | ",
    "  <a href='" + siteUrl + "/playground/'>Try the Playground</a></p>",
    "  <hr style='border-top: 2px solid #808080;'>",
    "  <p style='font-size: 12px; color: #808080;'>",
    "    You're receiving this because you subscribed at " + siteUrl + "<br>",
    "    To unsubscribe, reply to this email with 'unsubscribe'.",
    "  </p>",
    "  <p style='font-size: 12px; color: #808080;'>",
    "    Created by <a href='https://rondanini.net'>Luigi Pascal Rondanini</a> | ",
    "    Powered by <a href='https://berta.one'>Berta AI</a>",
    "  </p>",
    "</div>",
  ].join("\n");

  // ── Send emails ──
  var sent = 0;
  var failed = 0;

  for (var j = 0; j < subscribers.length; j++) {
    try {
      var response = await fetch("https://api.resend.com/emails", {
        method: "POST",
        headers: {
          "Authorization": "Bearer " + apiKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          from: "Berta Chapters <" + fromEmail + ">",
          to: [subscribers[j]],
          subject: "New Chapter: " + chapterTitle + " (Chapter " + chapterNumber + ")",
          html: htmlBody,
        }),
      });

      if (response.ok) {
        sent++;
        console.log("Sent to " + subscribers[j]);
      } else {
        failed++;
        var errorText = await response.text();
        console.log("Failed: " + subscribers[j] + " — " + errorText);
      }
    } catch (err) {
      failed++;
      console.log("Error: " + subscribers[j] + " — " + err.message);
    }
  }

  // ── Update LAST_NOTIFIED_CHAPTER via Netlify API ──
  if (sent > 0) {
    try {
      // Fetch existing env vars for this account (scoped to site)
      var envRes = await fetch(
        "https://api.netlify.com/api/v1/accounts/me/env?site_id=" + siteId,
        { headers: { "Authorization": "Bearer " + netlifyToken } }
      );
      var envVars = await envRes.json();
      var existing = Array.isArray(envVars)
        ? envVars.find(function (v) { return v.key === "LAST_NOTIFIED_CHAPTER"; })
        : null;

      if (existing) {
        // Delete then recreate (Netlify API pattern for updating)
        await fetch(
          "https://api.netlify.com/api/v1/accounts/me/env/LAST_NOTIFIED_CHAPTER?site_id=" + siteId,
          { method: "DELETE", headers: { "Authorization": "Bearer " + netlifyToken } }
        );
      }
      await fetch(
        "https://api.netlify.com/api/v1/accounts/me/env?site_id=" + siteId,
        {
          method: "POST",
          headers: {
            "Authorization": "Bearer " + netlifyToken,
            "Content-Type": "application/json",
          },
          body: JSON.stringify([{
            key: "LAST_NOTIFIED_CHAPTER",
            scopes: ["functions"],
            values: [{ value: chapterNumber, context: "all" }],
          }]),
        }
      );
      console.log("Updated LAST_NOTIFIED_CHAPTER to " + chapterNumber);
    } catch (err) {
      console.log("Could not update LAST_NOTIFIED_CHAPTER: " + err.message);
    }
  }

  var summary = "Chapter " + chapterNumber + " — Sent: " + sent +
    ", Failed: " + failed + ", Total: " + subscribers.length;
  console.log(summary);
  return { statusCode: 200, body: summary };
};
