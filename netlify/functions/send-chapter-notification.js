/**
 * Netlify Function: send-chapter-notification
 *
 * Sends a "new chapter published" email to all newsletter subscribers.
 * Fetches subscriber emails from Netlify Forms API, then sends each
 * a notification via Resend.
 *
 * Trigger: Call this function manually or via a webhook after publishing
 * a new chapter. Example:
 *   curl -X POST https://YOUR-SITE.netlify.app/.netlify/functions/send-chapter-notification \
 *     -H "Content-Type: application/json" \
 *     -d '{"chapter_number": 8, "chapter_title": "Unsupervised Learning"}'
 *
 * Required environment variables (set in Netlify Dashboard):
 *   RESEND_API_KEY       - Resend API key
 *   CONFIRM_FROM_EMAIL   - Verified sender email in Resend
 *   SITE_URL             - Site base URL (e.g. https://chapters.berta.one)
 *   NETLIFY_API_TOKEN    - Netlify personal access token (for reading form submissions)
 *   NETLIFY_SITE_ID      - Your Netlify site ID
 *   NEWSLETTER_SECRET    - Shared secret to authorize this function call
 */

exports.handler = async function (event) {
  if (event.httpMethod !== "POST") {
    return { statusCode: 405, body: "Method not allowed" };
  }

  var secret = process.env.NEWSLETTER_SECRET;
  var body;
  try {
    body = JSON.parse(event.body || "{}");
  } catch (e) {
    return { statusCode: 400, body: "Invalid JSON body" };
  }

  if (secret && body.secret !== secret) {
    return { statusCode: 403, body: "Unauthorized" };
  }

  var chapterNumber = body.chapter_number || 8;
  var chapterTitle = body.chapter_title || "Unsupervised Learning";
  var chapterDescription = body.chapter_description ||
    "K-Means clustering, hierarchical clustering, DBSCAN, PCA, t-SNE, anomaly detection, and a customer segmentation capstone project.";

  var apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    console.log("RESEND_API_KEY not set");
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

  var subscribers = [];
  try {
    var formsRes = await fetch(
      "https://api.netlify.com/api/v1/sites/" + siteId + "/forms",
      { headers: { "Authorization": "Bearer " + netlifyToken } }
    );
    var forms = await formsRes.json();
    var newsletterForm = forms.find(function (f) { return f.name === "newsletter"; });

    if (!newsletterForm) {
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
    return { statusCode: 200, body: "No subscribers found" };
  }

  console.log("Sending Chapter " + chapterNumber + " notification to " + subscribers.length + " subscribers");

  var chapterUrl = siteUrl + "/chapters/chapter-" + (chapterNumber < 10 ? "0" : "") + chapterNumber + "/";

  var htmlBody = [
    "<div style='font-family: Times New Roman, serif; max-width: 600px; margin: 0 auto;'>",
    "  <h2 style='color: #003366;'>New Chapter Published!</h2>",
    "  <h3 style='color: #4a90d9;'>Chapter " + chapterNumber + ": " + chapterTitle + "</h3>",
    "  <p>" + chapterDescription + "</p>",
    "  <p style='margin: 20px 0;'>",
    "    <a href='" + chapterUrl + "' style='background: #003366; color: #ffffff; padding: 10px 24px; text-decoration: none; font-weight: bold;'>",
    "      Read Chapter " + chapterNumber + "</a>",
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
      } else {
        failed++;
        var errorText = await response.text();
        console.log("Failed to send to " + subscribers[j] + ": " + errorText);
      }
    } catch (err) {
      failed++;
      console.log("Error sending to " + subscribers[j] + ": " + err.message);
    }
  }

  var summary = "Sent: " + sent + ", Failed: " + failed + ", Total subscribers: " + subscribers.length;
  console.log(summary);

  return { statusCode: 200, body: JSON.stringify({ sent: sent, failed: failed, total: subscribers.length }) };
};
