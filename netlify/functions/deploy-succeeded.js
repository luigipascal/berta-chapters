/**
 * Netlify Function: deploy-succeeded (Background Event)
 *
 * Automatically triggered after every successful Netlify deploy.
 * Sends a "new chapter published" email to all newsletter subscribers
 * when the NOTIFY_CHAPTER environment variable is set.
 *
 * How to use:
 * 1. Set these environment variables in Netlify Dashboard > Site settings > Environment:
 *    RESEND_API_KEY       - Your Resend API key (https://resend.com/api-keys)
 *    CONFIRM_FROM_EMAIL   - Verified sender email in Resend
 *    SITE_URL             - https://chapters.berta.one (or your site URL)
 *    NETLIFY_API_TOKEN    - Personal access token from https://app.netlify.com/user/applications
 *    NETLIFY_SITE_ID      - Your site ID (Site settings > General > Site ID)
 *    NOTIFY_CHAPTER       - Chapter number to notify about (e.g., "8")
 *    NOTIFY_TITLE         - Chapter title (e.g., "Unsupervised Learning")
 *    NOTIFY_DESCRIPTION   - Brief description for the email body (optional)
 *
 * 2. Deploy the site (merge the branch or trigger a deploy).
 *    The function will automatically send the notification.
 *
 * 3. After the emails are sent, REMOVE the NOTIFY_CHAPTER variable
 *    to prevent re-sending on subsequent deploys.
 *
 * If NOTIFY_CHAPTER is not set, this function does nothing.
 */

exports.handler = async function (event) {
  var chapterNumber = process.env.NOTIFY_CHAPTER;
  if (!chapterNumber) {
    console.log("NOTIFY_CHAPTER not set — skipping newsletter notification");
    return { statusCode: 200, body: "No notification configured" };
  }

  var chapterTitle = process.env.NOTIFY_TITLE || "Unsupervised Learning";
  var chapterDescription = process.env.NOTIFY_DESCRIPTION ||
    "A brand new chapter is available with interactive notebooks, exercises with solutions, toolkit scripts, SVG diagrams, and datasets.";

  var apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    console.log("RESEND_API_KEY not set — cannot send notification emails");
    return { statusCode: 200, body: "No email service configured" };
  }

  var netlifyToken = process.env.NETLIFY_API_TOKEN;
  var siteId = process.env.NETLIFY_SITE_ID;
  if (!netlifyToken || !siteId) {
    console.log("NETLIFY_API_TOKEN or NETLIFY_SITE_ID not set — cannot fetch subscribers");
    return { statusCode: 200, body: "Netlify API not configured — set NETLIFY_API_TOKEN and NETLIFY_SITE_ID" };
  }

  var fromEmail = process.env.CONFIRM_FROM_EMAIL || "onboarding@resend.dev";
  var siteUrl = process.env.SITE_URL || "https://chapters.berta.one";

  // Fetch all newsletter subscribers from Netlify Forms API
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
      console.log("Newsletter form not found in Netlify Forms");
      return { statusCode: 200, body: "Newsletter form not found" };
    }

    console.log("Found newsletter form: " + newsletterForm.id + " (" + newsletterForm.submission_count + " submissions)");

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
    console.log("No subscribers found — nothing to send");
    return { statusCode: 200, body: "No subscribers found" };
  }

  console.log("Sending Chapter " + chapterNumber + ": " + chapterTitle + " notification to " + subscribers.length + " subscriber(s)");

  var chapterUrl = siteUrl + "/chapters/chapter-" + (parseInt(chapterNumber) < 10 ? "0" : "") + chapterNumber + "/";

  var htmlBody = [
    "<div style='font-family: Times New Roman, serif; max-width: 600px; margin: 0 auto;'>",
    "  <h2 style='color: #003366;'>New Chapter Published!</h2>",
    "  <h3 style='color: #4a90d9;'>Chapter " + chapterNumber + ": " + chapterTitle + "</h3>",
    "  <p>" + chapterDescription + "</p>",
    "  <p style='margin: 20px 0;'>",
    "    <a href='" + chapterUrl + "' style='background: #003366; color: #ffffff; padding: 10px 24px; text-decoration: none; font-weight: bold;'>",
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
        console.log("Failed to send to " + subscribers[j] + ": " + errorText);
      }
    } catch (err) {
      failed++;
      console.log("Error sending to " + subscribers[j] + ": " + err.message);
    }
  }

  var summary = "Chapter " + chapterNumber + " notification — Sent: " + sent + ", Failed: " + failed + ", Total: " + subscribers.length;
  console.log(summary);
  console.log("IMPORTANT: Remove NOTIFY_CHAPTER from environment variables to prevent re-sending on the next deploy.");

  return { statusCode: 200, body: summary };
};
