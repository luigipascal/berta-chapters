/**
 * Netlify Function: submission-created
 *
 * Automatically triggered when a Netlify Form receives a submission.
 * Sends a confirmation email to the subscriber using Brevo (Sendinblue).
 *
 * Setup (one-time, in Netlify Dashboard > Environment variables):
 *   BREVO_API_KEY        - Brevo API key (https://app.brevo.com/settings/keys/api)
 *   SENDER_EMAIL         - Verified sender email in Brevo
 *   SENDER_NAME          - Sender display name (default: "Berta Chapters")
 *   SITE_URL             - https://chapters.berta.one
 *
 * If BREVO_API_KEY is not set, this function silently skips (form still works).
 */

exports.handler = async function (event) {
  var payload = JSON.parse(event.body).payload;

  if (!payload || payload.form_name !== "newsletter") {
    return { statusCode: 200, body: "Not a newsletter submission" };
  }

  var apiKey = process.env.BREVO_API_KEY;
  if (!apiKey) {
    console.log("BREVO_API_KEY not set, skipping confirmation email");
    return { statusCode: 200, body: "No email service configured" };
  }

  var email = payload.data.email;
  var name = payload.data.name || "Learner";
  var senderEmail = process.env.SENDER_EMAIL || "no-reply@berta.one";
  var senderName = process.env.SENDER_NAME || "Berta Chapters";
  var siteUrl = process.env.SITE_URL || "https://chapters.berta.one";

  var htmlBody = [
    "<div style='font-family: Times New Roman, serif; max-width: 600px; margin: 0 auto;'>",
    "  <h2 style='color: #003366;'>Welcome to Berta Chapters, " + name + "!</h2>",
    "  <p>Thank you for subscribing to updates.</p>",
    "  <p>You'll receive an email when new chapters are published. At most one email per week.</p>",
    "  <hr style='border-top: 2px solid #808080;'>",
    "  <p><strong>Latest chapter:</strong></p>",
    "  <p><a href='" + siteUrl + "/chapters/chapter-08/'>Chapter 8: Unsupervised Learning</a> &mdash; ",
    "  K-Means, hierarchical clustering, DBSCAN, PCA, t-SNE, anomaly detection, and a customer segmentation capstone.</p>",
    "  <p><strong>Start learning now:</strong></p>",
    "  <ul>",
    "    <li><a href='" + siteUrl + "/chapters/'>Browse all 8 chapters</a></li>",
    "    <li><a href='" + siteUrl + "/playground/'>Try the Python Playground</a></li>",
    "    <li><a href='https://github.com/luigipascal/berta-chapters'>Star on GitHub</a></li>",
    "  </ul>",
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

  try {
    var response = await fetch("https://api.brevo.com/v3/smtp/email", {
      method: "POST",
      headers: {
        "api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        sender: { name: senderName, email: senderEmail },
        to: [{ email: email, name: name }],
        subject: "Welcome to Berta Chapters!",
        htmlContent: htmlBody,
      }),
    });

    if (response.ok) {
      console.log("Confirmation email sent to " + email);
    } else {
      var errorText = await response.text();
      console.log("Failed to send email: " + errorText);
    }
  } catch (err) {
    console.log("Email send error: " + err.message);
  }

  return { statusCode: 200, body: "OK" };
};
