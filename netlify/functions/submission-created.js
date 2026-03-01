/**
 * Netlify Function: submission-created
 *
 * Automatically triggered when a Netlify Form receives a submission.
 * Sends a confirmation email to the subscriber using Resend.
 *
 * Setup:
 * 1. Create a free account at https://resend.com
 * 2. Get your API key from https://resend.com/api-keys
 * 3. Verify your domain at https://resend.com/domains
 *    (or use the free onboarding@resend.dev for testing)
 * 4. In Netlify Dashboard > Site settings > Environment variables, add:
 *    RESEND_API_KEY = your_resend_api_key
 *    CONFIRM_FROM_EMAIL = updates@yourdomain.com (must be verified in Resend)
 *    SITE_URL = https://chapters.berta.one
 *
 * If RESEND_API_KEY is not set, this function silently skips (form still works).
 */

exports.handler = async function (event) {
  var payload = JSON.parse(event.body).payload;

  if (!payload || payload.form_name !== "newsletter") {
    return { statusCode: 200, body: "Not a newsletter submission" };
  }

  var apiKey = process.env.RESEND_API_KEY;
  if (!apiKey) {
    console.log("RESEND_API_KEY not set, skipping confirmation email");
    return { statusCode: 200, body: "No email service configured" };
  }

  var email = payload.data.email;
  var name = payload.data.name || "Learner";
  var fromEmail = process.env.CONFIRM_FROM_EMAIL || "onboarding@resend.dev";
  var siteUrl = process.env.SITE_URL || "https://chapters.berta.one";

  var htmlBody = [
    "<div style='font-family: Times New Roman, serif; max-width: 600px; margin: 0 auto;'>",
    "  <h2 style='color: #003366;'>Welcome to Berta Chapters, " + name + "!</h2>",
    "  <p>Thank you for subscribing to updates.</p>",
    "  <p>You'll receive an email when new chapters are published. At most one email per week.</p>",
    "  <hr style='border-top: 2px solid #808080;'>",
    "  <p><strong>Start learning now:</strong></p>",
    "  <ul>",
    "    <li><a href='" + siteUrl + "/chapters/'>Browse all chapters</a></li>",
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
    var response = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Authorization": "Bearer " + apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        from: "Berta Chapters <" + fromEmail + ">",
        to: [email],
        subject: "Welcome to Berta Chapters!",
        html: htmlBody,
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
