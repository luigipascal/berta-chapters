# Newsletter Setup Guide

This page is **not linked in the site navigation** — it's a reference for switching
between Netlify Forms and Brevo.

---

## Option 1: Netlify Forms (Current Default)

Works automatically when deployed to Netlify. Zero configuration needed.

- Submissions appear in **Netlify Dashboard > Forms > newsletter**
- Free tier: 100 submissions/month
- You manually send newsletters or set up a Netlify Function to forward to Brevo/email

**To export subscribers**: Netlify Dashboard > Forms > newsletter > Download CSV

**To get notified of new signups**: Netlify Dashboard > Forms > newsletter >
Form notifications > Add email notification

---

## Option 2: Brevo (Sendinblue)

For automated email campaigns and subscriber management.

### Setup Steps

1. Log into [app.brevo.com](https://app.brevo.com)
2. Go to **Contacts > Lists** and create a list called "Berta Chapters Newsletter"
3. Go to **Contacts > Forms** and create a subscription form
4. Copy the form action URL (looks like `https://xxx.sibforms.com/serve/...`)
5. Replace the Netlify form in `docs/newsletter.md` with:

```html
<form action="YOUR_BREVO_FORM_URL" method="POST">
  <div class="newsletter-form">
    <h3>Subscribe to Berta Chapters Updates</h3>
    <p>Get an email when we publish new chapters or release major updates.<br>
    <strong>At most one email per week.</strong></p>
    <p>
      <label for="EMAIL"><strong>Your email address:</strong></label><br>
      <input type="email" name="EMAIL" id="EMAIL" placeholder="your@email.com" required
        style="font-family: 'Courier New', monospace; font-size: 14px; padding: 6px 10px;
        border: 2px inset #808080; width: 100%; box-sizing: border-box; border-radius: 0;">
    </p>
    <p>
      <button type="submit"
        style="font-family: 'Times New Roman', serif; font-size: 14px; font-weight: bold;
        padding: 8px 24px; border: 2px outset #808080; background: #008080; color: #ffffff;
        cursor: pointer; border-radius: 0;">Subscribe</button>
    </p>
  </div>
</form>
```

6. Remove the `data-netlify="true"` and `netlify-honeypot` attributes
7. Set up an automated welcome email in Brevo

### Brevo Advantages
- Automated email campaigns (send newsletters automatically)
- Subscriber management (lists, segments, unsubscribe)
- Email templates and analytics
- Free tier: 300 emails/day

---

## Option 3: Both Together

Use Netlify Forms to collect submissions, then a Netlify Function to forward
new subscribers to Brevo via their API.

Create `netlify/functions/new-subscriber.js`:

```javascript
const fetch = require('node-fetch');

exports.handler = async (event) => {
  const { email, name } = JSON.parse(event.body).payload.data;

  await fetch('https://api.brevo.com/v3/contacts', {
    method: 'POST',
    headers: {
      'api-key': process.env.BREVO_API_KEY,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      email,
      attributes: { FIRSTNAME: name || '' },
      listIds: [YOUR_LIST_ID],
      updateEnabled: true,
    }),
  });

  return { statusCode: 200 };
};
```

Then in Netlify Dashboard > Forms > newsletter > Form notifications >
Add outgoing webhook > URL: `/.netlify/functions/new-subscriber`

This gives you Netlify's simple form + Brevo's email automation.
