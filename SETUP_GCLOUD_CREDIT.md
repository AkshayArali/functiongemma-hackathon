# Using Your $5 Google Cloud Credit for Gemini

Your **$5 Google Cloud credit** is used by calling Gemini through **Vertex AI** (not the free Gemini API key). The code in `main.py` uses Vertex AI when it sees a GCP project in your environment.

## 1. Create / select a GCP project and apply the $5 credit

- Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project (or pick one).
- Make sure **billing** is enabled and your **$5 credit** is applied to this project.
- Note your **Project ID** (e.g. `my-project-123`), not the project name.

## 2. Enable the Vertex AI API

- In Cloud Console: **APIs & Services** → **Enable APIs** → search for **Vertex AI API** → Enable.

## 3. Log in with gcloud (so Vertex AI can use your account)

In a terminal:

```bash
gcloud auth application-default login
```

Sign in with the Google account that has the $5 credit. This sets “Application Default Credentials” so the script can call Vertex AI without an API key.

## 4. Set your project in `.env`

In `functiongemma-hackathon/.env` add (use your real Project ID):

```env
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

Example:

```env
GOOGLE_CLOUD_PROJECT=my-project-123
GOOGLE_CLOUD_LOCATION=us-central1
```

You can remove or leave `GEMINI_API_KEY` / `GOOGLE_API_KEY`; when `GOOGLE_CLOUD_PROJECT` is set, the script uses **Vertex AI** and your **GCloud billing** (including the $5 credit).

## 5. Run the benchmark

From the hackathon folder:

```bash
python benchmark.py
```

Cloud fallback will now go through Vertex AI and use your $5 credit.

---

**Summary:**  
- **Gemini API key** (from AI Studio) = separate free tier, often 429 if quota is 0.  
- **Vertex AI** (GCP project + `gcloud auth application-default login` + `GOOGLE_CLOUD_PROJECT` in `.env`) = uses your **$5 GCloud credit**.
