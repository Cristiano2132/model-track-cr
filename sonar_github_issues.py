import os
from datetime import datetime

import requests

# =========================
# CONFIG
# =========================

SONAR_TOKEN = os.getenv("SONAR_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

SONAR_PROJECT_KEY = "Cristiano2132_model-track-cr"
GITHUB_REPO = "Cristiano2132/model-track-cr"


SONAR_URL = "https://sonarcloud.io/api/issues/search"
GITHUB_URL = f"https://api.github.com/repos/{GITHUB_REPO}/issues"

EXPORT_FILE = "sonar_issues.md"

SONAR_URL = "https://sonarcloud.io/api/issues/search"
SONAR_AUTH_URL = "https://sonarcloud.io/api/authentication/validate"

GITHUB_URL = f"https://api.github.com/repos/{GITHUB_REPO}/issues"

sonar_auth = (SONAR_TOKEN, "")
github_headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


# =========================
# CHECK SONAR CONNECTION
# =========================
def check_sonar_connection():
    print("🔐 Checking SonarCloud connection...")

    response = requests.get(SONAR_AUTH_URL, auth=sonar_auth)

    if response.status_code != 200 or not response.json().get("valid"):
        raise Exception("❌ SonarCloud authentication failed")

    print("✅ SonarCloud authentication successful")


# =========================
# FETCH ALL ISSUES
# =========================
def get_all_issues():
    issues = []
    page = 1

    while True:
        params = {
            "projectKeys": SONAR_PROJECT_KEY,
            "ps": 100,
            "p": page,
            "statuses": "OPEN,CONFIRMED,REOPENED",
            "types": "BUG,VULNERABILITY,CODE_SMELL",
        }

        response = requests.get(SONAR_URL, params=params, auth=sonar_auth)
        response.raise_for_status()

        data = response.json()
        issues.extend(data["issues"])

        if page * 100 >= data["total"]:
            break

        page += 1

    return issues


# =========================
# EXPORT TO MARKDOWN
# =========================
def export_issues_to_markdown(issues):
    print(f"📝 Exporting issues to {EXPORT_FILE}...")

    with open(EXPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# SonarCloud Issues Export\n\n")
        f.write(f"Project: `{SONAR_PROJECT_KEY}`\n")
        f.write(f"Generated at: {datetime.utcnow()} UTC\n\n")
        f.write(f"Total issues: **{len(issues)}**\n\n---\n\n")

        for issue in issues:
            f.write(f"## [{issue['severity']}] {issue['message']}\n\n")

            f.write(f"- **Type:** {issue['type']}\n")
            f.write(f"- **Rule:** {issue['rule']}\n")
            f.write(f"- **File:** `{issue['component']}`\n")
            f.write(f"- **Line:** {issue.get('line', 'N/A')}\n")
            f.write(f"- **Effort:** {issue.get('effort', 'N/A')}\n")

            if issue.get("tags"):
                f.write(f"- **Tags:** {', '.join(issue['tags'])}\n")

            f.write(
                f"- **Link:** https://sonarcloud.io/project/issues?id={SONAR_PROJECT_KEY}&open={issue['key']}\n\n"
            )

            f.write("---\n\n")

    print("✅ Export completed")


# =========================
# EXISTING ISSUES
# =========================
def get_existing_titles():
    titles = set()
    page = 1

    while True:
        response = requests.get(
            GITHUB_URL,
            headers=github_headers,
            params={"state": "open", "per_page": 100, "page": page},
        )
        response.raise_for_status()

        data = response.json()
        if not data:
            break

        for issue in data:
            titles.add(issue["title"])

        page += 1

    return titles


# =========================
# BUILD TITLE
# =========================
def build_title(issue):
    return f"[SonarCloud][{issue['severity']}] {issue['message'][:80]}"


# =========================
# BUILD BODY
# =========================
def build_body(issue):
    return f"""
## 🧪 SonarCloud Issue

**Type:** {issue["type"]}  
**Severity:** {issue["severity"]}  
**Rule:** {issue["rule"]}  

---

### 📍 Location
- **File:** `{issue["component"]}`
- **Line:** {issue.get("line", "N/A")}

---

### 📝 Description
{issue["message"]}

---

### 🔗 SonarCloud Link
https://sonarcloud.io/project/issues?id={SONAR_PROJECT_KEY}&open={issue["key"]}

---

### ⚙️ Effort
{issue.get("effort", "N/A")}

---

### 🏷️ Tags
{", ".join(issue.get("tags", []))}

---

_This issue was automatically created from SonarCloud._
"""


# =========================
# LABELS
# =========================
def build_labels(issue):
    return ["sonarcloud", issue["severity"].lower(), issue["type"].lower()]


# =========================
# CREATE ISSUE
# =========================
def create_issue(title, body, labels):
    payload = {"title": title, "body": body, "labels": labels}

    response = requests.post(GITHUB_URL, headers=github_headers, json=payload)

    if response.status_code == 201:
        print(f"✅ Created: {title}")
    else:
        print(f"❌ Failed: {title}")
        print(response.text)


# =========================
# MAIN
# =========================
def main():
    check_sonar_connection()

    print("🔍 Fetching SonarCloud issues...")
    issues = get_all_issues()

    print(f"Total issues fetched: {len(issues)}")

    export_issues_to_markdown(issues)

    # existing_titles = get_existing_titles()

    # for issue in issues:
    #     title = build_title(issue)

    #     if title in existing_titles:
    #         print(f"⏭️ Skipping: {title}")
    #         continue

    #     body = build_body(issue)
    #     labels = build_labels(issue)

    #     create_issue(title, body, labels)


if __name__ == "__main__":
    main()
