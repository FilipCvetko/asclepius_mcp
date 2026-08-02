#!/usr/bin/env python3
"""Send an alert email. Body on stdin, subject as argv[1].

Deliberately standalone rather than reusing refresh.py's _email(), which is
coupled to the refresh report dict. Reads the same three credentials that are
already in deploy/.env, so there is nothing extra to configure.

    echo "body" | ./notify.py "[Asclepius MCP] something happened"
"""
import os
import smtplib
import sys
from email.message import EmailMessage


def main() -> int:
    user, pw = os.environ.get("SMTP_USER"), os.environ.get("SMTP_PASS")
    to = os.environ.get("NOTIFY_EMAIL")
    if not (user and pw and to):
        print("notify: SMTP not configured — skipping", file=sys.stderr)
        return 0

    msg = EmailMessage()
    msg["Subject"] = sys.argv[1] if len(sys.argv) > 1 else "Asclepius MCP alert"
    msg["From"], msg["To"] = user, to
    msg.set_content(sys.stdin.read())

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
    except Exception as e:
        print(f"notify: send failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
