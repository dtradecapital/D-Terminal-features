"""
server.py - Flask Web Server + APScheduler for Automated Reports
=================================================================
Routes:
    /          -> Health-check page
    /run-now   -> Manually trigger the weekly report pipeline

Scheduler:
    Runs the report automatically every Monday at 09:00 AM.
"""

from flask import Flask
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import atexit

from main import run

# -- Flask app --
app = Flask(__name__)


# -- Scheduled job wrapper --
def weekly_job():
    print(f"\n[SCHEDULER] Running weekly job at: {datetime.now()}")
    run()


# -- APScheduler: every Monday at 9:00 AM --
scheduler = BackgroundScheduler()
scheduler.add_job(
    func=weekly_job,
    trigger="cron",
    day_of_week="mon",
    hour=9,
    minute=0,
)
scheduler.start()
print("[OK] Scheduler started - next run: Monday 09:00 AM")

# Graceful shutdown
atexit.register(lambda: scheduler.shutdown())


# -- Routes --
@app.route("/")
def home():
    return "Server is running"


@app.route("/run-now")
def run_now():
    weekly_job()
    return "Email sent!"


# -- Entry point --
if __name__ == "__main__":
    app.run(port=8000, debug=True)
