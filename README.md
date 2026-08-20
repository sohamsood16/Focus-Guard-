# Focus-Guard-
This is an app that tracks through your webcam and prevents you from getting distracted 
# FocusGuard — Webcam Focus Monitor

A hackathon-ready prototype that uses a laptop webcam to estimate visual attention. It detects whether a face is present, whether the head turns away from the calibrated screen position, and whether the eyes appear to look away/close for too long. A short distraction streak triggers a loud browser alarm.

## Stack

- HTML/CSS/JavaScript
- MediaPipe Tasks Vision Face Landmarker
- Browser WebRTC webcam API
- Web Audio API for the alarm

## Run it

1. Make sure the folder contains `index.html`, `style.css`, and `app.js`.
2. In Terminal, enter this folder.
3. Run:

```bash
python3 -m http.server 8000
```

4. Open `http://localhost:8000` in Chrome, Edge, or Safari.
5. Enter the assigned work, click **Start monitoring**, and allow camera access.
6. Look directly at the screen for about one second so the system can calibrate your normal head position.
7. Turn your head away for a few seconds. The dashboard should show a falling attention score and then trigger the alarm.
8. Use **Test alarm** during the presentation to prove the sound output works.

## Hackathon demo flow

1. Start monitoring and show the live face/attention dashboard.
2. Keep your face centered: score stays high.
3. Turn your head away: score falls and the distraction streak starts.
4. Keep looking away for ~2.5 seconds: the alarm fires.
5. Return to the screen: the system logs `Focus restored` and clears the streak.

## What this prototype does NOT claim

This is visual-attention estimation, not mind-reading and not proof that a student is academically distracted. A face turned away can mean writing notes, reading another physical item, thinking, or talking. For a real deployment, use consent, local processing, clear privacy controls, and human review.

## Easy upgrades for the next hackathon round

- Add browser-tab / active-window monitoring.
- Add an assignment timer and Pomodoro mode.
- Add a local SQLite session report with focus percentage.
- Add a stronger object model to detect phone usage.
- Add teacher/admin mode only with explicit consent.
- Add personalized calibration per user and adaptive thresholds.

