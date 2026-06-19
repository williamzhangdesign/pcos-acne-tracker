# PCOS Acne Severity Tracker

CS 8803-O29 Health Sensing and Interventions — Group 19  
**Team:** William Hua Zhang, Rifat Kabir Sharna

A mobile app that uses computer vision to grade acne severity from smartphone images and correlates results with a dietary log, designed for PCOS patients tracking hormonal acne.

## Repo Structure

| Folder | Contents |
|--------|----------|
| `app/` | Android app (Jetpack Compose) |
| `model/` | CV model training, evaluation, and TFLite export |
| `data/` | Dataset scripts and split configs (raw data not committed) |
| `docs/` | Proposal, contract, report drafts, figures |

## Stack

- **Mobile:** Android (Kotlin + Jetpack Compose)
- **CV model:** TFLite (trained on ACNE04 + Roboflow; tested on Google SCIN)
- **Grading scale:** IGA (Investigator's Global Assessment)
