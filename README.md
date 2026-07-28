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
  
## Automatic dietary trigger detection

The Streamlit prototype includes a transparent keyword-based dietary
classification module. When a user enters a meal description, the module
suggests whether the entry may contain:

- high-glycemic foods
- dairy
- refined sugar

The matched keywords are shown to the user, and every suggestion can be
corrected before saving. This feature is intended for prototype usability
testing and self-reflection. It does not provide nutritional or medical
diagnosis.

A future version will replace or supplement the keyword lists with a
validated nutrition database or external nutrition API.
