Parameters Schema:
```json
{
  "meeting_id": {
    "type": "string",
    "description": "The meeting ID (can be either calendar event ID or Google Meet meeting code like 'abc-defg-hij')"
  },
  "meeting_url": {
    "type": "string",
    "description": "The Google Meet URL (e.g., https://meet.google.com/abc-defg-hij) to get transcript for"
  },
  "start_date": {
    "type": "string",
    "description": "Optional: Filter to records starting on or after this date (YYYY-MM-DD format). Useful when multiple recordings exist for the same meeting code."
  },
  "end_date": {
    "type": "string",
    "description": "Optional: Filter to records ending on or before this date (YYYY-MM-DD format). Useful when multiple recordings exist for the same meeting code."
  }
}
```
