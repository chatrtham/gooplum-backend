Parameters Schema:
```json
{
  "channel": {
    "type": "string",
    "description": "Slack channel ID"
  },
  "limit": {
    "type": "integer",
    "description": "Maximum number of messages to return (default: 20)"
  },
  "start_timestamp": {
    "type": "string",
    "description": "Start of time range: timestamp in epoch seconds (e.g. 1234567890.123456)"
  },
  "end_timestamp": {
    "type": "string",
    "description": "End of time range: timestamp in epoch seconds (e.g. 1234567890.123456)"
  },
  "as_user": {
    "type": "boolean",
    "description": "Whether to use user token instead of bot token (default: true)"
  }
}
```
