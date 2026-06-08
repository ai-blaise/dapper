# Record Structure

This document describes the JSONL record format used by D.A.P.P.E.R.

## Overview

Each line in a JSONL file represents a single conversation record containing messages, tool definitions, and metadata.

## Record Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `uuid` | string | Yes | Unique identifier for the record |
| `messages` | array | Yes | List of conversation messages |
| `tools` | array | No | List of tool/function definitions |
| `license` | string | Yes | License type (e.g., "cc-by-4.0") |
| `used_in` | array | Yes | List of datasets/models using this record |
| `reasoning` | string | No | Reasoning mode flag (value: "on") |

## Example Record

```json
{
  "uuid": "abc123-def456-ghi789",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What's the weather like?"
    },
    {
      "role": "assistant",
      "content": "I'll check the weather for you.",
      "tool_calls": [
        {
          "id": "call_123",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"location\": \"New York\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_123",
      "content": "{\"temp\": 72, \"condition\": \"sunny\"}"
    },
    {
      "role": "assistant",
      "content": "The weather in New York is 72°F and sunny."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "license": "cc-by-4.0",
  "used_in": ["dataset_v1", "model_training"],
  "reasoning": "on"
}
```

## Message Types

### System Message

Sets the context and behavior for the assistant.

```json
{
  "role": "system",
  "content": "You are a helpful assistant specialized in..."
}
```

### User Message

Input from the user/human.

```json
{
  "role": "user",
  "content": "How do I..."
}
```

### Assistant Message

Response from the AI assistant. May include tool calls.

```json
{
  "role": "assistant",
  "content": "Here's how you can...",
  "tool_calls": [...]
}
```

#### Tool Calls

When the assistant invokes tools:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "tool_name",
        "arguments": "{\"param\": \"value\"}"
      }
    }
  ]
}
```

### Tool Message

Result returned from a tool invocation.

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"result\": \"data\"}"
}
```

## Tool Definitions

Tools define functions available to the assistant.

```json
{
  "type": "function",
  "function": {
    "name": "function_name",
    "description": "What this function does",
    "parameters": {
      "type": "object",
      "properties": {
        "param1": {
          "type": "string",
          "description": "Parameter description"
        },
        "param2": {
          "type": "integer",
          "description": "Another parameter"
        }
      },
      "required": ["param1"]
    }
  }
}
```

## Metadata Fields

### License

Specifies the license under which the record is distributed:

- `cc-by-4.0` - Creative Commons Attribution 4.0
- Other licenses as specified

### Used In

Tracks which datasets or models incorporate this record:

```json
"used_in": ["nano_v3", "training_set_2024"]
```

### Reasoning

Optional flag indicating reasoning mode:

```json
"reasoning": "on"
```

When present, indicates the conversation uses extended reasoning capabilities.

## Message Role Distribution

Typical records contain:

- 1 system message (context setting)
- Multiple user messages (queries/inputs)
- Multiple assistant messages (responses)
- Tool messages (when tools are invoked)

Use `stats -v` to analyze role distribution in your dataset:

```bash
dapper stats dataset/file.jsonl -v
```
