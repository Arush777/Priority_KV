"""tool_schema templates: early tool contracts + long filler + final valid call."""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Mapping, Tuple

from prioritybench.schema import Category
from prioritybench.templates.base import TemplateSpec, pad_with_filler_turns

# Lexical banks (held-out style variations; plan §3.2).
SEARCH_QUERIES: tuple[str, ...] = (
    "mixed precision kv cache",
    "flashinfer page table",
    "tool schema drift under compression",
    "agent trace replay harness",
    "instruction supersession edge cases",
)
FILE_PATHS: tuple[str, ...] = (
    "/var/data/traces/session_a.jsonl",
    "/mnt/scratch/prioritybench/cal.jsonl",
    "/home/agent/cache/manifest.yaml",
    "/tmp/eval/run_0042/config.json",
)
DB_NAMES: tuple[str, ...] = ("analytics", "ops_metrics", "agent_state", "kv_lab")
TABLES: tuple[str, ...] = ("sessions", "tool_calls", "page_events", "scores")


def _tools_block(tools: List[Dict[str, Any]]) -> str:
    return (
        "You are an API-calling agent. Available tools (JSON Schema):\n"
        + json.dumps(tools, indent=2)
        + "\nRespond to the FINAL user request with a single JSON tool call of the form "
        '{"name": "<tool>", "arguments": {...}}. No prose.'
    )


def _build_search_docs(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    query = rng.choice(SEARCH_QUERIES)
    limit = rng.choice([5, 10, 20])
    tools = [
        {
            "name": "search_docs",
            "description": "Search internal documentation",
            "parameters": {
                "type": "object",
                "required": ["query", "limit"],
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "enum": [5, 10, 20]},
                    "corpus": {"type": "string", "enum": ["wiki", "code", "tickets"]},
                },
            },
        },
        {
            "name": "list_files",
            "description": "List files under a path",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        },
        {
            "name": "echo_debug",
            "description": "Echo a debug string (decoy)",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
    ]
    corpus = rng.choice(["wiki", "code", "tickets"])
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _tools_block(tools)},
        {
            "role": "user",
            "content": "Confirm you loaded the tool schemas. Do not call tools yet.",
        },
        {
            "role": "assistant",
            "content": "Schemas loaded: search_docs, list_files, echo_debug.",
        },
        {
            "role": "user",
            "content": (
                f"FINAL: Search the {corpus} corpus for '{query}' and return at most "
                f"{limit} hits using search_docs."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "allowed_tool_names": ["search_docs"],
        "path": "arguments",
        "expected_schema": {
            "type": "object",
            "required": ["query", "limit"],
            "properties": {
                "query": {"type": "string", "const": query},
                "limit": {"type": "integer", "const": limit},
                "corpus": {"type": "string", "enum": ["wiki", "code", "tickets"]},
            },
        },
        "required_fields": ["query", "limit"],
    }
    # Soft-require corpus when present in the ask (always in this template).
    scoring["expected_schema"]["required"] = ["query", "limit", "corpus"]
    scoring["expected_schema"]["properties"]["corpus"] = {
        "type": "string",
        "const": corpus,
    }
    return messages, scoring


def _build_read_file(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    path = rng.choice(FILE_PATHS)
    offset = rng.choice([0, 128, 1024])
    tools = [
        {
            "name": "read_file",
            "description": "Read a text file slice",
            "parameters": {
                "type": "object",
                "required": ["path", "offset"],
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "max_bytes": {"type": "integer"},
                },
            },
        },
        {
            "name": "write_file",
            "description": "Write bytes to a path (decoy)",
            "parameters": {
                "type": "object",
                "required": ["path", "content"],
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
            },
        },
        {
            "name": "stat_path",
            "description": "Stat a filesystem path",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        },
        {
            "name": "search_docs",
            "description": "Search docs (decoy)",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {"query": {"type": "string"}},
            },
        },
    ]
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _tools_block(tools)},
        {
            "role": "user",
            "content": f"We will later need to inspect {path}. Remember the path.",
        },
        {
            "role": "assistant",
            "content": f"Noted path {path} for a future read_file call.",
        },
        {
            "role": "user",
            "content": (
                f"FINAL: Call read_file on {path} starting at byte offset {offset}. "
                "Omit max_bytes."
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "allowed_tool_names": ["read_file"],
        "path": "arguments",
        "expected_schema": {
            "type": "object",
            "required": ["path", "offset"],
            "properties": {
                "path": {"type": "string", "const": path},
                "offset": {"type": "integer", "const": offset},
            },
        },
    }
    return messages, scoring


def _build_sql_query(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    db = rng.choice(DB_NAMES)
    table = rng.choice(TABLES)
    limit = rng.choice([1, 5, 50])
    tools = [
        {
            "name": "sql_query",
            "description": "Run a read-only SQL query",
            "parameters": {
                "type": "object",
                "required": ["database", "sql"],
                "properties": {
                    "database": {"type": "string", "enum": list(DB_NAMES)},
                    "sql": {"type": "string"},
                    "timeout_ms": {"type": "integer"},
                },
            },
        },
        {
            "name": "sql_explain",
            "description": "Explain a SQL plan (decoy)",
            "parameters": {
                "type": "object",
                "required": ["database", "sql"],
                "properties": {
                    "database": {"type": "string"},
                    "sql": {"type": "string"},
                },
            },
        },
        {
            "name": "list_tables",
            "description": "List tables in a database",
            "parameters": {
                "type": "object",
                "required": ["database"],
                "properties": {"database": {"type": "string"}},
            },
        },
    ]
    sql = f"SELECT * FROM {table} LIMIT {limit}"
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _tools_block(tools)},
        {
            "role": "user",
            "content": f"Prefer database '{db}' for analytics work this session.",
        },
        {
            "role": "assistant",
            "content": f"Default database set to {db}.",
        },
        {
            "role": "user",
            "content": (
                f"FINAL: Run sql_query on database {db} with exactly: {sql}"
            ),
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "allowed_tool_names": ["sql_query"],
        "path": "arguments",
        "expected_schema": {
            "type": "object",
            "required": ["database", "sql"],
            "properties": {
                "database": {"type": "string", "const": db},
                "sql": {"type": "string", "const": sql},
            },
        },
    }
    return messages, scoring


def _build_http_get(
    rng: random.Random, target_tokens: int
) -> Tuple[List[Dict[str, str]], Mapping[str, Any]]:
    host = rng.choice(["api.internal", "metrics.lab", "gateway.ops"])
    path = rng.choice(["/v1/health", "/v1/kv/stats", "/v1/agent/tools"])
    url = f"https://{host}{path}"
    tools = [
        {
            "name": "http_get",
            "description": "HTTP GET",
            "parameters": {
                "type": "object",
                "required": ["url"],
                "properties": {
                    "url": {"type": "string"},
                    "headers": {"type": "object"},
                },
            },
        },
        {
            "name": "http_post",
            "description": "HTTP POST (decoy)",
            "parameters": {
                "type": "object",
                "required": ["url", "body"],
                "properties": {
                    "url": {"type": "string"},
                    "body": {"type": "object"},
                },
            },
        },
        {
            "name": "dns_lookup",
            "description": "Resolve a hostname",
            "parameters": {
                "type": "object",
                "required": ["hostname"],
                "properties": {"hostname": {"type": "string"}},
            },
        },
        {
            "name": "echo_debug",
            "description": "Debug echo",
            "parameters": {
                "type": "object",
                "required": ["text"],
                "properties": {"text": {"type": "string"}},
            },
        },
        {
            "name": "list_files",
            "description": "List files",
            "parameters": {
                "type": "object",
                "required": ["path"],
                "properties": {"path": {"type": "string"}},
            },
        },
    ]
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": _tools_block(tools)},
        {
            "role": "user",
            "content": f"Service base host for this shift is {host}.",
        },
        {
            "role": "assistant",
            "content": f"Using host {host} for subsequent HTTP calls.",
        },
        {
            "role": "user",
            "content": f"FINAL: Issue http_get to {url} with no custom headers.",
        },
    ]
    messages = pad_with_filler_turns(messages, rng, target_tokens)
    scoring = {
        "allowed_tool_names": ["http_get"],
        "path": "arguments",
        "expected_schema": {
            "type": "object",
            "required": ["url"],
            "properties": {"url": {"type": "string", "const": url}},
        },
    }
    return messages, scoring


TOOL_SCHEMA_TEMPLATES: tuple[TemplateSpec, ...] = (
    TemplateSpec("tool_schema.search_docs.v1", Category.TOOL_SCHEMA, _build_search_docs),
    TemplateSpec("tool_schema.read_file.v1", Category.TOOL_SCHEMA, _build_read_file),
    TemplateSpec("tool_schema.sql_query.v1", Category.TOOL_SCHEMA, _build_sql_query),
    TemplateSpec("tool_schema.http_get.v1", Category.TOOL_SCHEMA, _build_http_get),
)
