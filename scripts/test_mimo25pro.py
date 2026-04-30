#!/usr/bin/env python3
"""Smoke test Xiaomi MiMo V2.5 Pro with an OpenAI-compatible chat API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable

import requests
from dotenv import load_dotenv


DEFAULT_MODEL = "mimo-v2.5-pro"
DEFAULT_BASE_URL = "https://api.xiaomimimo.com/v1"
TOKEN_PLAN_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def default_base_url(api_key: str) -> str:
    if api_key.startswith("tp-"):
        return TOKEN_PLAN_BASE_URL
    return DEFAULT_BASE_URL


def build_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "你是一个简洁、可靠的中文助手。",
            },
            {
                "role": "user",
                "content": args.prompt,
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": args.stream,
    }


def iter_sse_lines(response: requests.Response) -> Iterable[str]:
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        yield line


def print_stream(response: requests.Response) -> None:
    for line in iter_sse_lines(response):
        if line == "[DONE]":
            print()
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f"\n[unparsed] {line}", file=sys.stderr)
            continue

        delta = event.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if reasoning:
            print(reasoning, end="", flush=True)
        if content:
            print(content, end="", flush=True)


def print_non_stream(response: requests.Response) -> None:
    data = response.json()
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    reasoning = message.get("reasoning_content", "")

    if reasoning:
        print("[reasoning]")
        print(reasoning)
        print()

    print("[content]")
    print(content)

    usage = data.get("usage")
    if usage:
        print()
        print("[usage]")
        print(json.dumps(usage, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Xiaomi MiMo V2.5 Pro through an OpenAI-compatible API."
    )
    parser.add_argument(
        "--prompt",
        default="用三句话介绍你适合做旅行规划系统里的哪些任务。",
        help="Prompt sent to the model.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MIMO_MODEL", DEFAULT_MODEL),
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("MIMO_BASE_URL"),
        help=(
            "OpenAI-compatible base URL. Defaults to Xiaomi official URL, "
            "or token-plan URL when the key starts with tp-."
        ),
    )
    parser.add_argument(
        "--api-key-env",
        default="MIMO_API_KEY",
        help="Environment variable that stores the API key.",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--stream", action="store_true", help="Use streaming output.")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        print(
            f"Missing API key. Set {args.api_key_env}, for example:\n"
            f"  export {args.api_key_env}='YOUR_KEY'",
            file=sys.stderr,
        )
        return 2

    base_url = normalize_base_url(args.base_url or default_base_url(api_key))
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=build_payload(args),
            timeout=args.timeout,
            stream=args.stream,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        print(response.text, file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    if args.stream:
        print_stream(response)
    else:
        print_non_stream(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
