from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any, Dict, Optional

from embedd_The_project import build_engine, load_dotenv


def get_project_update_response(
    user_request: str,
    prompt_instructions: str = "Keep the existing architecture, coding style, and folder structure.",
    workspace_path: Optional[str] = None,
    project_root: Optional[str] = None,
    top_k: Optional[int] = None,
    force_reindex: bool = False,
    apply_changes: bool = True,
) -> Dict[str, Any]:
    engine = build_engine(workspace_path=workspace_path, project_root=project_root)

    if apply_changes:
        return engine.chat_and_update_project(
            user_message=user_request,
            prompt_instructions=prompt_instructions,
            top_k=top_k,
            force_reindex=force_reindex,
        )

    index_result = engine.build_or_update_index(force=force_reindex)
    chat_result = engine.chat_about_project(
        user_message=user_request,
        system_instruction=prompt_instructions,
        top_k=top_k,
    )
    update_plan = engine.build_update_plan(
        user_request=user_request,
        prompt_instructions=prompt_instructions,
        top_k=top_k,
    )
    return {
        "index_result": index_result,
        "chat_result": chat_result,
        "update_plan": update_plan,
        "apply_result": {
            "summary": "Preview only. No files were changed.",
            "applied_files": [],
            "skipped_files": [],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project update flow with LLM response")
    parser.add_argument("--request", help="User feature/update request")
    parser.add_argument("--workspace-path", dest="workspace_path", help="Absolute path to the current workspace")
    parser.add_argument("--project-root", dest="project_root", help="Absolute path to the target project")
    parser.add_argument(
        "--instructions",
        dest="prompt_instructions",
        default="Keep the existing architecture, coding style, and folder structure.",
        help="Extra instructions for the LLM",
    )
    parser.add_argument("--top-k", dest="top_k", type=int, default=None, help="Retrieved code chunk count")
    parser.add_argument(
        "--force-reindex",
        dest="force_reindex",
        action="store_true",
        help="Rebuild embeddings for all supported files",
    )
    parser.add_argument(
        "--preview-only",
        dest="preview_only",
        action="store_true",
        help="Return chat response and update plan without changing files",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Start a terminal chat session for project Q&A and updates",
    )
    return parser.parse_args()


def print_chat_response(result: Dict[str, Any]) -> None:
    chat_result = result.get("chat_result") or {}
    answer = chat_result.get("answer", "").strip()
    if answer:
        print("\nAssistant:\n")
        print(answer)

    apply_result = result.get("apply_result") or {}
    applied_files = apply_result.get("applied_files") or []
    skipped_files = apply_result.get("skipped_files") or []
    if applied_files:
        print("\nApplied files:")
        for path in applied_files:
            print(f"- {path}")
    if skipped_files:
        print("\nSkipped files:")
        for item in skipped_files:
            print(f"- {item.get('path', '')}: {item.get('reason', '')}")


def interactive_chat_loop(
    workspace_path: Optional[str],
    project_root: Optional[str],
    prompt_instructions: str,
    top_k: Optional[int],
    force_reindex: bool,
) -> None:
    engine = build_engine(workspace_path=workspace_path, project_root=project_root)
    index_result = engine.build_or_update_index(force=force_reindex)
    workspace_info = engine.describe_workspace()

    print("\nProject terminal chat is ready.")
    print(f"Newly embedded files: {index_result.get('embedded_files', 0)}")
    print(f"Total indexed files: {index_result.get('total_indexed_files', 0)}")
    print(f"Total indexed chunks: {index_result.get('total_indexed_chunks', 0)}")
    print(f"Workspace path: {workspace_info.get('workspace_path', '')}")
    print("Vector DB: hidden storage")
    if index_result.get("total_indexed_files", 0) == 0:
        print("Warning: no project files are currently indexed. Check PROJECT_ROOT and supported file extensions.")
    print(
        textwrap.dedent(
            """
            Commands:
            /chat <message>      Ask about the indexed project
            /update <message>    Ask Gemini and apply project changes
            /preview <message>   Ask Gemini and show plan without changing files
            /reindex             Refresh embeddings for the current project
            /exit                Close the terminal chat
            """
        ).strip()
    )

    while True:
        try:
            raw = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSession closed.")
            return

        if not raw:
            continue
        if raw in {"/exit", "exit", "quit"}:
            print("Session closed.")
            return
        if raw == "/reindex":
            try:
                refresh = engine.build_or_update_index(force=True)
                print(json.dumps(refresh, indent=2))
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        if raw.startswith("/update "):
            message = raw[len("/update ") :].strip()
            if not message:
                print("Provide a message after /update.")
                continue
            try:
                result = engine.chat_and_update_project(
                    user_message=message,
                    prompt_instructions=prompt_instructions,
                    top_k=top_k,
                    force_reindex=False,
                )
                print_chat_response(result)
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        if raw.startswith("/preview "):
            message = raw[len("/preview ") :].strip()
            if not message:
                print("Provide a message after /preview.")
                continue
            try:
                result = get_project_update_response(
                    user_request=message,
                    prompt_instructions=prompt_instructions,
                    workspace_path=workspace_path,
                    project_root=project_root,
                    top_k=top_k,
                    force_reindex=False,
                    apply_changes=False,
                )
                print_chat_response(result)
                update_plan = result.get("update_plan") or {}
                if update_plan:
                    print("\nUpdate plan:")
                    print(json.dumps(update_plan, indent=2))
            except Exception as exc:
                print(f"Error: {exc}")
            continue

        if raw.startswith("/chat "):
            message = raw[len("/chat ") :].strip()
        else:
            message = raw

        if not message:
            print("Enter a message to continue.")
            continue

        try:
            result = engine.chat_about_project(
                user_message=message,
                system_instruction=prompt_instructions,
                top_k=top_k,
            )
            print("\nAssistant:\n")
            print(result.get("answer", ""))
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.interactive:
        interactive_chat_loop(
            workspace_path=args.workspace_path,
            project_root=args.project_root,
            prompt_instructions=args.prompt_instructions,
            top_k=args.top_k,
            force_reindex=args.force_reindex,
        )
        return

    if not args.request:
        raise SystemExit("Provide --request, or use --interactive for terminal chat.")

    result = get_project_update_response(
        user_request=args.request,
        prompt_instructions=args.prompt_instructions,
        workspace_path=args.workspace_path,
        project_root=args.project_root,
        top_k=args.top_k,
        force_reindex=args.force_reindex,
        apply_changes=not args.preview_only,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
