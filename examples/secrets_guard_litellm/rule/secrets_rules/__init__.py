"""A rule for a sentinel that runs beside a proxy: credentials stay out of the agent's hands.

A coding agent that reads a repository can come across a credential, pass one to the model provider, or hand one back to its user. The rule watches for credentials at all four stages:

- `BeforeGenerate`: a request whose messages hold a credential ends the run, before the credential is sent to the model provider.
- `AfterGenerate`: a reply whose text, reasoning or tool call names hold a credential ends the run, before the client is handed the reply. A call's arguments are left to `BeforeToolCall`.
- `BeforeToolCall`: a call that would read a credential file is changed or rejected before it runs. A read of a `.env` file, such as `.env` or `.env.production`, is changed to a read of the `.env.example` beside it, which lists the same settings with safe values. A `.env` file can hold credentials no format gives away, such as a database password, so it is never read. A read of an SSH key, a `.pem` or `.key` file, or a credentials file such as `.aws/credentials` is rejected, and the agent is told what it can read instead. A call whose arguments hold a credential is rejected too.
- `AfterToolCall`: a tool result that holds a credential ends the run, before the model reads it.

A credential is anything that matches one of `CREDENTIALS`: the formats of AWS access key IDs, GitHub tokens and private keys, which a real secret scanner looks for too. `BeforeGenerate` checks the text of the user, system and assistant messages of a request. Assistant messages are checked because not all of them were judged when the model wrote them: a task's few-shot examples, or a conversation the sidecar joined midway. Tool results and the rejections the sidecar adds were judged at the stages they came from. An assistant message's text leaves out its tool calls, and their arguments aren't checked here: a call the rule rejected for passing a credential is put back in later requests as the model wrote it, so checking calls would end the run on the next request. So the arguments of a call in a history the sidecar never judged, such as a few-shot example's, reach the model provider unchecked.

The explanations say which kind of credential was found and where, never the credential itself, since they are recorded wherever the sentinel's decisions are.
"""

import dataclasses
import re
from collections.abc import Iterable
from pathlib import PurePosixPath

from inspect_sentinel import (
    AfterGenerate,
    AfterToolCall,
    BeforeGenerate,
    BeforeToolCall,
    Context,
    Decision,
    Escalations,
    Protocol,
    Step,
    call_text,
    protocol,
    result_text,
)

# The kinds of credential the rule looks for, by the format each comes in.
CREDENTIALS = {
    "an AWS access key ID": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "a GitHub token": re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})\b"
    ),
    "a private key": re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY(?: BLOCK)?-----"),
}

# Files that hold credentials, by name. Anything under `.ssh/` but a public
# key does too, and so does any `.pem` or `.key` file: private keys come in
# both, though so do public certificates.
CREDENTIAL_FILES = {
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    ".git-credentials",
    ".netrc",
    "application_default_credentials.json",
}

# The tools that read a file, and the argument that names it.
READ_TOOLS = {"read_file": "path"}

# What the agent is told when a call is rejected. It goes back to the agent as
# the call's result.
CREDENTIAL_FILE = "That file holds credentials, which can't be read here. Read a file that doesn't instead, yourself, and go on: for a key, its public half (the same path ending in `.pub`); for settings, the project's example configuration."
CREDENTIAL_ARGUMENT = "That call passes a credential, which can't be sent anywhere from here. Leave the credential out."


@protocol
def secrets_guard() -> Protocol:
    """Keep credentials out of everything that passes through the proxy, at every stage."""

    async def decide(
        context: Context, step: Step, escalations: Escalations
    ) -> Decision | None:
        match step:
            case BeforeGenerate():
                found = _credential(
                    message.text
                    for message in step.input
                    if message.role in ("system", "user", "assistant")
                )
                if found:
                    return Decision.terminate(f"the request holds {found}")
            case AfterGenerate():
                found = _credential(_reply_text(step))
                if found:
                    return Decision.terminate(f"the model's reply holds {found}")
            case BeforeToolCall():
                return _before_tool_call(step)
            case AfterToolCall():
                found = _credential([result_text(step)])
                if found:
                    return Decision.terminate(
                        f"the result of {step.call.function} holds {found}"
                    )
        return Decision.proceed()

    return decide


def _before_tool_call(step: BeforeToolCall) -> Decision:
    call = step.call
    # arguments that didn't parse are only in the parse error
    found = _credential([call_text(call), call.parse_error or ""])
    if found:
        return Decision.reject(
            f"the call to {call.function} passes {found}", message=CREDENTIAL_ARGUMENT
        )
    argument = READ_TOOLS.get(call.function)
    path = call.arguments.get(argument) if argument else None
    if not isinstance(path, str):
        return Decision.proceed()
    file = PurePosixPath(path)
    if file.name == ".env" or (
        file.name.startswith(".env.") and file.name != ".env.example"
    ):
        safe = str(file.with_name(".env.example"))
        return Decision(
            action="modify",
            modified=dataclasses.replace(
                call, arguments={**call.arguments, argument: safe}
            ),
            explanation=f"{call.function} of a .env file, changed to its .env.example",
        )
    if _is_credential_file(file):
        return Decision.reject(
            f"{call.function} of a credential file", message=CREDENTIAL_FILE
        )
    return Decision.proceed()


def _is_credential_file(file: PurePosixPath) -> bool:
    if file.suffix == ".pub":
        return False
    return (
        ".ssh" in file.parts
        or file.name in CREDENTIAL_FILES
        or file.suffix in (".pem", ".key")
        or file.parts[-2:] == (".aws", "credentials")
    )


def _credential(texts: Iterable[str]) -> str:
    text = "\n".join(texts)
    return ", ".join(
        kind for kind, pattern in CREDENTIALS.items() if pattern.search(text)
    )


def _reply_text(step: AfterGenerate) -> Iterable[str]:
    # a call's arguments are left to `BeforeToolCall`, which can change or
    # reject one and let the agent go on. Its name is checked here, since no
    # stage after this one does, and a rejection would quote it.
    if not step.output.choices:
        return
    message = step.output.message
    yield message.text
    for call in message.tool_calls or []:
        yield call.function
    if not isinstance(message.content, str):
        for content in message.content:
            if content.type == "reasoning":
                yield content.reasoning
                if content.summary is not None:
                    yield content.summary
