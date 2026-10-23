"""Selected OneBot history references without QQ, AstrBot or generation APIs."""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class Image:
    def __init__(self, *, url="", file=""):
        self.url = url
        self.file = file

    async def convert_to_base64(self):
        return base64.b64encode(Path(self.file).read_bytes()).decode()


class At:
    pass


class Reply:
    pass


class Event:
    def __init__(self, chain=(), *, group="group-a", sender="peer-a", response=None):
        self.message_obj = SimpleNamespace(message=list(chain), self_id="bot-a")
        self.bot = SimpleNamespace(call_action=AsyncMock(return_value=response))
        self.group = group
        self.sender = sender

    def get_messages(self):
        return self.message_obj.message

    def get_group_id(self):
        return self.group

    def get_sender_id(self):
        return self.sender

    def get_self_id(self):
        return "bot-a"

    def get_platform_name(self):
        return "aiocqhttp"


@pytest.fixture
def handler(monkeypatch):
    """Load the real handler with isolated, minimal component/download doubles."""
    components = types.ModuleType("astrbot.api.message_components")
    components.At, components.Image, components.Reply = At, Image, Reply
    monkeypatch.setitem(sys.modules, components.__name__, components)

    utils = types.ModuleType("tl.tl_utils")
    utils.AvatarManager = type("AvatarManager", (), {})
    utils.encode_file_to_base64 = lambda path: base64.b64encode(
        Path(path).read_bytes()
    ).decode()

    def is_png(value):
        try:
            return base64.b64decode(value, validate=True).startswith(
                b"\x89PNG\r\n\x1a\n"
            )
        except (ValueError, TypeError):
            return False

    utils.is_valid_base64_image_str = is_png
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    name = "tl._history_reference_image_handler_test"
    source = Path(__file__).resolve().parents[1] / "tl" / "image_handler.py"
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module.ImageHandler()


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "reference.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nselected-reference")
    return path


def response(path, *, group="group-a", **kwargs):
    result = {
        "message_id": 101,
        "group_id": group,
        "message_type": "group",
        "message": [{"type": "image", "data": {"file": str(path)}}],
    }
    result.update(kwargs)
    return result


@pytest.mark.asyncio
async def test_selected_same_group_message_supplies_image_bytes(handler, reference):
    event = Event(response=response(reference))
    images, avatars = await handler.fetch_images_from_event(
        event, reference_message_ids=["101"]
    )

    assert [base64.b64decode(image) for image in images] == [reference.read_bytes()]
    assert avatars == []
    event.bot.call_action.assert_awaited_once_with(action="get_msg", message_id=101)


@pytest.mark.asyncio
async def test_explicit_ids_do_not_mix_current_images(handler, reference, tmp_path):
    other = tmp_path / "other.png"
    other.write_bytes(b"\x89PNG\r\n\x1a\nnot-selected")
    current = Image(file=str(other))
    event = Event([current], response=response(reference))

    images, _ = await handler.fetch_images_from_event(
        event, reference_message_ids=["101"]
    )

    assert [base64.b64decode(image) for image in images] == [reference.read_bytes()]
    assert event.get_messages() == [current]


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [{"group_id": "other-group"}, {"message_id": 102}])
async def test_other_group_or_wrong_message_is_rejected(handler, reference, changes):
    event = Event(response=response(reference, **changes))
    with pytest.raises(ValueError, match="reference_message_unavailable"):
        await handler.fetch_images_from_event(event, reference_message_ids=["101"])


@pytest.mark.asyncio
async def test_private_peer_is_checked(handler, reference):
    good = response(
        reference, group="", message_type="private", sender={"user_id": "peer-a"}
    )
    images, _ = await handler.fetch_images_from_event(
        Event(group="", response=good), reference_message_ids=["101"]
    )
    assert len(images) == 1

    for bad in (
        response(reference),
        dict(good, sender={"user_id": "other-peer"}),
        dict(good, sender={"user_id": "bot-a"}, target_id="other-peer"),
    ):
        with pytest.raises(ValueError, match="reference_message_unavailable"):
            await handler.fetch_images_from_event(
                Event(group="", response=bad), reference_message_ids=["101"]
            )


@pytest.mark.asyncio
async def test_private_bot_message_requires_matching_target(handler, reference):
    data = response(
        reference,
        group="",
        message_type="private",
        sender={"user_id": "bot-a"},
        target_id="peer-a",
    )
    images, _ = await handler.fetch_images_from_event(
        Event(group="", response=data), reference_message_ids=["101"]
    )
    assert len(images) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", ["101", [True], ["not-an-id"], ["1"] * 7])
async def test_invalid_ids_are_rejected_before_lookup(handler, reference, ids):
    event = Event(response=response(reference))
    with pytest.raises(ValueError, match="reference_message_unavailable"):
        await handler.fetch_images_from_event(event, reference_message_ids=ids)
    event.bot.call_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_unavailable_or_non_image_message_is_rejected(handler, reference):
    for data in (
        None,
        response(reference, message=[{"type": "file", "data": {"file": "ref.png"}}]),
    ):
        with pytest.raises(ValueError, match="reference_message_unavailable"):
            await handler.fetch_images_from_event(
                Event(response=data), reference_message_ids=["101"]
            )


@pytest.mark.asyncio
async def test_partial_selection_cannot_drop_missing_reference(handler, reference):
    event = Event()
    event.bot.call_action.side_effect = [
        response(reference),
        response(reference, message_id=102, message=[]),
    ]
    with pytest.raises(ValueError, match="reference_message_unavailable"):
        await handler.fetch_images_from_event(
            event, reference_message_ids=["101", "102"]
        )


@pytest.mark.asyncio
async def test_failed_image_conversion_cannot_become_text_only(handler, reference):
    reference.unlink()
    handler.api_client = SimpleNamespace(
        _normalize_reference_image_input=AsyncMock(return_value=(None, None))
    )
    with pytest.raises(ValueError, match="reference_message_unavailable"):
        await handler.fetch_images_from_event(
            Event(response=response(reference)), reference_message_ids=["101"]
        )


@pytest.mark.asyncio
async def test_cancelled_lookup_propagates(handler):
    event = Event()
    event.bot.call_action.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await handler.fetch_images_from_event(event, reference_message_ids=["101"])


@pytest.mark.asyncio
async def test_current_image_still_works_without_lookup(handler, reference):
    event = Event([Image(file=str(reference))])
    images, _ = await handler.fetch_images_from_event(event)
    assert len(images) == 1
    event.bot.call_action.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,file_value,expected",
    [
        (
            "https://example.com/reference.png",
            None,
            "https://example.com/reference.png",
        ),
        (None, "cached-reference.png", "cached-reference.png"),
    ],
)
async def test_metadata_collection_avoids_sync_file_getter(url, file_value, expected):
    # This utility is self-contained; load its actual function without importing
    # unrelated OpenCV or AstrBot runtime modules into the offline test suite.
    source = Path(__file__).resolve().parents[1] / "tl" / "tl_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "collect_image_sources"
    )
    namespace = {"logger": SimpleNamespace(debug=lambda *args: None)}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    class File:
        def __init__(self):
            self.url = url
            self.file_ = file_value

        @property
        def file(self):
            raise AssertionError("synchronous File.file getter must not run")

    sources = await namespace["collect_image_sources"](Event([File()]))
    assert sources == [expected]
