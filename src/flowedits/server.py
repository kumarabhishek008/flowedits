"""MCP server: edit videos and photos frame by frame, with the mechanical work done here.

The AI agent only does the creative part (deciding the edit, generating a replacement image,
hand-editing a few crops or frames). Decoding, finding the target in every frame, applying the
edit to every frame and encoding the result with the original audio all happen in this server.
"""

from __future__ import annotations

import asyncio
import functools
import json
from typing import Annotated, Literal

from mcp.server.mcpserver import Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

from . import operations as ops
from .jobs import JOBS, Job
from .media import FFmpegError
from .paths import find_media as find_files
from .paths import resolve_media_path
from .workspace import ProjectError, delete_project, list_projects, workspace_root

INSTRUCTIONS = f"""\
flowedits edits VIDEOS and PHOTOS. You never handle frames yourself: open the media, describe the
edit, render. Pick tools by what the user asks:

| User says / wants                                         | Do this                                              |
|-----------------------------------------------------------|------------------------------------------------------|
| gives a video (any path form, or just a file name)        | open_video(video_path, target_image_path if given)   |
| gives a photo or a folder of photos                       | open_photos(photo_path, target_image_path if given)  |
| "replace/swap this logo/image/poster with X"              | open_* with target -> make ONE image of X -> apply_to_target(operation="replace", replacement_path=...) |
| "remove/erase/delete this logo/watermark/object"          | open_* with target -> apply_to_target(operation="remove") |
| "hide/blur/censor/pixelate it"                            | apply_to_target(operation="blur" or "pixelate")      |
| "cover it with a color"                                   | apply_to_target(operation="fill", fill_color="#hex") |
| "show different images over time on it"                   | save images in a folder -> apply_to_target(replace, replacement_path=<folder>) |
| "only between 0:12 and 0:18" / "first 5 seconds"          | open_video(..., edit_start_time, edit_end_time)      |
| a different change for specific frames                    | export_target_regions -> edit crops -> import_target_regions |
| edit whole frames by hand                                 | export_frames -> edit files -> update_frame          |
| "undo" / "go back"                                        | reset_frames                                         |
| "show me" / check the result                              | contact_sheet(only_edited=true) or view_frame        |
| finished editing a video                                  | render_video(project_id)                             |
| finished editing photos                                   | save_photos(project_id)                              |
| can't find the file                                       | find_media(query)                                    |

Defaults and rules:
- Pass paths exactly as the user gave them; quotes, escaped spaces, file:// and bare names are resolved.
- The target image is the thing to change; it should be a tight crop of how it looks in the media.
  If no target is found, retry locate_target with match_threshold 0.55 or method="features".
- apply_to_target is instant (the edit is recorded); the pixels are produced by render_video /
  save_photos. Always finish with one of them and give the user the output path.
- Results are saved next to the original as <name>_edited.<ext>. Only use replace_original=true when
  the user explicitly asks to overwrite the original (a <name>.original.<ext> backup is kept).
- Long videos (up to tens of thousands of frames, e.g. 10+ min at 30-60 fps) are supported. Tools that
  take long (open_video/open_photos with a target, locate_target, export_*, render_video, save_photos)
  return the result directly when done within ~40 s, otherwise a job_id: then call job_status(job_id)
  repeatedly (each call waits up to ~40 s) until status is "done", and tell the user the progress.
- trim_start_time/trim_end_time CUT the output video; edit_start_time/edit_end_time only limit WHERE
  edits happen (the full video is kept). Times: 12, 12.5, "12s", "0:12", "00:00:12.5".
- Frame indices accept ints, lists or ranges like "0-40,55". Edits start from the original frames by
  default (base="original"), so re-running apply_to_target with other settings is safe.
Workspace: {workspace_root()}
"""

mcp = MCPServer("flowedits", instructions=INSTRUCTIONS)

WAIT = 40  # seconds a tool waits for a background job before handing back a job_id (clients time out ~60 s)

RO = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
EDIT = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False)
WRITE_FILES = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)
DESTRUCTIVE = ToolAnnotations(read_only_hint=False, destructive_hint=True, idempotent_hint=False, open_world_hint=False)

PastedPath = "Path exactly as the user gave it: absolute, ~/..., quoted, file:// URL, or just a file name."
FrameSelection = Annotated[
    list[int] | str | None,
    Field(description='Frames to act on: list of indices, a range string like "0-120,200-", or null for all.'),
]
TimeValue = float | str | None
StartTime = Annotated[
    TimeValue,
    Field(description='Only frames at or after this time (seconds on the source timeline: 12.5, "12.5s", "0:12.5").'),
]
EndTime = Annotated[TimeValue, Field(description="Only frames before this time (same formats as start_time).")]
WaitSeconds = Annotated[
    float, Field(ge=0, le=50, description="How long to wait for completion before returning a job_id.")
]


def tool(annotations: ToolAnnotations, name: str | None = None):
    """Register a tool whose expected failures reach the model with their message.

    The SDK hides the text of arbitrary exceptions (only ToolError messages are shown), but our
    errors carry the fix: "several files match: ...", "lower match_threshold", "outside the window".
    """

    def decorate(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except (ProjectError, FFmpegError) as exc:
                raise ToolError(str(exc)) from exc

        return mcp.tool(name=name, annotations=annotations)(wrapper)

    return decorate


def _out(result: dict, image: bytes | None = None) -> list:
    content: list = [json.dumps(result, indent=1, default=str)]
    if image:
        content.append(Image(data=image, format="jpeg"))
    return content


async def _run(fn, *args, **kwargs):
    # Quick work still touches disk/ffmpeg; keep the event loop responsive.
    return await asyncio.to_thread(fn, *args, **kwargs)


def _job_reply(job: Job) -> list:
    if job.status == "done":
        result = job.result if isinstance(job.result, dict) else {"result": job.result}
        return _out({**result, "job": {"job_id": job.id, "status": "done",
                                       "elapsed_seconds": job.to_dict()["elapsed_seconds"]}}, job.preview)
    if job.status == "error":
        raise ProjectError(f"{job.kind} failed: {job.error}")
    info = job.to_dict()
    if job.status == "running":
        info["next_steps"] = (f"Still running in the background. Call job_status(job_id='{job.id}') — it waits up "
                              f"to {WAIT}s and returns the result when done. Tell the user the progress/ETA.")
    return _out(info)


async def _job(key: str, kind: str, fn, wait_seconds: float) -> list:
    job = JOBS.start(key, kind, fn)
    await asyncio.to_thread(job.wait, wait_seconds)
    return _job_reply(job)


def _idle(project_id: str) -> None:
    JOBS.ensure_idle(project_id)


# ---- find & open -------------------------------------------------------------


@tool(RO)
async def find_media(
    query: Annotated[str, Field(description='File name or part of it, e.g. "demo.mov", "Screen Recording", "logo".')],
    kind: Literal["video", "image", "any"] = "video",
    limit: Annotated[int, Field(ge=1, le=50)] = 10,
) -> str:
    """Use when a video/photo the user mentions can't be found or matches several files. Searches Desktop,
    Downloads, Movies, Documents, Pictures and home by name, newest first."""

    def search():
        from datetime import datetime

        out = []
        for p in find_files(query, kind, limit):
            st = p.stat()
            out.append({"path": str(p), "size_mb": round(st.st_size / 1e6, 2),
                        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")})
        return {"query": query, "matches": out}

    return json.dumps(await _run(search), indent=1)


@tool(RO)
async def probe_video(video_path: Annotated[str, Field(description=f"The video. {PastedPath}")]) -> str:
    """Use to answer questions about a video (length, resolution, fps, frame count, audio) without opening it."""
    from .media import probe

    return json.dumps(await _run(lambda: probe(resolve_media_path(video_path, "video"))), indent=1)


@tool(EDIT)
async def open_video(
    video_path: Annotated[str, Field(description=f"The video to edit. {PastedPath}")],
    target_image_path: Annotated[
        str | None,
        Field(description="Image of the thing to replace/remove/modify (logo, poster, product...), found in every "
                          f"frame. A tight crop of how it appears in the video. {PastedPath}"),
    ] = None,
    edit_start_time: Annotated[
        TimeValue,
        Field(description="Start of the period to edit; frames before it are never changed. "
                          'Seconds on the source timeline: 12.5, "12.5s", "0:12.5".'),
    ] = None,
    edit_end_time: Annotated[TimeValue, Field(description="End of the period to edit; frames after it are never changed.")] = None,
    match_threshold: Annotated[float, Field(ge=0.2, le=0.99, description="Detection confidence needed (0-1).")] = 0.7,
    detection_method: Annotated[
        Literal["auto", "template", "features"],
        Field(description="template = flat overlays; features = rotated/perspective/moving camera; auto = both."),
    ] = "auto",
    trim_start_time: Annotated[TimeValue, Field(description="CUT: drop everything before this time from the output.")] = None,
    trim_end_time: Annotated[TimeValue, Field(description="CUT: drop everything after this time from the output.")] = None,
    fps: Annotated[float | None, Field(description="Output frame rate. Default: the video's own.")] = None,
    max_width: Annotated[int | None, Field(description="Downscale wider videos to this width.")] = None,
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use FIRST whenever the user wants to change a video. Opens it for frame-by-frame editing (any length: frames
    are decoded on demand) and, with target_image_path, finds that image in every frame. Returns project_id, which
    frames/times contain the target and a preview. Photos/folders are also accepted (same as open_photos)."""
    return await _job(f"open:{video_path}", "open_video", lambda job: ops.split_video(
        video_path, target_image_path, fps, trim_start_time, trim_end_time, None, max_width, match_threshold,
        detection_method, edit_start_time, edit_end_time, job=job), wait_seconds)


@tool(EDIT)
async def open_photos(
    photo_path: Annotated[str, Field(description=f"A photo, or a folder of photos (each is one frame). {PastedPath}")],
    target_image_path: Annotated[
        str | None,
        Field(description=f"Image of the thing to replace/remove/modify, found in every photo. {PastedPath}"),
    ] = None,
    match_threshold: Annotated[float, Field(ge=0.2, le=0.99)] = 0.7,
    detection_method: Literal["auto", "template", "features"] = "auto",
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use FIRST whenever the user wants to change a photo or a set of photos. Each photo becomes one frame, so all
    editing tools work the same. Finish with save_photos."""
    return await _job(f"open:{photo_path}", "open_photos", lambda job: ops.open_photos(
        photo_path, target_image_path, match_threshold, detection_method, job=job), wait_seconds)


@tool(EDIT)
async def locate_target(
    project_id: str,
    target_image_path: Annotated[str, Field(description=f"Image to find in each frame. {PastedPath}")],
    match_threshold: Annotated[float, Field(ge=0.2, le=0.99)] = 0.7,
    method: Literal["auto", "template", "features"] = "auto",
    max_gap: Annotated[int, Field(ge=0, description="Fill up to this many missed frames between detections.")] = 3,
    smooth_window: Annotated[int, Field(ge=1, description="Average corners over N frames to remove jitter.")] = 3,
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use to set or change the image to track, or to retry when too few/too many frames matched (lower
    match_threshold, or method='features' for tilted/rotated targets). Searches only inside the edit window."""
    _idle(project_id)
    return await _job(project_id, "locate_target", lambda job: ops.locate_target(
        project_id, target_image_path, match_threshold, method, max_gap, smooth_window, job=job), wait_seconds)


@tool(EDIT)
async def set_edit_window(
    project_id: str,
    start_time: Annotated[TimeValue, Field(description='Window start (12.5, "12.5s", "0:12.5"). null = from 0.')] = None,
    end_time: Annotated[TimeValue, Field(description="Window end. null = to the end of the video.")] = None,
) -> str:
    """Use when the user limits changes to a time period after opening the video (or pass both null to clear).
    Frames outside the window are never modified."""
    _idle(project_id)
    return json.dumps(await _run(ops.set_edit_window, project_id, start_time, end_time), indent=1)


# ---- edit ----------------------------------------------------------------------


@tool(EDIT)
async def apply_to_target(
    project_id: str,
    operation: Annotated[
        Literal["replace", "remove", "blur", "pixelate", "fill", "highlight"],
        Field(description="replace: put replacement image on the target; remove: erase + fill from surroundings; "
                          "blur/pixelate: obscure; fill: solid color; highlight: outline it (to check tracking)."),
    ],
    replacement_path: Annotated[
        str | None,
        Field(description="For replace: ONE image you made (PNG transparency supported), or a folder of images to "
                          f"show in sequence over time. {PastedPath}"),
    ] = None,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
    fit: Annotated[
        Literal["cover", "contain", "stretch"],
        Field(description="How the replacement fills the target's shape when aspect ratios differ."),
    ] = "cover",
    feather: Annotated[int, Field(ge=0, le=50, description="Edge softening in pixels.")] = 2,
    padding: Annotated[int, Field(ge=0, le=100, description="Grow the affected area by this many pixels.")] = 4,
    opacity: Annotated[float, Field(ge=0, le=1)] = 1.0,
    strength: Annotated[int, Field(ge=1, le=200, description="Blur radius / pixel block size.")] = 25,
    fill_color: Annotated[str, Field(description="Hex color for fill.")] = "#000000",
    frames_per_image: Annotated[
        int | None,
        Field(description="With a folder of replacements: frames each image stays on screen. Default: spread evenly."),
    ] = None,
    base: Annotated[
        Literal["original", "current"],
        Field(description="original = replace earlier edits on these frames; current = stack on top of them."),
    ] = "original",
) -> list:
    """Use to change the located target in EVERY frame where it appears with one call — it follows the target's
    position, size and perspective. Instant: the edit is recorded and previewed; render_video/save_photos
    produces the result. Only frames inside the edit window (and start_time/end_time) are changed."""
    _idle(project_id)
    result, preview = await _run(
        ops.apply_to_target, project_id, operation, replacement_path, frame_indices, fit, feather, padding,
        opacity, strength, fill_color, frames_per_image, base, start_time, end_time,
    )
    return _out(result, preview)


@tool(WRITE_FILES)
async def export_target_regions(
    project_id: str,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
    base: Literal["original", "current"] = "original",
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use when specific frames need DIFFERENT artwork on the target. Saves the target area of each frame as an
    upright crop (regions/region_<index>.png) to edit in place; then call import_target_regions."""
    _idle(project_id)
    return await _job(project_id, "export_target_regions", lambda job: ops.export_target_regions(
        project_id, frame_indices, base, start_time, end_time, job=job), wait_seconds)


@tool(EDIT)
async def import_target_regions(project_id: str, feather: Annotated[int, Field(ge=0, le=50)] = 2) -> list:
    """Use after editing exported region crops: pastes changed crops back with the original perspective."""
    _idle(project_id)
    result, preview = await _run(ops.import_target_regions, project_id, feather)
    return _out(result, preview)


@tool(WRITE_FILES)
async def export_frames(
    project_id: str,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
    edited: Annotated[bool, Field(description="Include edits made so far.")] = True,
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use to get whole frames as PNG files (max 300 per call) to edit by hand; send each back with update_frame."""
    return await _job(project_id, "export_frames", lambda job: ops.export_frames(
        project_id, frame_indices, start_time, end_time, edited, job=job), wait_seconds)


@tool(EDIT)
async def update_frame(
    project_id: str,
    frame_index: int,
    image_path: Annotated[str, Field(description="Edited full frame (resized to fit; transparent pixels keep the "
                                                 f"current frame). {PastedPath}")],
) -> str:
    """Use to replace one whole frame with an image you edited. Refused outside the edit window."""
    _idle(project_id)
    return json.dumps(await _run(ops.update_frame, project_id, frame_index, image_path), indent=1)


@tool(EDIT)
async def reset_frames(
    project_id: str,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
) -> str:
    """Use for undo: frames revert to the original (all edited frames when nothing is selected)."""
    _idle(project_id)
    return json.dumps(await _run(ops.reset_frames, project_id, frame_indices, start_time, end_time), indent=1)


# ---- look ----------------------------------------------------------------------


@tool(RO)
async def view_frame(
    project_id: str,
    frame_index: int,
    edited: Annotated[bool, Field(description="Show with edits applied.")] = True,
    annotate: Annotated[bool, Field(description="Outline the detected target.")] = False,
    max_width: Annotated[int, Field(ge=128, le=4096)] = 1024,
) -> list:
    """Use to look at one frame closely (returns the image and saves it as a PNG you can edit)."""
    result, image = await _run(ops.view_frame, project_id, frame_index, edited, annotate, max_width)
    return _out(result, image)


@tool(RO)
async def contact_sheet(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=36)] = 12,
    only_edited: bool = False,
    only_with_target: bool = False,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
    edited: bool = True,
    annotate: bool = False,
    columns: Annotated[int, Field(ge=1, le=8)] = 4,
) -> list:
    """Use to check edits or tracking across the whole video in ONE image (evenly spaced frames). After
    apply_to_target, call with only_edited=true before rendering."""
    result, image = await _run(ops.contact_sheet, project_id, count, frame_indices, only_with_target, only_edited,
                               edited, annotate, columns, start_time, end_time)
    return _out(result, image)


@tool(RO)
async def list_frames(
    project_id: str,
    frame_indices: FrameSelection = None,
    start_time: StartTime = None,
    end_time: EndTime = None,
    only_with_target: bool = False,
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
) -> str:
    """Use for exact per-frame data: time, edit state, edit-window membership, target position (quad/bbox)."""
    return json.dumps(
        await _run(ops.list_frames, project_id, frame_indices, only_with_target, limit, start_time, end_time),
        indent=1,
    )


@tool(RO)
async def project_status(project_id: str) -> str:
    """Use to recall a project's state: source, edit window, target, recorded edits, outputs, running job."""
    status = await _run(ops.project_status, project_id)
    job = JOBS.running(project_id)
    if job:
        status["running_job"] = job.to_dict()
    return json.dumps(status, indent=1, default=str)


# ---- finish --------------------------------------------------------------------


@tool(WRITE_FILES)
async def render_video(
    project_id: str,
    output_path: Annotated[
        str | None,
        Field(description="Where to write (.mp4, .mov, .mkv, .webm or .gif). Default: next to the original as "
                          "<name>_edited.<ext>."),
    ] = None,
    quality: Annotated[
        Literal["draft", "standard", "high"],
        Field(description="draft = fastest (quick check of a long video), standard = default, high = best/slowest."),
    ] = "standard",
    keep_audio: bool = True,
    replace_original: Annotated[
        bool,
        Field(description="Overwrite the user's original video (only when they ask). Keeps <name>.original.<ext>."),
    ] = False,
    use_edited: Annotated[bool, Field(description="false = re-encode without any edits.")] = True,
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use LAST for videos: renders every frame (edits applied, untouched frames copied) into the final video with
    the original audio. 10 minutes of 1080p takes a few minutes — returns a job_id to poll with job_status."""
    _idle(project_id)
    return await _job(project_id, "render_video", lambda job: ops.assemble(
        project_id, output_path, keep_audio, None, use_edited, replace_original, quality, job=job), wait_seconds)


@tool(WRITE_FILES)
async def save_photos(
    project_id: str,
    output_path: Annotated[
        str | None,
        Field(description="A file (one photo) or a folder. Default: next to each original as <name>_edited.<ext>."),
    ] = None,
    replace_original: Annotated[
        bool,
        Field(description="Overwrite the original photos (only when the user asks). Keeps <name>.original.<ext>."),
    ] = False,
    frame_indices: FrameSelection = None,
    wait_seconds: WaitSeconds = WAIT,
) -> list:
    """Use LAST for photos: writes every edited photo in its own format (unedited ones are skipped)."""
    _idle(project_id)
    return await _job(project_id, "save_photos", lambda job: ops.save_photos(
        project_id, output_path, replace_original, frame_indices, job=job), wait_seconds)


# ---- jobs & projects -------------------------------------------------------------


@tool(RO)
async def job_status(job_id: str, wait_seconds: WaitSeconds = WAIT) -> list:
    """Use whenever a tool returned a job_id: waits up to wait_seconds, then returns progress/ETA, or the full
    result (and preview) once done. Keep calling until status is "done"."""
    job = JOBS.get(job_id)
    await asyncio.to_thread(job.wait, wait_seconds)
    return _job_reply(job)


@tool(EDIT)
async def cancel_job(job_id: str) -> str:
    """Use when the user wants to stop a running render/detection. Partial output files are removed."""
    job = JOBS.get(job_id)
    job.cancel()
    await asyncio.to_thread(job.wait, 10)
    return json.dumps(job.to_dict(), indent=1)


@tool(RO, name="list_projects")
async def list_projects_tool() -> str:
    """Use to find earlier projects (most recent first) and running jobs."""
    return json.dumps({"projects": list_projects()[:50], "jobs": JOBS.list()[:20]}, indent=1)


@tool(DESTRUCTIVE, name="delete_project")
async def delete_project_tool(project_id: str) -> str:
    """Use only when the user asks to delete a project's working data. Never touches their videos/photos or
    rendered outputs."""
    _idle(project_id)
    await _run(delete_project, project_id)
    return json.dumps({"deleted": project_id})



@mcp.prompt()
def edit_image_in_video(
    video_path: str, target_image_path: str, instruction: str, start_time: str = "", end_time: str = ""
) -> str:
    """Replace/remove/change an image that appears in a video (or photo), optionally only during a time period."""
    window = ""
    if start_time or end_time:
        window = (f"\nOnly change the video between {start_time or 'the start'} and {end_time or 'the end'}; every "
                  f"other frame must stay untouched (pass edit_start_time / edit_end_time).")
    return f"""Edit {video_path}.
The image at {target_image_path} appears in it. Instruction: {instruction}{window}

1. open_video (or open_photos for photos) with the path and target_image_path. If it returns a job_id,
   call job_status until done. Check the preview outlines the right thing; if few frames matched,
   call locate_target with a lower match_threshold or method="features".
2. remove/hide -> apply_to_target(operation remove/blur/pixelate). Replace -> create ONE image matching
   the target's aspect ratio, save it, apply_to_target(operation="replace", replacement_path=...).
3. contact_sheet(only_edited=true) to verify, then render_video (videos) or save_photos (photos),
   polling job_status if needed. Report the output path."""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
