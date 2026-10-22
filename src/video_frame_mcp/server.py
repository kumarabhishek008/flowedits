"""MCP server: split a video into frames, find a target image in each frame, edit it in bulk, rebuild the video.

The AI agent only does the creative part (looking at frames, generating a replacement image or
editing a few crops). Everything mechanical - decoding, locating the target in every frame,
warping edits onto each frame, encoding the result with the original audio - happens here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Annotated, Literal

from mcp.server.mcpserver import Image, MCPServer
from pydantic import Field

from . import operations as ops
from .workspace import WORKSPACE, delete_project, list_projects

INSTRUCTIONS = f"""\
Video frame editing tools. Typical workflow for "replace / remove / change <this image> in the video":

1. split_video(video_path, target_image_path=<image to find>)
   -> extracts every frame and locates the target image in each one (position + perspective).
      The reply lists which frames contain it, and shows a preview with the target outlined.
2. Decide the edit. You do NOT need to edit every frame yourself:
   - replace: generate/choose ONE replacement image and call apply_to_target(operation="replace",
     replacement_path=...). It is warped onto the target in every frame, following its motion.
     Pass a folder of images to show different images over time (frames_per_image controls pacing).
   - remove / blur / pixelate / fill: apply_to_target(operation=...) — no image needed.
   - frame-specific changes: export_target_regions -> edit the region_*.png crops in place ->
     import_target_regions. Or update_frame for a full-frame replacement of a single frame.
3. Verify with contact_sheet(only_edited=True) or view_frame.
4. assemble_video(project_id) -> joins all frames (edited where available) into a video with the
   original audio.

Edits always start from the original frames by default, so re-running apply_to_target with other
settings is safe; reset_frames reverts. Frame indices accept ints, lists or strings like "0-40,55".
Workspace: {WORKSPACE}
"""

mcp = MCPServer("video-frame-editor", instructions=INSTRUCTIONS)

FrameSelection = Annotated[
    list[int] | str | None,
    Field(description='Frames to act on: list of indices, a range string like "0-120,200-", or null for all.'),
]


def _out(result: dict, image: bytes | None = None) -> list:
    content: list = [json.dumps(result, indent=1, default=str)]
    if image:
        content.append(Image(data=image, format="jpeg"))
    return content


async def _run(fn, *args, **kwargs):
    # The heavy lifting is CPU/ffmpeg bound; keep the event loop responsive.
    return await asyncio.to_thread(fn, *args, **kwargs)


@mcp.tool()
async def probe_video(video_path: Annotated[str, Field(description="Absolute path to a video file.")]) -> str:
    """Read video metadata (size, fps, frame count, duration, audio) without extracting anything."""
    from .media import probe

    return json.dumps(await _run(probe, video_path), indent=1)


@mcp.tool()
async def split_video(
    video_path: Annotated[str, Field(description="Absolute path to the input video.")],
    target_image_path: Annotated[
        str | None,
        Field(description="Optional image (logo, poster, product...) to find in every frame. It is the thing to "
                          "replace/remove/modify. Should be a tight crop of how it appears in the video."),
    ] = None,
    fps: Annotated[float | None, Field(description="Frames per second to extract. Default: source fps.")] = None,
    start_time: Annotated[float | None, Field(description="Start time in seconds.")] = None,
    end_time: Annotated[float | None, Field(description="End time in seconds.")] = None,
    max_frames: Annotated[int | None, Field(description="Stop after this many frames.")] = None,
    max_width: Annotated[int | None, Field(description="Downscale frames wider than this (keeps aspect).")] = None,
    image_format: Literal["png", "jpg"] = "png",
    match_threshold: Annotated[float, Field(ge=0.2, le=0.99, description="Detection confidence needed (0-1).")] = 0.7,
    detection_method: Literal["auto", "template", "features"] = "auto",
) -> list:
    """Split a video into individual frames (new project) and, if target_image_path is given, locate that image in
    every frame. Returns project_id, frame count, which frames contain the target and a preview of the first match."""
    result, preview = await _run(
        ops.split_video, video_path, target_image_path, fps, start_time, end_time, max_frames, max_width,
        image_format, match_threshold, detection_method,
    )
    return _out(result, preview)


@mcp.tool()
async def locate_target(
    project_id: str,
    target_image_path: Annotated[str, Field(description="Image to find in each frame.")],
    match_threshold: Annotated[float, Field(ge=0.2, le=0.99)] = 0.7,
    method: Annotated[
        Literal["auto", "template", "features"],
        Field(description="template = fast, axis-aligned overlays; features = rotated/perspective/moving camera; "
                          "auto = template, falling back to features."),
    ] = "auto",
    max_gap: Annotated[int, Field(ge=0, description="Fill up to this many missed frames between detections.")] = 3,
    smooth_window: Annotated[int, Field(ge=1, description="Average corners over N frames to remove jitter.")] = 3,
) -> list:
    """(Re)locate a target image in every frame of a project. Use to change the target, or retry with a different
    threshold/method when split_video found too few or too many matches. Replaces previous detections."""
    result, preview = await _run(ops.locate_target, project_id, target_image_path, match_threshold, method,
                                 max_gap, smooth_window)
    return _out(result, preview)


@mcp.tool()
async def apply_to_target(
    project_id: str,
    operation: Annotated[
        Literal["replace", "remove", "blur", "pixelate", "fill", "highlight"],
        Field(description="replace: put replacement image on the target; remove: erase + inpaint; blur/pixelate: "
                          "obscure; fill: solid color; highlight: draw the detected outline (debug)."),
    ],
    replacement_path: Annotated[
        str | None,
        Field(description="For replace: an image file (PNG transparency supported), or a folder of images to show "
                          "in sequence across the frames."),
    ] = None,
    frame_indices: FrameSelection = None,
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
        Field(description="With a folder of replacements: frames each image stays on screen (cycles). "
                          "Default spreads images evenly."),
    ] = None,
    base: Annotated[
        Literal["original", "current"],
        Field(description="original = start from untouched frames (idempotent); current = stack on earlier edits."),
    ] = "original",
) -> list:
    """Apply one edit to the located target in every (selected) frame where it was found, following its position,
    scale and perspective. This is the bulk operation: generate one replacement image, call this once."""
    result, preview = await _run(
        ops.apply_to_target, project_id, operation, replacement_path, frame_indices, fit, feather, padding,
        opacity, strength, fill_color, frames_per_image, base,
    )
    return _out(result, preview)


@mcp.tool()
async def export_target_regions(
    project_id: str,
    frame_indices: FrameSelection = None,
    base: Literal["original", "current"] = "original",
) -> str:
    """Save the target area of each frame as an upright, same-sized crop (regions/region_<index>.png) so you can edit
    them individually (e.g. different content per frame). Follow with import_target_regions."""
    return json.dumps(await _run(ops.export_target_regions, project_id, frame_indices, base), indent=1)


@mcp.tool()
async def import_target_regions(
    project_id: str,
    feather: Annotated[int, Field(ge=0, le=50)] = 2,
) -> list:
    """Paste edited region crops back into their frames (with the original perspective). Unchanged crops are skipped."""
    result, preview = await _run(ops.import_target_regions, project_id, feather)
    return _out(result, preview)


@mcp.tool()
async def update_frame(
    project_id: str,
    frame_index: int,
    image_path: Annotated[str, Field(description="Edited full frame. Resized to the frame size if needed; transparent "
                                                 "pixels keep the current frame.")],
) -> str:
    """Replace one whole frame with an image you edited."""
    return json.dumps(await _run(ops.update_frame, project_id, frame_index, image_path), indent=1)


@mcp.tool()
async def reset_frames(project_id: str, frame_indices: FrameSelection = None) -> str:
    """Discard edits so frames revert to the originals (all frames when frame_indices is null)."""
    return json.dumps(await _run(ops.reset_frames, project_id, frame_indices), indent=1)


@mcp.tool()
async def view_frame(
    project_id: str,
    frame_index: int,
    edited: Annotated[bool, Field(description="Show the edited version if one exists.")] = True,
    annotate: Annotated[bool, Field(description="Outline the detected target.")] = False,
    max_width: Annotated[int, Field(ge=128, le=4096)] = 1024,
) -> list:
    """Look at a single frame (as an image) plus its file path and detection info."""
    result, image = await _run(ops.view_frame, project_id, frame_index, edited, annotate, max_width)
    return _out(result, image)


@mcp.tool()
async def contact_sheet(
    project_id: str,
    count: Annotated[int, Field(ge=1, le=36)] = 12,
    frame_indices: FrameSelection = None,
    only_with_target: bool = False,
    only_edited: bool = False,
    edited: bool = True,
    annotate: bool = False,
    columns: Annotated[int, Field(ge=1, le=8)] = 4,
) -> list:
    """Grid of evenly spaced frames in one image — the cheapest way to check tracking or edits across the video."""
    result, image = await _run(ops.contact_sheet, project_id, count, frame_indices, only_with_target, only_edited,
                               edited, annotate, columns)
    return _out(result, image)


@mcp.tool()
async def list_frames(
    project_id: str,
    frame_indices: FrameSelection = None,
    only_with_target: bool = False,
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
) -> str:
    """List frame file paths, timestamps, edit state and target position (quad/bbox)."""
    return json.dumps(await _run(ops.list_frames, project_id, frame_indices, only_with_target, limit), indent=1)


@mcp.tool()
async def assemble_video(
    project_id: str,
    output_path: Annotated[
        str | None,
        Field(description="Where to write the video (.mp4, .mov, .webm or .gif). Default: project output folder."),
    ] = None,
    keep_audio: bool = True,
    fps: Annotated[float | None, Field(description="Override frame rate (drops audio if it differs).")] = None,
    crf: Annotated[int, Field(ge=0, le=51, description="Quality: lower is better/larger.")] = 18,
    use_edited: Annotated[bool, Field(description="false = rebuild from original frames.")] = True,
) -> str:
    """Join all frames (edited where available, original elsewhere) back into a video with the source audio."""
    return json.dumps(await _run(ops.assemble, project_id, output_path, keep_audio, fps, crf, use_edited), indent=1)


@mcp.tool()
async def project_status(project_id: str) -> str:
    """Summary of a project: source, extraction, target detections, edited frames, outputs."""
    return json.dumps(await _run(ops.project_status, project_id), indent=1)


@mcp.tool()
async def list_video_projects() -> str:
    """List existing projects in the workspace (most recent first)."""
    return json.dumps(list_projects(), indent=1)


@mcp.tool()
async def delete_video_project(project_id: str) -> str:
    """Delete a project's frames and outputs from the workspace. Files written to a custom output_path are kept."""
    await _run(delete_project, project_id)
    return json.dumps({"deleted": project_id})


@mcp.prompt()
def edit_image_in_video(video_path: str, target_image_path: str, instruction: str) -> str:
    """Guide the agent through replacing/removing/changing an image that appears throughout a video."""
    return f"""Edit the video at {video_path}.
The image at {target_image_path} appears in the video. Instruction: {instruction}

Steps:
1. Call split_video with video_path and target_image_path. Check the preview: is the outline on the right thing?
   If few frames matched, call locate_target with a lower match_threshold or method="features".
2. If the instruction is to remove/hide it, call apply_to_target with operation remove/blur/pixelate.
   If it should become something else, create ONE replacement image matching the target's aspect ratio
   (see target size in the reply), save it to disk, and call apply_to_target(operation="replace").
   If it should change over time, save several images in a folder and pass the folder.
   Only if each frame needs unique hand edits, use export_target_regions / import_target_regions.
3. Call contact_sheet(only_edited=True) to verify, adjust if needed, then assemble_video.
Report the output path at the end."""


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
