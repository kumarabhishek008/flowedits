# video-frame-mcp

An MCP server for editing videos frame by frame. The AI agent handles the creative part: it
looks at frames and makes one replacement image or edits a few crops. The server handles
everything mechanical:

- splitting the video into frames
- finding a target image (logo, poster, product, sign...) in **every** frame, including its
  position, scale and perspective
- applying an edit to that target in every frame
- joining the frames back into a video with the original audio

## How it works

```
video.mp4 ──split_video──► frames/frame_000000.png …      (ffmpeg, constant fps)
target.png ─────────────► detections per frame            (template match + SIFT homography,
                                                            gap filling, jitter smoothing)
agent makes ONE image ──apply_to_target──► edited/…       (warped onto the target in each frame)
                         export/import_target_regions      (per-frame hand edits, optional)
edited + original frames ──assemble_video──► output.mp4    (H.264 + source audio)
```

Edits are written to `edited/` and never overwrite the original frames. `assemble_video` uses the
edited version of each frame when there is one and the original otherwise.

## Tools

| Tool | Purpose |
|---|---|
| `probe_video` | Metadata: size, fps, frame count, duration, audio |
| `split_video` | Extract frames into a new project. If `target_image_path` is given, it also locates the target and returns a preview with the target outlined |
| `locate_target` | Find the target again with another image, threshold or method (`auto` / `template` / `features`) |
| `apply_to_target` | Bulk edit on every frame that contains the target: `replace` (an image file, or a folder of images shown in sequence), `remove` (inpaint), `blur`, `pixelate`, `fill`, `highlight` |
| `export_target_regions` / `import_target_regions` | Save the target area of each frame as an upright crop, let the agent edit any of the crops, then paste them back with the original perspective |
| `update_frame` | Replace one whole frame with an edited image |
| `reset_frames` | Revert edits |
| `view_frame`, `contact_sheet`, `list_frames` | Check frames and edits (images are returned to the model) |
| `assemble_video` | Join the frames into `.mp4` / `.mov` / `.webm` / `.gif`, keeping the source audio |
| `project_status`, `list_video_projects`, `delete_video_project` | Manage projects |

There is also an MCP prompt, `edit_image_in_video(video_path, target_image_path, instruction)`,
that walks the agent through the workflow.

### Typical agent session

1. `split_video("/videos/ad.mp4", target_image_path="/videos/old_logo.png")`: the reply says 412 of 480
   frames contain the target, gives their ranges, and includes a preview.
2. The agent generates `new_logo.png` with its image model.
3. `apply_to_target(project_id, "replace", replacement_path="/videos/new_logo.png")`
4. `contact_sheet(project_id, only_edited=True)` to check the result.
5. `assemble_video(project_id)` returns the path of the output video.

That is five tool calls and one generated image, however long the video is.

## Install

Requires Python 3.10+. ffmpeg is used from `PATH` if present. Otherwise the static build bundled
with `imageio-ffmpeg` is used, so you don't need to install anything else.

```bash
uv sync
```

### Claude Code / Claude Desktop config

```json
{
  "mcpServers": {
    "video-frame-editor": {
      "command": "uv",
      "args": ["--directory", "/Users/abhishek/stiddle/video-frame-mcp", "run", "video-frame-mcp"],
      "env": { "VIDEO_FRAME_MCP_WORKSPACE": "/Users/abhishek/video-frame-projects" }
    }
  }
}
```

Or with the Claude Code CLI:

```bash
claude mcp add video-frame-editor -- uv --directory /Users/abhishek/stiddle/video-frame-mcp run video-frame-mcp
```

## Configuration

| Env var | Default | |
|---|---|---|
| `VIDEO_FRAME_MCP_WORKSPACE` | `~/.video-frame-mcp/projects` | Where frames, edits and outputs are stored |
| `FFMPEG_BINARY` | ffmpeg on `PATH`, otherwise the bundled build | ffmpeg to use |

## Tips

- The target image should be a tight crop of how the target looks in the video. A screenshot
  taken from one of the frames works best.
- If `split_video` finds too few frames, check `best_score_in_unmatched_frames`. Lower
  `match_threshold`, or switch to `method="features"` when the target is rotated, tilted or filmed
  by a moving camera.
- Long or 4K videos produce a lot of frames. Use `start_time`/`end_time`, `max_width` or
  `image_format="jpg"` to keep the workspace small.
- `remove` uses OpenCV inpainting, which works best for small or medium targets. For a large target,
  a `replace` with a generated background patch usually looks better.

## Tests

```bash
uv run pytest
```
