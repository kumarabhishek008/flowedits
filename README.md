# flowedits

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
| `split_video` | Extract frames into a new project. If `target_image_path` is given, it also locates the target and returns a preview with the target outlined. `edit_start_time` / `edit_end_time` limit editing to a time period |
| `set_edit_window` | Set, change or clear the time period that edits are limited to |
| `locate_target` | Find the target again with another image, threshold or method (`auto` / `template` / `features`) |
| `apply_to_target` | Bulk edit on every frame that contains the target: `replace` (an image file, or a folder of images shown in sequence), `remove` (inpaint), `blur`, `pixelate`, `fill`, `highlight` |
| `export_target_regions` / `import_target_regions` | Save the target area of each frame as an upright crop, let the agent edit any of the crops, then paste them back with the original perspective |
| `update_frame` | Replace one whole frame with an edited image |
| `reset_frames` | Revert edits (all, by frame, or by time range) |
| `view_frame`, `contact_sheet`, `list_frames` | Check frames and edits (images are returned to the model) |
| `assemble_video` | Join the frames into `.mp4` / `.mov` / `.webm` / `.gif`, keeping the source audio |
| `project_status`, `list_video_projects`, `delete_video_project` | Manage projects |

There is also an MCP prompt, `edit_image_in_video(video_path, target_image_path, instruction,
start_time, end_time)`, that walks the agent through the workflow.

### Editing only part of the video

Pass a time period to `split_video` and only that part of the video can change:

```
split_video("/videos/ad.mp4", target_image_path="/videos/old_logo.png",
            edit_start_time="0:12", edit_end_time="0:18.5")
```

- The target is only searched for between 12 s and 18.5 s.
- `apply_to_target`, `import_target_regions` and `update_frame` never modify frames outside that
  period. `update_frame` refuses them with an error.
- `assemble_video` still outputs the **whole** video. Frames outside the period are the original
  frames, unchanged.
- `set_edit_window` changes or clears the period later. If the new period includes frames that
  haven't been searched, it says to run `locate_target` again.
- `apply_to_target`, `export_target_regions`, `reset_frames`, `contact_sheet` and `list_frames` also
  take `start_time` / `end_time` to narrow a single call further. They can never go outside the edit
  window.

Times are on the original video's timeline and can be written as `12`, `12.5`, `"12.5s"`, `"0:12.5"`
or `"00:00:12.5"`. `trim_start_time` / `trim_end_time` are different: they cut the output video
itself.

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
    "flowedits": {
      "command": "uv",
      "args": ["--directory", "/Users/abhishek/flowedits", "run", "flowedits"],
      "env": { "FLOWEDITS_WORKSPACE": "/Users/abhishek/flowedits-projects" }
    }
  }
}
```

Or with the Claude Code CLI:

```bash
claude mcp add flowedits -- uv --directory /Users/abhishek/flowedits run flowedits
```

## Configuration

| Env var | Default | |
|---|---|---|
| `FLOWEDITS_WORKSPACE` | `~/.flowedits/projects` | Where frames, edits and outputs are stored |
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
