# flowedits

An MCP server for editing videos frame by frame, and photos the same way. The AI agent handles the
creative part: it decides the edit and makes one replacement image (or edits a few crops). The server
does all the mechanical work:

- decoding the video frame by frame (any length; nothing is extracted to disk)
- finding a target image (logo, poster, product, sign...) in **every** frame, including its position,
  scale and perspective
- applying the edit to that target in every frame
- encoding the result with the original audio, saved next to your original

It handles long videos: 10 minutes at 30 fps (18,000 frames) is supported. See
[Long videos](#long-videos).

## How it works

```
video.mp4 ─open_video─► project (frames decoded on demand through ffmpeg pipes)
target.png ───────────► position of the target in each frame   (template match + SIFT homography,
                                                                  parallel, gap filling, smoothing)
ONE image ─apply_to_target─► recorded in the edit recipe       (instant; re-renders if the target
                                                                  is located again)
render_video ─► decode ► apply the recipe to edited frames ► encode ─► video_edited.mp4 (+ audio)
```

Edits are stored as a **recipe**, an ordered list of operations, not as edited pixels.
`apply_to_target` returns instantly with a preview. The pixels are produced in one streaming pass by
`render_video` (videos) or `save_photos` (photos). Frames without edits pass straight through.

## Tools

The server's instructions include a table of **what the user says → which tool to call**. Every tool
description starts with "Use when…" and has MCP hints (read-only / destructive), so an agent picks the
right tools without extra prompting.

| Tool | Use when |
|---|---|
| `open_video` | The user wants to change a video. Opens it and, with `target_image_path`, finds that image in every frame. `edit_start_time`/`edit_end_time` limit edits to a time period |
| `open_photos` | The user wants to change a photo or a folder of photos (each photo is one frame) |
| `locate_target` | Setting or changing the image to track, or retrying with another threshold/method |
| `set_edit_window` | Limiting (or un-limiting) edits to a time period after opening |
| `apply_to_target` | Changing the target in every frame at once: `replace` (an image, or a folder shown in sequence), `remove`, `blur`, `pixelate`, `fill`, `highlight` |
| `export_target_regions` / `import_target_regions` | Specific frames need different artwork on the target |
| `export_frames` / `update_frame` | Editing whole frames by hand |
| `reset_frames` | Undo |
| `contact_sheet`, `view_frame`, `list_frames`, `project_status` | Checking tracking and edits (images are returned to the model) |
| `render_video` | Finished with a video: writes `<name>_edited.<ext>` next to the original, with audio |
| `save_photos` | Finished with photos: writes `<name>_edited.<ext>` next to each original |
| `job_status`, `cancel_job` | A tool returned a `job_id` (long work runs in the background) |
| `find_media`, `probe_video` | Finding a file by name, or reading a video's length / fps / resolution |
| `list_projects`, `delete_project` | Managing earlier projects |

There is also an MCP prompt, `edit_image_in_video(video_path, target_image_path, instruction,
start_time, end_time)`, that walks the agent through the workflow.

### Typical agent session

1. `open_video("~/Movies/ad.mp4", target_image_path="old_logo.png")`: the reply lists the frames and
   times where the logo appears, with a preview.
2. The agent generates `new_logo.png` with its image model.
3. `apply_to_target(project_id, "replace", replacement_path="new_logo.png")`
4. `contact_sheet(project_id, only_edited=true)` to check the result.
5. `render_video(project_id)` → `~/Movies/ad_edited.mp4`

That's five tool calls and one generated image, however long the video is.

## Long videos

A tool that may run for a while (`open_video`/`open_photos` with a target, `locate_target`,
`export_*`, `render_video`, `save_photos`) waits up to about 40 seconds. If it finishes in that time it
returns the result directly. Otherwise it returns a `job_id`, and the agent calls
`job_status(job_id)` (each call waits up to about 40 seconds) until the job is done, reporting
progress and an ETA along the way. This keeps every call under MCP client timeouts. While a job
runs, editing tools on that project are refused with a message naming the job. `cancel_job` stops a
job and removes any partial output.

What makes long videos work:

- **Nothing is extracted.** Frames are decoded on demand through ffmpeg pipes, so 18,000 frames of
  1080p need no temporary disk space (as PNGs they would take about 50 GB).
- **Detection is parallel.** The video is split into chunks, one thread per chunk (up to 8), each
  with its own ffmpeg decoder. Frames are analysed at up to 1280 px wide and positions are scaled
  back to full size.
- **The target is not searched for in every frame while it's missing.** While it's being tracked,
  every frame gets a cheap search near its last size. While it's missing, only every 5th frame gets
  the full search, and when the target reappears the skipped frames are searched backwards, so no
  appearance is missed.
- **Rendering is one pass.** Decode → apply edits → encode with the source audio. Frames without
  edits are copied through untouched.

Benchmark (Apple M5, 10 cores): a 10-minute 1080p, 30 fps video with a moving logo (18,000 frames):

| Step | Time |
|---|---|
| `open_video` | 0.2 s |
| `locate_target` (all 18,000 frames; the logo was found in exactly its 5 on-screen periods) | 92 s |
| `apply_to_target` (replace, 15,005 frames) | 0.2 s |
| `render_video` (standard quality, with audio) | 214 s |

That's about 5 minutes in total, with no temporary disk space. Rendering runs at encoding speed
(edits are applied only to a crop around the target, on a thread pool).
`render_video(quality="draft")` encodes faster for a quick check, and `"high"` gives the best
quality. Use `edit_start_time`/`edit_end_time` when only part of a long video needs changing: only
that part is searched.

## Photos

Photos use the same workflow. Open one photo or a whole folder:

```
open_photos("~/Pictures/Shop Front.jpg", target_image_path="old_logo.png")
apply_to_target(project_id, "replace", replacement_path="new_logo.png")
save_photos(project_id)   # -> ~/Pictures/Shop Front_edited.jpg
```

- `open_video` also accepts a photo or a folder.
- Each photo is one frame (`frame_index`), and the target is found in each photo independently. A
  folder can mix sizes, and only some photos need to contain the target.
- Edited photos keep their format and full resolution. Unedited photos are skipped. `output_path`
  takes a file (one photo) or a folder. `replace_original=true` overwrites the originals and keeps
  `<name>.original.<ext>` backups.
- Time options are video-only. Select photos with `frame_indices` instead.
- Saved photos don't keep EXIF metadata. The rotation from EXIF is applied to the pixels, so the
  photo still displays the right way up.

## Pasting paths

Paste the path to a video or image in any form and the server finds the file:

- `/Users/me/Movies/My Trip.mov`, `~/Movies/My Trip.mov` or `"…/My Trip.mov"` (quoted)
- `/Users/me/Movies/My\ Trip.mov` (escaped spaces, as when you drag a file into a terminal)
- `file:///Users/me/Movies/My%20Trip.mov`
- only the name, `My Trip.mov` or `my trip`: the server searches Desktop, Downloads, Movies, Documents,
  Pictures, your home folder and any folders listed in `FLOWEDITS_SEARCH_DIRS`
- macOS screen recordings work even though their file names contain an invisible special space
  before "PM"

If the name matches several files, the server lists them so you can pick one.

Results are saved **next to the original** as `<name>_edited.<ext>`. Your original is never
overwritten unless you ask: `replace_original=true` writes the edited version to the original's path
and keeps the untouched file as `<name>.original.<ext>`.

## Editing only part of a video

```
open_video("~/Movies/ad.mp4", target_image_path="old_logo.png",
           edit_start_time="0:12", edit_end_time="0:18.5")
```

- The target is only searched for between 12 s and 18.5 s, and only those frames can be changed.
  `update_frame` refuses frames outside that period.
- `render_video` still outputs the **whole** video. Frames outside the period are copied unchanged.
- `set_edit_window` changes or clears the period later. If the new period includes frames that
  haven't been searched, it says to run `locate_target` again.
- `apply_to_target`, `export_*`, `reset_frames`, `contact_sheet` and `list_frames` also take
  `start_time` / `end_time` to narrow a single call further. They can never go outside the edit
  window.

Times are on the original video's timeline: `12`, `12.5`, `"12.5s"`, `"0:12.5"` or `"00:00:12.5"`.
`trim_start_time` / `trim_end_time` are different: they cut the output video itself.

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
      "env": { "FLOWEDITS_WORKSPACE": "/Users/abhishek/.flowedits/projects" }
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
| `FLOWEDITS_WORKSPACE` | `~/.flowedits/projects` | Where projects (detections, edit recipes, photo copies) are stored |
| `FLOWEDITS_SEARCH_DIRS` | (none) | Extra folders, separated by `:`, to search when only a file name is given |
| `FFMPEG_BINARY` | ffmpeg on `PATH`, otherwise the bundled build | ffmpeg to use |

## Tips

- The target image should be a tight crop of how the target looks in the video. A screenshot
  taken from one of the frames works best.
- If too few frames match, check `best_score_in_unmatched_frames`. Lower `match_threshold`, or use
  `method="features"` when the target is rotated, tilted or filmed by a moving camera.
- `remove` fills the area from its surroundings (OpenCV inpainting), which works best for small or
  medium targets. For a large target, `replace` with a generated background patch usually looks
  better.
- Projects made by flowedits before the streaming redesign can't be opened. Open the video again.

## Tests

```bash
uv run pytest
```
