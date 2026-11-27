# Masking frames while `sam extract` writes them

`spirula sam extract` can tell another process which frames it is writing and
when each one is complete, so masking (`spirula sam track --follow`) runs on
frames as they land instead of after the whole video. The code is
`src/app/FrameStream.h`; `frame_stream_test` covers the watermark and the files.
Without these flags extraction writes and prints exactly what it did before.

## `--plan-out <file>`

Written once, atomically (`<file>.tmp`, then renamed), before the first frame:

```json
{"version":1,"tracks":["/abs/out/cam0","/abs/out/cam1"],"keep":3,"skip":15,
 "decoder":"ffmpeg","input":{"path":"/abs/clip.insv","size":123,"mtime":"..."},
 "options":{"quality":"95","blur_reject":"0","similar":"0",...},"ends":[14,29]}
```

- `tracks`: absolute folders each instant's pictures go to: one track, `cam0`...
  with `--sync`, or the views of a 360 capture. Every track gets the same stems.
  An unsynced multi-track run refuses `--plan-out`, `--progress` and `--plan-in`.
- `ends`: the adaptive plan, the source frame each window closes at. Absent for
  a fixed rate, where window `w` is frames `[w*skip, (w+1)*skip)`.
- `decoder` is the one chosen before decoding; the final progress line names
  the one that ran.
- No stems: the frame a window writes is the sharpest of `keep` and is known
  only once decoded. The progress lines are the only source of stems.

If the plan cannot be written the run stops before the first frame.

## `--progress <file>`

Opened first, so every failure, even one before the plan, ends it with a line.
Each frame (and mask, when extraction masks) is written as `.<name>.<pid>.tmp`
and renamed, so `*.jpg` never matches half a file. At start, such temps of a
process no longer running are removed (on Windows: older than ten minutes);
nothing else is touched. Lines:

```json
{"written":2,"frames":[{"stem":"00012","window":0},{"stem":"00027","window":1}]}
{"written":2,"frames":[{"window":2,"rejected":true}]}
{"written":3,"frames":[{"stem":"00041","window":3}],"done":true,"decoder":"ffmpeg"}
{"written":3,"frames":[],"done":false,"error":"..."}
```

- `frames` lists the entries this line adds, in time order; their concatenation
  is the run. A `rejected` window is one `--blur-reject` / `--similar` left
  empty: it will never get a frame.
- `written` is a low watermark: the first `written` stems are on disk in every
  track. Files land out of order (WriterPool), so a stem waits for every one
  before it. A failed write, or a frame whose mask failed, stops it for good.
- At most one line each 250 ms, written by a thread of its own. The last line
  always has `done`: `true` with `decoder` only when everything was written
  (the progress file itself included), `false` with `error` otherwise. A file
  whose last line has no `done` belongs to a run still going or killed.
- A reader skips a last line with no newline yet: it is still being appended.

## `--plan-in <file>`

Reuses a `--plan-out` file's `ends` and skips the motion pass. Refused, before
the first frame, when the input (size and mtime), the decoder, `--keep`,
`--skip`, the output folders or an option in `options` differ. `--start` /
`--end` clip the plan as usual.

Resuming is by window: a window with a frame on disk in every track is left as
it is (not masked either) and lands with that stem, even if this decode would
pick another frame of it, so a window never gets two. Files of a window found
in only some tracks are removed unless they are the stem picked now. Frames
already there count towards `written`: the watermark is over the plan, not over
this process's writes.

## `sam track --follow [--prime <file>]`

Frame paths come from stdin, one per line, instead of `--frames`; EOF ends the
run. Masks are `<out>/<stem>.png`, written through the same temp names (dead
ones are removed from `<out>` at start), and stdout gets `done <stem>` once each
is on disk, or `failed <stem>` when the frame did not load, the model failed on
it or the mask did not write; the exit code is then 1. `--overlay` is refused.
`--prime` is tracked as frame 0 and not written, so the tracker state equals a
folder run over `[prime, stdin...]`; `--at-frame` counts in that sequence.
`--max-frames n` stops after `n` stdin frames without reading the rest.
