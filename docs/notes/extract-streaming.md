# Masking frames while `sam extract` writes them

`spirula sam extract` can tell another process which frames it is writing and
when each one is complete, so masking (`spirula sam track --follow`) runs on
frames as they land instead of after the whole video. The code is
`src/app/FrameStream.h`; `frame_stream_test` covers the watermark and the files.

## `--plan-out <file>`

Written once, atomically (`<file>.tmp`, then renamed), after the motion pass
and before the first frame:

```json
{"version":1,"tracks":["out/cam0","out/cam1"],"stems":["00014","00029"],
 "keep":3,"skip":15,"ends":[14,29]}
```

- `tracks`: the folder each instant's pictures go to: one track, `cam0`... with
  `--sync`, or the views of a 360 capture. Every track gets the same stems.
  An unsynced multi-track run refuses `--plan-out` and `--progress`.
- `ends`: the adaptive plan, the source frame each window closes at. Absent
  for a fixed rate.
- `stems`: present only when the stems follow from the plan alone: a plan,
  `--keep` 0 or 1, no `--blur-reject` / `--similar`, no `--start` / `--end` /
  `--max-frames`. Otherwise the written frame is the sharpest of its window and
  is known only once decoded, so the stems arrive in the progress lines. A
  video that ends early can still write fewer than listed.

## `--progress <file>`

Each frame (and mask, when extraction masks) is written as `.<name>.tmp` and
renamed, so `*.jpg` never matches half a file; such leftovers of an interrupted
run are removed from the output folders at start. The file gets JSON lines:

```json
{"written":2,"stems":["00012","00027"]}
{"written":3,"stems":["00041"],"done":true}
```

`written` is a low watermark: the first `written` stems in time order are on
disk in every track. `stems` lists the ones this line adds; their
concatenation is the stem list. Files land out of order (WriterPool), so a
stem waits for every one before it. At most one line each 250 ms; the last line
carries `"done":true` and is written only when the run succeeded with no write
failures. A failed write stops the watermark for good. A reader polling the
file skips a last line that has no newline yet: it is still being appended.
If `--plan-out` cannot be written the run stops before the first frame.

## `--plan-in <file>`

Reuses a `--plan-out` file's `ends` and skips the motion pass, so a resumed run
picks the same windows (and, the decode being deterministic, the same frames).
`--keep` must equal the file's. `--start` / `--end` clip the plan as usual.
A frame file already on disk is left as it is and not rewritten, and counts
towards `written`: the watermark is over the plan, not over this process's
writes. A truncated file left by a run without `--progress` is kept too.

The stats table on stdout ends with `decoder: vulkan` or `decoder: ffmpeg`,
English in every language, naming the decoder that ran.

## `sam track --follow [--prime <file>]`

Frame paths come from stdin, one per line, instead of `--frames`; EOF ends the
run. Masks are `<out>/<stem>.png` (written via `.<name>.tmp` and renamed; leftovers
are removed from `<out>` at start, so give each process its own `--out`) and stdout
gets `done <stem>` once each is on disk, or `failed <stem>` when the frame did
not load or the mask did not write (exit code 1 then). `--prime` is tracked as
frame 0 and not written, so the tracker state equals a folder run over
`[prime, stdin...]`; `--at-frame` counts in that sequence. `--max-frames`
counts stdin frames.
