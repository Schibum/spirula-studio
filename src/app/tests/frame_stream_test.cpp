// frame_stream -- the plan file and the low watermark `sam extract --plan-out
// --progress` hand a masking process (app/FrameStream.h). A watermark ahead of
// the disk feeds SAM half a JPEG, so its corners are asserted: files landing
// out of order, a stem whose last file is not submitted yet, a failed write.

#include "app/FrameStream.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "ok  " : "BAD ", what.c_str());
    if (!ok) g_failures++;
}

std::string joined(const std::vector<app::FrameLanded>& v) {
    std::string s;
    for (const app::FrameLanded& x : v)
        s += (s.empty() ? "" : " ") + (x.rejected ? "r" + std::to_string(x.window) : x.stem);
    return s;
}

void watermark() {
    std::vector<app::FrameLanded> seen;
    app::StemWatermark w([&](const std::vector<app::FrameLanded>& s) {
        seen.insert(seen.end(), s.begin(), s.end());
    });
    const int64_t a = w.open({"00002", 0, false});
    auto a0 = w.add(a);
    auto a1 = w.add(a);
    const int64_t b = w.open({"00005", 1, false});
    auto b0 = w.add(b);
    auto b1 = w.add(b);
    w.seal(b);
    b0(true);
    b1(true);
    expect(seen.empty(), "a later stem complete before an earlier one is held back");
    a0(true);
    a1(true);
    expect(seen.empty(), "an unsealed stem is held back with every file landed");
    w.seal(a);
    expect(joined(seen) == "00002 00005", "sealing releases it and the one behind it");

    w.seal(w.open({"", 2, true}));
    w.seal(w.open({"00008", 3, false}));
    expect(joined(seen) == "00002 00005 r2 00008",
           "a rejected window and a frame already on disk land in order");

    const int64_t c = w.open({"00009", 4, false});
    auto c0 = w.add(c);
    w.seal(c);
    const int64_t d = w.open({"00012", 5, false});
    auto d0 = w.add(d);
    w.seal(d);
    d0(true);
    c0(false);
    expect(joined(seen) == "00002 00005 r2 00008" && w.stuck(), "a failed file stops the watermark");
    w.seal(w.open({"00015", 6, false}));
    expect(joined(seen) == "00002 00005 r2 00008", "and it stays stopped");
}

void windows() {
    expect(app::frame_window({}, 15, 2) == 0 && app::frame_window({}, 15, 14) == 0 &&
               app::frame_window({}, 15, 16) == 1,
           "a fixed rate's window is its step");
    const std::vector<int64_t> ends = {14, 29, 41};
    expect(app::frame_window(ends, 15, 12) == 0 && app::frame_window(ends, 15, 14) == 0 &&
               app::frame_window(ends, 15, 28) == 1 && app::frame_window(ends, 15, 41) == 2,
           "a plan's window is the entry it closes at");
}

void plan_file() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "frame_stream_test_plan.json").string();
    app::FramePlan p;
    p.tracks = {"/out/cam0", "/out/cam\"1"};
    p.ends = {14, 29, 41};
    p.keep = 3;
    p.skip = 15;
    p.decoder = "ffmpeg";
    p.input = "/v/clip.insv";
    p.input_size = 123456789012;
    p.input_mtime = "1795811073876123456";
    p.options = {{"quality", "95"}, {"blur_reject", "0"}};
    std::string err;
    expect(app::write_text_atomic(path, app::frame_plan_json(p), err), "plan written");
    expect(!std::filesystem::exists(path + ".tmp"), "no temporary left beside it");
    app::FramePlan q;
    expect(app::read_frame_plan(path, q, err), "plan read back");
    expect(q.tracks == p.tracks && q.ends == p.ends && q.keep == 3 && q.skip == 15 &&
               q.decoder == "ffmpeg" && q.input == p.input && q.input_size == p.input_size &&
               q.input_mtime == p.input_mtime && q.options == p.options,
           "plan round trip, the mtime past 2^53 exact");
    expect(app::frame_plan_mismatch(q, p).empty(), "the same plan matches");
    app::FramePlan r = p;
    r.decoder = "vulkan";
    expect(!app::frame_plan_mismatch(q, r).empty(), "another decoder is refused");
    r = p;
    r.options[1].second = "4";
    expect(!app::frame_plan_mismatch(q, r).empty(), "another pick option is refused");
    r = p;
    r.input_mtime = "1";
    expect(!app::frame_plan_mismatch(q, r).empty(), "another input is refused");
    std::filesystem::remove(path);
    expect(!app::read_frame_plan(path, q, err), "a missing plan is an error");

    expect(app::frame_progress_line(2, {{"00014", 0, false}, {"", 1, true}}) ==
               "{\"written\":2,\"frames\":[{\"stem\":\"00014\",\"window\":0},"
               "{\"window\":1,\"rejected\":true}]}\n",
           "progress line");
    expect(app::frame_progress_line(2, {}, ",\"done\":false,\"error\":\"x\"") ==
               "{\"written\":2,\"frames\":[],\"done\":false,\"error\":\"x\"}\n",
           "final progress line");
}

}  // namespace

int main() {
    watermark();
    windows();
    plan_file();
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
}
