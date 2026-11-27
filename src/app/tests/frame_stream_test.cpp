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

std::string joined(const std::vector<std::string>& v) {
    std::string s;
    for (const std::string& x : v) s += (s.empty() ? "" : " ") + x;
    return s;
}

void watermark() {
    std::vector<std::string> seen;
    app::StemWatermark w([&](const std::vector<std::string>& s) {
        seen.insert(seen.end(), s.begin(), s.end());
    });
    const int64_t a = w.open("00002");
    auto a0 = w.add(a);
    auto a1 = w.add(a);
    const int64_t b = w.open("00005");
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

    const int64_t c = w.open("00009");
    auto c0 = w.add(c);
    w.seal(c);
    const int64_t d = w.open("00012");
    auto d0 = w.add(d);
    w.seal(d);
    d0(true);
    c0(false);
    expect(joined(seen) == "00002 00005", "a failed file stops the watermark");
    const int64_t e = w.open("00015");
    w.seal(e);
    expect(joined(seen) == "00002 00005", "and it stays stopped");
}

void plan_file() {
    const std::string path =
        (std::filesystem::temp_directory_path() / "frame_stream_test_plan.json").string();
    app::FramePlan p;
    p.tracks = {"out/cam0", "out/cam\"1"};
    p.ends = {14, 29, 41};
    p.keep = 3;
    p.skip = 15;
    std::string err;
    expect(app::write_text_atomic(path, app::frame_plan_json(p), err), "plan written");
    expect(!std::filesystem::exists(path + ".tmp"), "no temporary left beside it");
    app::FramePlan q;
    expect(app::read_frame_plan(path, q, err), "plan read back");
    expect(q.tracks == p.tracks && q.ends == p.ends && q.keep == 3 && q.skip == 15 &&
               q.stems.empty(),
           "plan round trip, stems absent when unknown");
    p.stems = {app::frame_stem(14), app::frame_stem(29)};
    app::write_text_atomic(path, app::frame_plan_json(p), err);
    expect(app::read_frame_plan(path, q, err) && joined(q.stems) == "00014 00029",
           "stems round trip");
    std::filesystem::remove(path);
    expect(!app::read_frame_plan(path, q, err), "a missing plan is an error");

    expect(app::frame_progress_line(2, {"00014", "00029"}, false) ==
               "{\"written\":2,\"stems\":[\"00014\",\"00029\"]}\n",
           "progress line");
    expect(app::frame_progress_line(2, {}, true) == "{\"written\":2,\"stems\":[],\"done\":true}\n",
           "final progress line");
}

}  // namespace

int main() {
    watermark();
    plan_file();
    std::printf("%s\n", g_failures ? "FAILED" : "all passed");
    return g_failures ? 1 : 0;
}
