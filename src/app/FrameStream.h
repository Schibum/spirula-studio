#pragma once

// What a process masking frames while `spirula sam extract` still writes them
// reads: the plan file, the progress lines, and the low watermark behind them
// (docs/notes/extract-streaming.md has the protocol).

#include "data/Json.h"
#include "data/JsonWrite.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace app {

// Frames are named by source frame index, whichever decoder ran.
inline std::string frame_stem(int64_t index) {
    char stem[32];
    std::snprintf(stem, sizeof stem, "%05lld", (long long)index);
    return stem;
}

// A run's frames as known before the first one is written, and what has to
// match for --plan-in to pick the same ones.
struct FramePlan {
    std::vector<std::string> tracks;   // absolute: the folder each instant's pictures go to
    std::vector<int64_t> ends;         // the motion plan; empty for a fixed rate
    int keep = 0, skip = 1;
    std::string decoder;               // "vulkan" or "ffmpeg"
    std::string input;                 // absolute
    int64_t input_size = -1;
    std::string input_mtime;           // file_time_type ticks, as text: past 2^53
    std::vector<std::pair<std::string, std::string>> options;   // what else picks or shapes a frame
};

// The window a chosen frame closes: its plan entry, or its rate step.
inline int64_t frame_window(const std::vector<int64_t>& ends, int skip, int64_t index) {
    if (ends.empty()) return index / std::max(skip, 1);
    return (int64_t)(std::lower_bound(ends.begin(), ends.end(), index) - ends.begin());
}

inline std::string frame_plan_json(const FramePlan& p) {
    std::string s = "{\"version\":1,\"tracks\":[";
    for (size_t i = 0; i < p.tracks.size(); i++) s += (i ? "," : "") + json_quote(p.tracks[i]);
    s += "],\"keep\":" + std::to_string(p.keep) + ",\"skip\":" + std::to_string(p.skip);
    s += ",\"decoder\":" + json_quote(p.decoder);
    s += ",\"input\":{\"path\":" + json_quote(p.input) +
         ",\"size\":" + std::to_string((long long)p.input_size) +
         ",\"mtime\":" + json_quote(p.input_mtime) + "}";
    s += ",\"options\":{";
    for (size_t i = 0; i < p.options.size(); i++)
        s += (i ? "," : "") + json_quote(p.options[i].first) + ":" + json_quote(p.options[i].second);
    s += "}";
    if (!p.ends.empty()) {
        s += ",\"ends\":[";
        for (size_t i = 0; i < p.ends.size(); i++)
            s += (i ? "," : "") + std::to_string((long long)p.ends[i]);
        s += "]";
    }
    return s + "}\n";
}

// Written beside `path` and renamed over it, so a reader sees all or nothing.
inline bool write_text_atomic(const std::string& path, const std::string& text,
                              std::string& error) {
    const std::string tmp = path + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    bool ok = f && std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (f) ok = std::fclose(f) == 0 && ok;
    std::error_code ec;
    if (ok) std::filesystem::rename(tmp, path, ec);
    if (!ok || ec) {
        std::filesystem::remove(tmp, ec);
        error = "could not write " + path;
        return false;
    }
    return true;
}

inline bool read_frame_plan(const std::string& path, FramePlan& out, std::string& error) {
    JsonValue v;
    try {
        v = json_parse_file(path);
    } catch (const std::exception& e) {
        error = path + ": " + e.what();
        return false;
    }
    const JsonValue* ver = v.find("version");
    if (!ver || ver->as_int() != 1) {
        error = path + ": not a version 1 frame plan";
        return false;
    }
    out = FramePlan{};
    auto num = [&](const JsonValue* o, const char* key, int64_t def) {
        const JsonValue* x = o ? o->find(key) : nullptr;
        return x ? x->as_int(def) : def;
    };
    auto str = [&](const JsonValue* o, const char* key) {
        const JsonValue* x = o ? o->find(key) : nullptr;
        return x ? x->as_string() : std::string();
    };
    out.keep = (int)num(&v, "keep", -1);
    out.skip = (int)num(&v, "skip", 1);
    out.decoder = str(&v, "decoder");
    const JsonValue* in = v.find("input");
    out.input = str(in, "path");
    out.input_size = num(in, "size", -1);
    out.input_mtime = str(in, "mtime");
    if (const JsonValue* t = v.find("tracks"))
        for (const JsonValue& x : t->arr) out.tracks.push_back(x.as_string());
    if (const JsonValue* o = v.find("options"))
        for (const auto& [k, x] : o->obj) out.options.push_back({k, x.as_string()});
    if (const JsonValue* e = v.find("ends"))
        for (const JsonValue& x : e->arr) out.ends.push_back(x.as_int());
    return true;
}

// Why `now` would not pick the frames `was` did; empty when it would. The
// input is matched by size and time, so a moved file still resumes.
inline std::string frame_plan_mismatch(const FramePlan& was, const FramePlan& now) {
    if (was.input_size != now.input_size || was.input_mtime != now.input_mtime)
        return "made from another input file";
    if (was.decoder != now.decoder) return "made with the " + was.decoder + " decoder";
    if (was.keep != now.keep || was.skip != now.skip) return "made with another --keep or --skip";
    if (was.tracks != now.tracks) return "made for other output folders";
    for (const auto& [k, v] : was.options) {
        std::string cur;
        for (const auto& [k2, v2] : now.options)
            if (k2 == k) cur = v2;
        if (cur != v) return "made with " + k + " " + v + ", not " + cur;
    }
    return {};
}

// One entry of a progress line: a frame on disk in every track, or a window
// PickRule left empty.
struct FrameLanded {
    std::string stem;
    int64_t window = -1;
    bool rejected = false;
};

inline std::string frame_landed_json(const FrameLanded& f) {
    if (f.rejected) return "{\"window\":" + std::to_string((long long)f.window) + ",\"rejected\":true}";
    std::string s = "{\"stem\":" + json_quote(f.stem);
    if (f.window >= 0) s += ",\"window\":" + std::to_string((long long)f.window);
    return s + "}";
}

// One line of the progress file. `written` counts frames (not rejections) on
// disk so far; `tail` is "" or the final line's ,"done":... fields.
inline std::string frame_progress_line(int64_t written, const std::vector<FrameLanded>& frames,
                                       const std::string& tail = {}) {
    std::string s = "{\"written\":" + std::to_string((long long)written) + ",\"frames\":[";
    for (size_t i = 0; i < frames.size(); i++) s += (i ? "," : "") + frame_landed_json(frames[i]);
    return s + "]" + tail + "}\n";
}

// The low watermark over files that land out of order: an entry is reported
// once every file of it and of every entry before it is on disk. A file that
// failed stops it for good.
class StemWatermark {
public:
    using Sink = std::function<void(const std::vector<FrameLanded>&)>;
    explicit StemWatermark(Sink sink) : sink_(std::move(sink)) {}

    // The next entry in time order; its files are add()ed, then it is seal()ed.
    int64_t open(FrameLanded f) {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back({std::move(f), 0, false});
        return base_ + (int64_t)q_.size() - 1;
    }
    // A file of `seq`; call the result once when it has landed or failed.
    std::function<void(bool)> add(int64_t seq) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            at(seq).left++;
        }
        return [this, seq](bool ok) {
            std::lock_guard<std::mutex> lk(mu_);
            at(seq).left--;
            if (!ok) stuck_ = true;
            drain();
        };
    }
    void seal(int64_t seq) {
        std::lock_guard<std::mutex> lk(mu_);
        at(seq).sealed = true;
        drain();
    }
    bool stuck() {
        std::lock_guard<std::mutex> lk(mu_);
        return stuck_;
    }

private:
    struct Entry {
        FrameLanded frame;
        int left;
        bool sealed;
    };
    Entry& at(int64_t seq) { return q_[(size_t)(seq - base_)]; }
    // Under mu_, so the sink sees entries in order; it must not block.
    void drain() {
        if (stuck_) return;
        std::vector<FrameLanded> out;
        while (!q_.empty() && q_.front().sealed && q_.front().left == 0) {
            out.push_back(std::move(q_.front().frame));
            q_.pop_front();
            ++base_;
        }
        if (!out.empty() && sink_) sink_(out);
    }

    Sink sink_;
    std::mutex mu_;
    std::deque<Entry> q_;
    int64_t base_ = 0;
    bool stuck_ = false;
};

}  // namespace app
