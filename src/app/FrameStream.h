#pragma once

// What a process masking frames while `spirula sam extract` still writes them
// reads: the plan file, the progress lines, and the low watermark behind them
// (docs/notes/extract-streaming.md has the protocol).

#include "data/Json.h"
#include "data/JsonWrite.h"

#include <cstdint>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace app {

// Frames are named by source frame index, whichever decoder ran.
inline std::string frame_stem(int64_t index) {
    char stem[32];
    std::snprintf(stem, sizeof stem, "%05lld", (long long)index);
    return stem;
}

// A run's frames as known before the first one is written.
struct FramePlan {
    std::vector<std::string> tracks;   // the folder each instant's pictures go to
    std::vector<int64_t> ends;         // the motion plan; empty for a fixed rate
    int keep = 0, skip = 1;
    // Empty unless the stems follow from the plan alone (one frame a window,
    // no PickRule, no clip, no cap). The video ending early can still cut it.
    std::vector<std::string> stems;
};

inline std::string frame_plan_json(const FramePlan& p) {
    auto strings = [](const std::vector<std::string>& v) {
        std::string s = "[";
        for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + json_quote(v[i]);
        return s + "]";
    };
    std::string s = "{\"version\":1,\"tracks\":" + strings(p.tracks);
    if (!p.stems.empty()) s += ",\"stems\":" + strings(p.stems);
    s += ",\"keep\":" + std::to_string(p.keep) + ",\"skip\":" + std::to_string(p.skip);
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
    out.keep = (int)(v.find("keep") ? v.find("keep")->as_int(-1) : -1);
    if (const JsonValue* s = v.find("skip")) out.skip = (int)s->as_int(1);
    if (const JsonValue* t = v.find("tracks"))
        for (const JsonValue& x : t->arr) out.tracks.push_back(x.as_string());
    if (const JsonValue* s = v.find("stems"))
        for (const JsonValue& x : s->arr) out.stems.push_back(x.as_string());
    if (const JsonValue* e = v.find("ends"))
        for (const JsonValue& x : e->arr) out.ends.push_back(x.as_int());
    return true;
}

// One line of the progress file: `written` stems on disk so far, `stems` the
// ones this line adds.
inline std::string frame_progress_line(int64_t written,
                                       const std::vector<std::string>& stems, bool done) {
    std::string s = "{\"written\":" + std::to_string((long long)written) + ",\"stems\":[";
    for (size_t i = 0; i < stems.size(); i++) s += (i ? "," : "") + json_quote(stems[i]);
    return s + (done ? "],\"done\":true}\n" : "]}\n");
}

// The low watermark over files that land out of order: a stem is reported
// once every file of it and of every stem before it is on disk. A file that
// failed stops it for good.
class StemWatermark {
public:
    using Sink = std::function<void(const std::vector<std::string>&)>;
    explicit StemWatermark(Sink sink) : sink_(std::move(sink)) {}

    // The next stem in time order; its files are add()ed, then it is seal()ed.
    int64_t open(const std::string& stem) {
        std::lock_guard<std::mutex> lk(mu_);
        q_.push_back({stem, 0, false});
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

private:
    struct Entry {
        std::string stem;
        int left;
        bool sealed;
    };
    Entry& at(int64_t seq) { return q_[(size_t)(seq - base_)]; }
    // Under mu_, so the sink sees stems in order.
    void drain() {
        if (stuck_) return;
        std::vector<std::string> out;
        while (!q_.empty() && q_.front().sealed && q_.front().left == 0) {
            out.push_back(std::move(q_.front().stem));
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
