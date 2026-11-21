#pragma once
// When each source frame of a video was shown: the per-frame time table a
// capture is timed by, from a file the caller wrote (a manifest capture's
// `frame_times`) or from the video's own sample table (Telemetry's
// video_frame_times). A frame's time is looked up, never computed from a
// rate: a clip with a hole, or a variable frame rate, has no rate that fits.
//
// The file:
//   {"version": 1, "video": "clip.mp4", "frames": 1574, "times": [0.0, 0.033333, ...]}
// times[n] is the presentation time in seconds of source frame n (the number
// in the frame's file name), strictly increasing.

#include "data/Yaml.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace sfm {

// Why `times` cannot time frames, or "" when it can: at least one entry,
// every one finite and later than the one before.
inline std::string frameTimesProblem(const std::vector<double>& times) {
    if (times.empty()) return "no frame times";
    for (size_t i = 0; i < times.size(); i++) {
        if (!std::isfinite(times[i])) return "frame " + std::to_string(i) + ": not a time";
        if (i > 0 && !(times[i] > times[i - 1]))
            return "frame " + std::to_string(i) + ": not later than frame " + std::to_string(i - 1);
    }
    return {};
}

// Reads a frame-time file. False with `error` (the file and, for a bad
// entry, its index) when it is missing, malformed or not strictly increasing.
inline bool frameTimesRead(const std::string& path, std::vector<double>& times,
                           std::string& error) {
    times.clear();
    JsonValue root;
    try {
        root = yaml_parse_file(path);
    } catch (const std::exception& e) {
        error = path + ": " + e.what();
        return false;
    }
    const JsonValue* version = root.find("version");
    if (!version || version->type != JsonValue::Type::Number || version->num != 1) {
        error = path + ": version: expected 1";
        return false;
    }
    const JsonValue* list = root.find("times");
    if (!list || !list->is_array()) {
        error = path + ": times: expected a list of seconds";
        return false;
    }
    times.reserve(list->arr.size());
    for (size_t i = 0; i < list->arr.size(); i++) {
        if (list->arr[i].type != JsonValue::Type::Number) {
            error = path + ": times: frame " + std::to_string(i) + ": expected a number";
            times.clear();
            return false;
        }
        times.push_back(list->arr[i].num);
    }
    if (const JsonValue* n = root.find("frames"))
        if (n->type != JsonValue::Type::Number || (size_t)n->num != times.size()) {
            error = path + ": frames: does not match the " + std::to_string(times.size()) +
                    " times listed";
            times.clear();
            return false;
        }
    const std::string why = frameTimesProblem(times);
    if (!why.empty()) {
        error = path + ": times: " + why;
        times.clear();
        return false;
    }
    return true;
}

// The largest distance, in seconds, of any frame from the constant rate that
// fits the table best (least squares over every frame). What timing the
// frames by one rate would get wrong; 0 for fewer than three frames.
inline double frameTimesOffConstantRate(const std::vector<double>& times) {
    const size_t n = times.size();
    if (n < 3) return 0.0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; i++) {
        const double x = (double)i, y = times[i] - times[0];
        sx += x; sy += y; sxx += x * x; sxy += x * y;
    }
    const double den = (double)n * sxx - sx * sx;
    if (!(den > 0)) return 0.0;
    const double slope = ((double)n * sxy - sx * sy) / den;
    const double icpt = (sy - slope * sx) / (double)n;
    double worst = 0;
    for (size_t i = 0; i < n; i++)
        worst = std::max(worst, std::fabs(times[i] - times[0] - (slope * (double)i + icpt)));
    return worst;
}

}  // namespace sfm
