#pragma once

// SS_SFM_TRACE_DIR=<dir>: a snapshot of the mapper's model after every event, for offline
// study of where a model's scale departs (cameras in full, points of an image window only).
// Read-only on the model; unset, nothing is computed. Format: src/sfm/README.md, "Mapper trace".

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/Env.h"
#include "sfm/core/Model.h"

namespace sfm {

class MapTrace {
public:
    // The tracer for the current SS_SFM_TRACE_DIR, or nullptr; a new directory starts over.
    static MapTrace* attach() {
        const char* dir = spirula::env("SFM_TRACE_DIR");
        if (!dir || !*dir) return nullptr;
        static MapTrace t;
        std::lock_guard<std::mutex> lk(t.mu_);
        if (t.dir_ != dir) t.open(dir);
        return &t;
    }

    uint32_t newMapper(const std::vector<std::string>& names) {
        std::lock_guard<std::mutex> lk(mu_);
        if (!names_written_) {
            if (FILE* f = std::fopen((dir_ + "/names.tsv").c_str(), "w")) {
                for (size_t i = 0; i < names.size(); i++) std::fprintf(f, "%zu\t%s\n", i, names[i].c_str());
                std::fclose(f);
            }
            names_written_ = true;
        }
        return mappers_++;
    }

    bool inWindow(uint32_t img) const { return win_lo_ <= win_hi_ && img >= win_lo_ && img <= win_hi_; }

    void event(const char* type, uint32_t mapper, const char* phase, const Reconstruction& rec,
               const std::string& fields) {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<uint32_t> reg;
        for (const auto& kv : rec.images)
            if (kv.second.registered) reg.push_back(kv.first);
        std::vector<Vec3> c(reg.size());
        for (size_t i = 0; i < reg.size(); i++) c[i] = cameraCenter(rec.images.at(reg[i]).pose);

        size_t obs = 0, win_pts = 0;
        std::vector<const Point3D*> wp;
        std::vector<uint16_t> wn;
        for (const auto& kv : rec.points3D) {
            obs += kv.second.track.size();
            if (win_lo_ > win_hi_) continue;
            uint16_t n = 0;
            for (const TrackElement& e : kv.second.track) n += inWindow(e.image_id) ? 1 : 0;
            if (!n) continue;
            win_pts++;
            wp.push_back(&kv.second);
            wn.push_back(n);
        }
        const size_t take = std::min(wp.size(), max_points_);

        const std::string bin = dir_ + "/" + fileName(index_);
        if (FILE* f = std::fopen(bin.c_str(), "wb")) {
            char head[4] = {'S', 'S', 'M', 'T'};
            char t16[16] = {};
            std::snprintf(t16, sizeof t16, "%s", type);
            put(f, head, 4);
            putU32(f, 1);
            putU32(f, index_);
            putU32(f, mapper);
            put(f, t16, 16);
            putU32(f, (uint32_t)reg.size());
            for (size_t i = 0; i < reg.size(); i++) {
                const Image& im = rec.images.at(reg[i]);
                putU32(f, reg[i]);
                putU32(f, im.camera_id);
                const double cc[3] = {c[i].x, c[i].y, c[i].z};
                put(f, cc, sizeof cc);
                put(f, im.pose.R.data(), 9 * sizeof(double));
            }
            putU32(f, (uint32_t)rec.cameras.size());
            for (const auto& kv : rec.cameras) {
                const Camera& k = kv.second;
                putU32(f, kv.first);
                putU32(f, (uint32_t)k.model);
                putU32(f, (uint32_t)k.width);
                putU32(f, (uint32_t)k.height);
                const double p[14] = {k.fx, k.fy, k.cx, k.cy, k.k1, k.k2, k.p1,
                                      k.p2, k.k3, k.k4, k.k5, k.k6, k.sx1, k.sy1};
                put(f, p, sizeof p);
            }
            putU64(f, rec.points3D.size());
            putU64(f, obs);
            putU32(f, win_lo_);
            putU32(f, win_hi_);
            putU64(f, win_pts);
            putU32(f, (uint32_t)take);
            for (size_t k = 0; k < take; k++) {
                // Strided over the point-id order, so a sample covers the whole window.
                const size_t i = take == wp.size() ? k : k * wp.size() / take;
                const Point3D& p = *wp[i];
                const float xyz[3] = {(float)p.xyz.x, (float)p.xyz.y, (float)p.xyz.z};
                const uint16_t tl = (uint16_t)std::min<size_t>(p.track.size(), 65535);
                put(f, xyz, sizeof xyz);
                put(f, &tl, 2);
                put(f, &wn[i], 2);
            }
            std::fclose(f);
        }

        std::string added, removed;
        std::set<uint32_t> now(reg.begin(), reg.end());
        for (uint32_t id : now)
            if (!last_reg_.count(id)) appendId(added, id);
        for (uint32_t id : last_reg_)
            if (!now.count(id)) appendId(removed, id);
        last_reg_.swap(now);

        const double sw = medianStep(reg, c, true), sa = medianStep(reg, c, false);
        const Camera* k0 = rec.cameras.empty() ? nullptr : &rec.cameras.begin()->second;
        if (log_) {
            std::fprintf(log_,
                         "%06u %s mapper=%u phase=%s reg=%zu pts=%zu obs=%zu f=%.2f cx=%.2f "
                         "cy=%.2f k1=%.5f k2=%.5f step_win=%.6g step_all=%.6g win_over_all=%.4f "
                         "win_pts=%zu%s%s%s%s%s%s\n",
                         index_, type, mapper, phase, reg.size(), rec.points3D.size(), obs,
                         k0 ? k0->focal() : 0.0, k0 ? k0->cx : 0.0, k0 ? k0->cy : 0.0,
                         k0 ? k0->k1 : 0.0, k0 ? k0->k2 : 0.0, sw, sa, sa > 0 ? sw / sa : 0.0,
                         win_pts, fields.empty() ? "" : " ", fields.c_str(),
                         added.empty() ? "" : " added=", added.c_str(),
                         removed.empty() ? "" : " removed=", removed.c_str());
            std::fflush(log_);
        }
        index_++;
    }

    static std::string fields(const char* fmt, ...) {
        char buf[512];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        return buf;
    }

    static std::string fileName(uint32_t index) {
        char nm[32];
        std::snprintf(nm, sizeof nm, "e%06u.bin", index);
        return nm;
    }

private:
    MapTrace() = default;

    void open(const std::string& dir) {
        if (log_) std::fclose(log_);
        dir_ = dir;
        std::filesystem::create_directories(dir_);
        log_ = std::fopen((dir_ + "/events.log").c_str(), "w");
        if (!log_) throw std::runtime_error("SS_SFM_TRACE_DIR: cannot write " + dir_ + "/events.log");
        index_ = mappers_ = 0;
        names_written_ = false;
        last_reg_.clear();
        win_lo_ = 1;
        win_hi_ = 0;
        if (const char* w = spirula::env("SFM_TRACE_IMAGES")) {
            unsigned lo = 0, hi = 0;
            char tail = 0;
            if (std::sscanf(w, "%u-%u%c", &lo, &hi, &tail) != 2 || lo > hi)
                throw std::invalid_argument("invalid SS_SFM_TRACE_IMAGES (lo-hi)");
            win_lo_ = lo;
            win_hi_ = hi;
        }
        max_points_ = 2000;
        if (const char* p = spirula::env("SFM_TRACE_POINTS")) {
            char* end = nullptr;
            const long n = std::strtol(p, &end, 10);
            if (end == p || *end || n < 0 || n > 10000000)
                throw std::invalid_argument("invalid SS_SFM_TRACE_POINTS");
            max_points_ = (size_t)n;
        }
    }

    // Median distance between consecutive registered images (in id order) over their id gap,
    // within the window (`win`) or over the whole model.
    double medianStep(const std::vector<uint32_t>& reg, const std::vector<Vec3>& c, bool win) const {
        std::vector<double> s;
        for (size_t i = 1; i < reg.size(); i++) {
            if (win && !(inWindow(reg[i - 1]) && inWindow(reg[i]))) continue;
            s.push_back((c[i] - c[i - 1]).norm() / (double)(reg[i] - reg[i - 1]));
        }
        if (s.empty()) return 0;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        return s[s.size() / 2];
    }

    static void appendId(std::string& s, uint32_t id) {
        if (!s.empty()) s += ',';
        s += std::to_string(id);
    }
    static void put(FILE* f, const void* p, size_t n) { std::fwrite(p, 1, n, f); }
    static void putU32(FILE* f, uint32_t v) { put(f, &v, 4); }
    static void putU64(FILE* f, uint64_t v) { put(f, &v, 8); }

    std::mutex mu_;
    std::string dir_;
    FILE* log_ = nullptr;
    uint32_t index_ = 0, mappers_ = 0;
    bool names_written_ = false;
    std::set<uint32_t> last_reg_;
    uint32_t win_lo_ = 1, win_hi_ = 0;
    size_t max_points_ = 2000;
};

} // namespace sfm
