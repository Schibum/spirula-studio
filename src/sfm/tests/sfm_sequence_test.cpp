// Sequences (sfm/core/Sequence.h, D79): the table, its window pairs, and a
// synthetic walk past a duplicated structure through the mapper.
//
// Prints PASS/FAIL and returns 0/1. See docs/testing.md.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

#include "sfm/core/Sequence.h"
#include "sfm/feature/Pairing.h"
#include "sfm/map/Mapper.h"
#include "sfm/tests/TestMain.h"

using namespace sfm;

static int fails = 0;

static void check(bool ok, const char* what) {
    if (!ok) {
        printf("  FAIL: %s\n", what);
        fails++;
    }
}

static void testTable() {
    SequenceDef d;
    check(parseSequenceArg("cam0,cam1/", d).empty() && d.members.size() == 2 &&
              d.members[1] == "cam1",
          "parse: two members, trailing slash dropped");
    check(parseSequenceArg(".", d).empty() && d.members.size() == 1 && d.members[0].empty(),
          "parse: '.' is the image directory");

    const std::vector<std::string> names = {"cam0/00010", "cam0/00020", "cam0/00030",
                                            "cam1/00010", "cam1/00030", "other/a",
                                            "other/b"};
    std::vector<SequenceDef> defs(2);
    defs[0].members = {"cam0", "cam1"};
    defs[1].members = {"other"};
    SequenceTable t = buildSequenceTable(names, defs);
    check(t.length.size() == 2 && t.length[0] == 3 && t.length[1] == 2, "two sequences sized");
    check(t.pos[0] == 0 && t.pos[1] == 1 && t.pos[2] == 2, "positions follow the stems");
    check(t.pos[3] == 0 && t.pos[4] == 2 && t.member[3] == 1, "a rig-mate shares its position");
    check(t.distance(0, 4) == 2 && t.nearby(0, 4, 2) && !t.nearby(0, 4, 1), "distance across lenses");
    check(t.distance(0, 5) == INT_MAX && t.seq[5] == 1, "no distance across sequences");
    check(t.has(6) && t.sequenceOf(6) == 1, "membership");

    const std::vector<uint32_t> local = {kNoImage, 0, 1, kNoImage, 2, kNoImage, kNoImage};
    SequenceTable sub = t.subset(local, 3);
    check(sub.pos[0] == 1 && sub.pos[1] == 2 && sub.pos[2] == 2 && sub.distance(0, 2) == 1,
          "subset keeps positions");

    const auto pairs = sequenceWindowPairs(t, 1, false);
    auto has = [&](uint32_t a, uint32_t b) {
        return std::binary_search(pairs.begin(), pairs.end(), std::make_pair(a, b));
    };
    check(has(0, 1) && has(1, 2) && has(3, 4) && has(5, 6) && !has(2, 3) && !has(0, 3),
          "window pairs run per member and never cross");

    bool threw = false;
    try {
        std::vector<SequenceDef> bad(1);
        bad[0].members = {"cam0", "nothing"};
        buildSequenceTable(names, bad);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "a member no image matches is an error");
    threw = false;
    try {
        std::vector<SequenceDef> twice(2);
        twice[0].members = {"cam0"};
        twice[1].members = {""};
        buildSequenceTable(names, twice);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "an image in two sequences is an error");
}

// A walk along a wall whose far end repeats its near end feature for feature:
// room A, a corridor of its own, and room B a copy of A. Every camera of B
// matches every camera of A that saw the same part of the pattern.
struct Walk {
    int W = 1280, H = 960;
    double focal = 800;
    std::vector<Pose> gt;
    std::vector<FeatureSet> feats;
    MatchesDatabase db;
    std::vector<std::string> names;
};

static Pose lookAtX(const Vec3& C) {
    // Camera z along world +x, camera x along world +z, camera y along world -y.
    Mat3 R = {0, 0, 1, 0, -1, 0, 1, 0, 0};
    Vec3 t = mul(R, C);
    return {R, {-t.x, -t.y, -t.z}};
}

static Walk makeWalk() {
    Walk w;
    const int NA = 300, NC = 300;
    Camera K = Camera::defaultFor(1, w.W, w.H, w.focal);
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> ux(4.5, 7.5), uy(-3.5, 3.5), uz(0, 9), uc(9, 21);
    std::normal_distribution<double> noise(0.0, 0.4);
    // ids 0..NA-1: room A and, 21 units on, its copy; ids NA..: the corridor.
    std::vector<Vec3> A(NA), C(NC);
    for (Vec3& p : A) p = {ux(rng), uy(rng), uz(rng)};
    for (Vec3& p : C) p = {ux(rng), uy(rng), uc(rng)};
    const double D = 21.0;
    const int NF = NA + NC;
    for (double t = -3.0; t <= 33.0 + 1e-9; t += 0.5) w.gt.push_back(lookAtX({0, 0, t}));
    const int M = (int)w.gt.size();
    w.feats.resize(M);
    w.names.resize(M);
    std::vector<std::vector<char>> vis(M, std::vector<char>(NF, 0));
    for (int c = 0; c < M; c++) {
        char nm[32];
        snprintf(nm, sizeof nm, "%03d", c);
        w.names[c] = nm;
        FeatureSet& fs = w.feats[c];
        fs.width = w.W;
        fs.height = w.H;
        fs.keypoints.assign(NF, {-1000, -1000, 2, 0, 0});
        auto place = [&](int id, const Vec3& X) {
            Vec3 pc = mul(w.gt[c].R, X) + w.gt[c].t;
            Vec2 px = K.project(pc);
            if (pc.z > 0.1 && px.x > 0 && px.x < w.W && px.y > 0 && px.y < w.H) {
                fs.keypoints[id] = {(float)(px.x + noise(rng)), (float)(px.y + noise(rng)), 2, 0, 0};
                vis[c][id] = 1;
            }
        };
        for (int p = 0; p < NA; p++) {
            place(p, A[p]);
            if (!vis[c][p]) place(p, A[p] + Vec3{0, 0, D});
        }
        for (int p = 0; p < NC; p++) place(NA + p, C[p]);
    }
    w.db.images.resize(M);
    for (int c = 0; c < M; c++) w.db.images[c] = {w.names[c], (uint32_t)NF};
    for (int i = 0; i < M; i++)
        for (int j = i + 1; j < M; j++) {
            TwoViewMatches tv;
            tv.image1 = i;
            tv.image2 = j;
            tv.config = (int)TwoViewConfig::Uncalibrated;
            for (int p = 0; p < NF; p++)
                if (vis[i][p] && vis[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p});
            if (tv.matches.size() >= 15) w.db.pairs.push_back(std::move(tv));
        }
    return w;
}

// Alignment-free shape check: every inter-camera distance of the model over
// the truth's is one scale. Returns the largest relative departure from the
// median ratio, or -1 with too few cameras.
static double shapeError(const Reconstruction& m, const Walk& w, uint32_t& registered) {
    std::vector<uint32_t> ids;
    for (const auto& kv : m.images)
        if (kv.second.registered) ids.push_back(kv.first);
    registered = (uint32_t)ids.size();
    if (ids.size() < 3) return -1;
    std::vector<double> ratio;
    for (size_t a = 0; a < ids.size(); a++)
        for (size_t b = a + 1; b < ids.size(); b++) {
            const double dm = (cameraCenter(m.images.at(ids[a]).pose) -
                               cameraCenter(m.images.at(ids[b]).pose)).norm();
            const double dg = (cameraCenter(w.gt[ids[a]]) - cameraCenter(w.gt[ids[b]])).norm();
            if (dg > 2.0) ratio.push_back(dm / dg);
        }
    std::sort(ratio.begin(), ratio.end());
    const double med = ratio[ratio.size() / 2];
    double worst = 0;
    for (double r : ratio) worst = std::max(worst, std::fabs(r / med - 1.0));
    return worst;
}

static void testMapperWalk(int device, bool verbose) {
    Walk w = makeWalk();
    MapperOptions opt;
    opt.verbose = verbose;
    opt.focal = w.focal;
    opt.device = device;
    std::vector<SequenceDef> defs(1);
    defs[0].members = {""};
    const SequenceTable seqs = buildSequenceTable(w.names, defs);

    Mapper plain(w.db, w.feats, opt);
    std::vector<Reconstruction> before = plain.run();
    uint32_t reg0 = 0;
    const double err0 = shapeError(before.front(), w, reg0);
    printf("sequence-walk: without sequences %u/%zu cameras in %zu model(s), shape error %.3f\n",
           reg0, w.gt.size(), before.size(), err0);

    Mapper mapper(w.db, w.feats, opt, {}, nullptr, &seqs);
    std::vector<Reconstruction> models = mapper.run();
    uint32_t reg = 0;
    const double err = shapeError(models.front(), w, reg);
    printf("sequence-walk: with the sequence %u/%zu cameras in %zu model(s), shape error %.3f\n",
           reg, w.gt.size(), models.size(), err);
    check(models.size() == 1, "the walk is one model");
    check(reg >= (uint32_t)(0.95 * w.gt.size()), "the walk registers");
    check(err >= 0 && err < 0.05, "the copy of the room is not folded onto the original");
}

#ifndef _WIN32
// The mapper's verbose log of one run, from stdout and stderr, with every number that is a
// time (hh:mm:ss.mmm, "N s", "N ms") blanked.
static std::vector<std::string> mapperLog(const Walk& w, const MapperOptions& opt,
                                          const SequenceTable& seqs, const std::string& path,
                                          std::vector<Reconstruction>& out) {
    fflush(stdout);
    fflush(stderr);
    const int o = dup(1), e = dup(2);
    FILE* f = fopen(path.c_str(), "w");
    dup2(fileno(f), 1);
    dup2(fileno(f), 2);
    {
        Mapper m(w.db, w.feats, opt, {}, nullptr, &seqs);
        out = m.run();
    }
    fflush(stdout);
    fflush(stderr);
    dup2(o, 1);
    dup2(e, 2);
    close(o);
    close(e);
    fclose(f);
    std::vector<std::string> lines;
    std::ifstream in(path);
    for (std::string l; std::getline(in, l);) {
        std::string t;
        for (size_t i = 0; i < l.size();) {
            if (l[i] < '0' || l[i] > '9') {
                t += l[i++];
                continue;
            }
            size_t j = l.find_first_not_of("0123456789.:", i);
            if (j == std::string::npos) j = l.size();
            const std::string run = l.substr(i, j - i), rest = l.substr(j, 3);
            auto unit = [&](const char* u) {
                const size_t n = std::strlen(u);
                return rest.compare(0, n, u) == 0 &&
                       (rest.size() <= n || !std::isalpha((unsigned char)rest[n]));
            };
            const bool time = run.find(':') != std::string::npos || unit(" s") || unit("s") ||
                              unit(" ms") || unit("ms");
            t += time ? "#" : run;
            i = j;
        }
        lines.push_back(t);
    }
    return lines;
}

static bool sameModel(const Reconstruction& a, const Reconstruction& b) {
    if (a.points3D.size() != b.points3D.size() || a.images.size() != b.images.size()) return false;
    for (const auto& kv : a.images) {
        auto it = b.images.find(kv.first);
        if (it == b.images.end() || it->second.registered != kv.second.registered ||
            it->second.pose.R != kv.second.pose.R || it->second.pose.t.x != kv.second.pose.t.x ||
            it->second.pose.t.y != kv.second.pose.t.y || it->second.pose.t.z != kv.second.pose.t.z)
            return false;
    }
    for (const auto& kv : a.cameras) {
        auto it = b.cameras.find(kv.first);
        if (it == b.cameras.end() || it->second.fx != kv.second.fx || it->second.k1 != kv.second.k1)
            return false;
    }
    return true;
}

// SS_SFM_TRACE_DIR (sfm/map/Trace.h): off writes nothing; on, the same model and the same log,
// and one readable snapshot per events.log line.
static void testTrace(int device) {
    namespace fs = std::filesystem;
    Walk w = makeWalk();
    MapperOptions opt;
    opt.verbose = true;
    opt.focal = w.focal;
    opt.device = device;
    std::vector<SequenceDef> defs(1);
    defs[0].members = {""};
    const SequenceTable seqs = buildSequenceTable(w.names, defs);
    const fs::path root = fs::temp_directory_path() / ("sfm_trace_test_" + std::to_string(getpid()));
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path dir = root / "trace";
    unsetenv("SS_SFM_TRACE_DIR");

    std::vector<Reconstruction> off, on;
    const std::vector<std::string> log_off = mapperLog(w, opt, seqs, (root / "off.log").string(), off);
    check(!fs::exists(dir), "tracing off writes no trace");

    setenv("SS_SFM_TRACE_DIR", dir.string().c_str(), 1);
    setenv("SS_SFM_TRACE_IMAGES", "10-30", 1);
    setenv("SS_SFM_TRACE_POINTS", "50", 1);
    const std::vector<std::string> log_on = mapperLog(w, opt, seqs, (root / "on.log").string(), on);
    unsetenv("SS_SFM_TRACE_DIR");
    unsetenv("SS_SFM_TRACE_IMAGES");
    unsetenv("SS_SFM_TRACE_POINTS");

    check(!off.empty() && !on.empty() && off.size() == on.size() && sameModel(off[0], on[0]),
          "tracing on gives the same model");
    size_t diff = log_off.size() == log_on.size() ? 0 : 1;
    for (size_t i = 0; i < std::min(log_off.size(), log_on.size()); i++)
        if (log_off[i] != log_on[i]) {
            if (!diff) printf("  log differs at line %zu:\n    %s\n    %s\n", i, log_off[i].c_str(),
                              log_on[i].c_str());
            diff++;
        }
    check(diff == 0 && log_off.size() > 20, "tracing on leaves the log line for line as it was");

    std::ifstream ev((dir / "events.log").string());
    size_t lines = 0, regs = 0, bas = 0, filters = 0, seeds = 0, bins = 0, sampled_ok = 0;
    for (std::string l; std::getline(ev, l); lines++) {
        std::istringstream ls(l);
        std::string idx, type;
        ls >> idx >> type;
        regs += type == "reg";
        bas += type == "ba";
        filters += type == "filter";
        seeds += type == "seed";
        std::ifstream b((dir / MapTrace::fileName((uint32_t)std::stoul(idx))).string(), std::ios::binary);
        char magic[4] = {};
        uint32_t ver = 0, index = 0;
        b.read(magic, 4);
        b.read((char*)&ver, 4);
        b.read((char*)&index, 4);
        if (b && std::string(magic, 4) == "SSMT" && ver == 1 && index == std::stoul(idx)) bins++;
        uint32_t mapper = 0, nimg = 0, ncam = 0, lo = 0, hi = 0, take = 0;
        char type16[16];
        uint64_t npts = 0, nobs = 0, nwin = 0;
        b.read((char*)&mapper, 4);
        b.read(type16, 16);
        b.read((char*)&nimg, 4);
        b.seekg(nimg * (8 + 12 * 8), std::ios::cur);
        b.read((char*)&ncam, 4);
        b.seekg(ncam * (16 + 14 * 8), std::ios::cur);
        b.read((char*)&npts, 8);
        b.read((char*)&nobs, 8);
        b.read((char*)&lo, 4);
        b.read((char*)&hi, 4);
        b.read((char*)&nwin, 8);
        b.read((char*)&take, 4);
        b.seekg(take * 16, std::ios::cur);
        const bool at_end = b && b.peek() == EOF;
        if (at_end && lo == 10 && hi == 30 && take <= 50 && take <= nwin && nwin <= npts) sampled_ok++;
    }
    printf("trace: %zu events (%zu registrations, %zu BAs, %zu filters, %zu seeds), %zu snapshots "
           "read\n", lines, regs, bas, filters, seeds, bins);
    check(lines > 0 && bins == lines && sampled_ok == lines, "one well-formed snapshot per event");
    check(regs + 2 * seeds >= on[0].numRegistered() && bas > 0 && filters == bas && seeds > 0,
          "registrations, seeds, BAs and filters are traced");

    setenv("SS_SFM_TRACE_DIR", (root / "bad").string().c_str(), 1);
    setenv("SS_SFM_TRACE_IMAGES", "9", 1);
    bool threw = false;
    try {
        Mapper m(w.db, w.feats, opt, {}, nullptr, &seqs);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    unsetenv("SS_SFM_TRACE_DIR");
    unsetenv("SS_SFM_TRACE_IMAGES");
    check(threw, "an invalid SS_SFM_TRACE_IMAGES is refused");
    fs::remove_all(root);
}

// SS_SFM_TIE_MERGE: a wall seen by 20 cameras, cut between 9 and 10. The model holds the
// block 10-19 at twice its scale about camera 10, with its own copy of every point both
// sides see; only points seen by 9 and 10 alone are shared. Verified matches tie the copies.
struct TieScene {
    Walk w;
    std::vector<Vec3> X;       // per feature, the true point
    std::vector<uint8_t> dup;  // per feature: seen twice or more on each side
};

static TieScene makeTieScene() {
    TieScene s;
    Walk& w = s.w;
    const int NP = 500, NL = 40, M = 20;
    Camera K = Camera::defaultFor(1, w.W, w.H, w.focal);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ux(4.0, 8.0), uy(-3.5, 3.5), uz(-2.0, 11.5);
    std::normal_distribution<double> noise(0.0, 0.3);
    std::vector<Vec3>& X = s.X;
    X.resize(NP + NL);
    for (int p = 0; p < NP; p++) X[p] = {ux(rng), uy(rng), uz(rng)};
    std::uniform_real_distribution<double> ul(3.5, 6.0);
    for (int p = NP; p < NP + NL; p++) X[p] = {ux(rng), uy(rng), ul(rng)};
    for (int c = 0; c < M; c++) w.gt.push_back(lookAtX({0, 0, 0.5 * c}));
    w.feats.resize(M);
    w.names.resize(M);
    std::vector<std::vector<char>> vis(M, std::vector<char>(NP + NL, 0));
    for (int c = 0; c < M; c++) {
        char nm[32];
        snprintf(nm, sizeof nm, "%03d", c);
        w.names[c] = nm;
        FeatureSet& fs = w.feats[c];
        fs.width = w.W;
        fs.height = w.H;
        fs.keypoints.assign(NP + NL, {-1000, -1000, 2, 0, 0});
        for (int p = 0; p < NP + NL; p++) {
            if (p >= NP && c != 9 && c != 10) continue;
            Vec3 pc = mul(w.gt[c].R, X[p]) + w.gt[c].t;
            Vec2 px = K.project(pc);
            if (pc.z > 0.1 && px.x > 0 && px.x < w.W && px.y > 0 && px.y < w.H) {
                fs.keypoints[p] = {(float)(px.x + noise(rng)), (float)(px.y + noise(rng)), 2, 0, 0};
                vis[c][p] = 1;
            }
        }
    }
    w.db.images.resize(M);
    for (int c = 0; c < M; c++) w.db.images[c] = {w.names[c], (uint32_t)(NP + NL)};
    for (int i = 0; i < M; i++)
        for (int j = i + 1; j < M; j++) {
            TwoViewMatches tv;
            tv.image1 = i;
            tv.image2 = j;
            tv.config = (int)TwoViewConfig::Uncalibrated;
            for (int p = 0; p < NP + NL; p++)
                if (vis[i][p] && vis[j][p]) tv.matches.push_back({(uint32_t)p, (uint32_t)p});
            if (tv.matches.size() >= 15) w.db.pairs.push_back(std::move(tv));
        }
    s.dup.assign(NP + NL, 0);
    for (int p = 0; p < NP; p++) {
        int pre = 0, blk = 0;
        for (int c = 0; c < M; c++) (c < 10 ? pre : blk) += vis[c][p];
        s.dup[p] = pre >= 2 && blk >= 2;
    }
    return s;
}

// `base` (any model of the scene with every image registered) with the true poses, the
// block scaled about camera 10, and each side's points triangulated where that side has them.
static Reconstruction splitModel(const TieScene& s, const Reconstruction& base, double scale) {
    const Walk& w = s.w;
    Reconstruction m = base;
    const Vec3 pivot = cameraCenter(w.gt[10]);
    for (auto& kv : m.cameras) {
        Camera K = Camera::defaultFor(1, w.W, w.H, w.focal);
        K.id = kv.first;
        kv.second = K;
    }
    for (auto& kv : m.images) {
        Pose p = w.gt[kv.first];
        if (kv.first >= 10) {
            const Vec3 c = pivot + (cameraCenter(p) - pivot) * scale;
            const Vec3 t = mul(p.R, c);
            p.t = {-t.x, -t.y, -t.z};
        }
        kv.second.pose = p;
        std::fill(kv.second.point3D_ids.begin(), kv.second.point3D_ids.end(), kInvalidPoint3D);
    }
    m.points3D.clear();
    m.next_point3D_id = 1;
    for (uint32_t p = 0; p < (uint32_t)s.X.size(); p++)
        for (int side = 0; side < 2; side++) {
            std::vector<TrackElement> tr;
            for (uint32_t c = side ? 10 : 0; c < (side ? 20u : 10u); c++)
                if (w.feats[c].keypoints[p].x >= 0) tr.push_back({c, p});
            if (tr.size() < 2) continue;
            m.addPoint3D(side ? pivot + (s.X[p] - pivot) * scale : s.X[p], tr);
        }
    return m;
}

// Copies of one scene point that end up as one point id.
static size_t joinedCopies(const TieScene& s, const Reconstruction& m) {
    size_t n = 0;
    auto idOf = [&](uint32_t c, uint32_t p) {
        auto it = m.images.find(c);
        return it == m.images.end() || !it->second.registered ? kInvalidPoint3D
                                                               : it->second.point3D_ids[p];
    };
    for (uint32_t p = 0; p < (uint32_t)s.dup.size(); p++) {
        if (!s.dup[p]) continue;
        std::vector<uint64_t> pre;
        for (uint32_t c = 0; c < 10; c++)
            if (idOf(c, p) != kInvalidPoint3D) pre.push_back(idOf(c, p));
        bool joined = false;
        for (uint32_t c = 10; c < 20 && !joined; c++)
            joined = idOf(c, p) != kInvalidPoint3D &&
                     std::find(pre.begin(), pre.end(), idOf(c, p)) != pre.end();
        n += joined;
    }
    return n;
}

// Median camera step over 11-19 over the median over 1-9: 1 is one scale.
static double blockRatio(const Reconstruction& m) {
    auto steps = [&](uint32_t lo, uint32_t hi) {
        std::vector<double> v;
        for (uint32_t c = lo; c <= hi; c++) {
            auto a = m.images.find(c - 1), b = m.images.find(c);
            if (a == m.images.end() || b == m.images.end() || !a->second.registered ||
                !b->second.registered)
                continue;
            v.push_back((cameraCenter(a->second.pose) - cameraCenter(b->second.pose)).norm());
        }
        std::sort(v.begin(), v.end());
        return v.empty() ? 0.0 : v[v.size() / 2];
    };
    const double pre = steps(1, 9);
    return pre > 0 ? steps(11, 19) / pre : -1;
}

static void setTieEnv(const char* n, const char* keep, const char* near_d) {
    const char* names[] = {"SS_SFM_TIE_MERGE", "SS_SFM_TIE_KEEP", "SS_SFM_TIE_NEAR"};
    const char* vals[] = {n, keep, near_d};
    for (int i = 0; i < 3; i++) {
        if (vals[i]) setenv(names[i], vals[i], 1);
        else unsetenv(names[i]);
    }
}

static Reconstruction tieRefine(const TieScene& s, const Reconstruction& split,
                                const MapperOptions& opt, const char* n, const char* keep,
                                const char* near_d = nullptr) {
    setTieEnv(n, keep, near_d);
    Reconstruction out;
    {
        Mapper m(s.w.db, s.w.feats, opt);
        out = m.refine(split);
    }
    setTieEnv(nullptr, nullptr, nullptr);
    return out;
}

static void testTieMerge(int device) {
    TieScene s = makeTieScene();
    MapperOptions opt;
    opt.focal = s.w.focal;
    opt.device = device;
    setTieEnv(nullptr, nullptr, nullptr);
    std::vector<Reconstruction> base;
    {
        Mapper m(s.w.db, s.w.feats, opt);
        base = m.run();
    }
    check(!base.empty() && base[0].numRegistered() == 20, "tie scene: the clean scene registers");
    if (base.empty() || base[0].numRegistered() != 20) return;
    const Reconstruction split = splitModel(s, base[0], 2.0);
    size_t ndup = 0;
    for (uint8_t d : s.dup) ndup += d;

    const Reconstruction off = tieRefine(s, split, opt, nullptr, nullptr);
    const Reconstruction zero = tieRefine(s, split, opt, "0", nullptr);
    const Reconstruction keep = tieRefine(s, split, opt, "3", nullptr);
    const Reconstruction drop = tieRefine(s, split, opt, "3", "0");
    const Reconstruction gated = tieRefine(s, split, opt, "3", "1", "20");
    const Reconstruction high = tieRefine(s, split, opt, "1000", nullptr);
    struct Row { const char* name; const Reconstruction* m; } rows[] = {
        {"split", &split}, {"off", &off}, {"N=3 keep", &keep}, {"N=3 drop", &drop},
        {"N=3 keep near 20", &gated}, {"N=1000", &high}};
    for (const Row& r : rows)
        printf("tie-merge: %-16s block/pre %.3f, %zu/%zu copies one point, %zu points, %u "
               "images\n", r.name, blockRatio(*r.m), joinedCopies(s, *r.m), ndup,
               r.m->points3D.size(), r.m->numRegistered());
    check(std::fabs(blockRatio(split) - 2.0) < 0.01 && joinedCopies(s, split) == 0,
          "the split model holds the block at x2, every copy apart");
    check(sameModel(off, zero) && off.points3D.size() == zero.points3D.size(),
          "SS_SFM_TIE_MERGE=0 is off");
    check(blockRatio(off) > 1.9 && joinedCopies(s, off) <= ndup / 50,
          "off: the copies stay two points, the block x2");
    check(sameModel(off, high) && off.points3D.size() == high.points3D.size(),
          "a tie count nothing reaches merges nothing");
    check(std::fabs(blockRatio(keep) - 1.0) < 0.03 && joinedCopies(s, keep) >= 0.9 * ndup,
          "N=3 keep: the copies become one point and the block one scale with the rest");
    check(std::fabs(blockRatio(gated) - 1.0) < 0.03, "the near gate keeps the wall's ties");

    // Off leaves the mapper's log line for line as it was.
    Walk w = makeWalk();
    MapperOptions wo;
    wo.verbose = true;
    wo.focal = w.focal;
    wo.device = device;
    std::vector<SequenceDef> defs(1);
    defs[0].members = {""};
    const SequenceTable seqs = buildSequenceTable(w.names, defs);
    const std::string tmp = (std::filesystem::temp_directory_path() /
                             ("sfm_tie_test_" + std::to_string(getpid()))).string();
    std::vector<Reconstruction> a, b;
    const std::vector<std::string> log_unset = mapperLog(w, wo, seqs, tmp + ".a.log", a);
    setTieEnv("0", "0", "5");
    const std::vector<std::string> log_zero = mapperLog(w, wo, seqs, tmp + ".b.log", b);
    setTieEnv(nullptr, nullptr, nullptr);
    std::filesystem::remove(tmp + ".a.log");
    std::filesystem::remove(tmp + ".b.log");
    check(log_unset == log_zero && !a.empty() && !b.empty() && sameModel(a[0], b[0]),
          "SS_SFM_TIE_MERGE=0: the walk's model and log as without it");

    // On over a whole run (the walk repeats its room feature for feature).
    setTieEnv("3", nullptr, nullptr);
    std::vector<Reconstruction> on;
    {
        MapperOptions quiet = wo;
        quiet.verbose = false;
        Mapper m(w.db, w.feats, quiet, {}, nullptr, &seqs);
        on = m.run();
    }
    setTieEnv(nullptr, nullptr, nullptr);
    uint32_t reg = 0;
    const double err = on.empty() ? -1 : shapeError(on.front(), w, reg);
    printf("tie-merge: the walk with N=3 keep: %u/%zu cameras in %zu model(s), shape error %.3f\n",
           reg, w.gt.size(), on.size(), err);
}

// SS_SFM_TIE_SPAN_IMAGES / _MAXSPAN_IMAGES / _CLASH. The tie scene with a revisit: cameras
// 20-29 see a repeat of the wall 40 further on (as 0-9 see the wall, 0.3 to the side), so
// verified matches tie distinct places; the block 10-19 at x2 as before; each side its own
// copy of every point it sees twice. Sequence positions 0-9, 20-29 (block) and 320-329
// (revisit), so the block's ties are matches at most 29 positions apart, the revisit's at
// least 291. For the first ND points camera 10 has a second keypoint `off` px away, which the
// pre side's copy holds:
// fusing that copy with the block's holds camera 10 twice (a clash).
struct SpanScene {
    Walk w;
    std::vector<Vec3> X;
    std::vector<uint8_t> dupB, dupR;   // seen twice or more on the pre side and the block / revisit
    std::vector<uint32_t> clash_p;     // points with camera 10's second keypoint
    SequenceTable seqs;
};

static constexpr uint32_t kSpanNP = 500, kSpanND = 30, kSpanM = 30;
static constexpr double kSpanD = 40.0;

static SpanScene makeSpanScene(float off) {
    SpanScene s;
    Walk& w = s.w;
    const uint32_t NP = kSpanNP, ND = kSpanND, M = kSpanM, NF = NP + ND;
    Camera K = Camera::defaultFor(1, w.W, w.H, w.focal);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> ux(4.0, 8.0), uy(-3.5, 3.5), uz(-2.0, 11.5);
    std::normal_distribution<double> noise(0.0, 0.3);
    s.X.resize(NP);
    for (Vec3& p : s.X) p = {ux(rng), uy(rng), uz(rng)};
    for (uint32_t c = 0; c < M; c++)
        w.gt.push_back(lookAtX({c >= 20 ? 0.3 : 0.0, 0, 0.5 * (c % 20) + (c >= 20 ? kSpanD : 0)}));
    w.feats.resize(M);
    w.names.resize(M);
    std::vector<std::vector<char>> vis(M, std::vector<char>(NF, 0));
    for (uint32_t c = 0; c < M; c++) {
        char nm[32];
        snprintf(nm, sizeof nm, "%03u", c);
        w.names[c] = nm;
        FeatureSet& fs = w.feats[c];
        fs.width = w.W;
        fs.height = w.H;
        fs.keypoints.assign(NF, {-1000, -1000, 2, 0, 0});
        for (uint32_t p = 0; p < NP; p++) {
            Vec3 pc = mul(w.gt[c].R, s.X[p] + Vec3{0, 0, c >= 20 ? kSpanD : 0}) + w.gt[c].t;
            Vec2 px = K.project(pc);
            if (pc.z > 0.1 && px.x > 5 && px.x < w.W - 5 && px.y > 0 && px.y < w.H) {
                fs.keypoints[p] = {(float)(px.x + noise(rng)), (float)(px.y + noise(rng)), 2, 0, 0};
                vis[c][p] = 1;
            }
        }
    }
    w.db.images.resize(M);
    for (uint32_t c = 0; c < M; c++) w.db.images[c] = {w.names[c], NF};
    for (uint32_t i = 0; i < M; i++)
        for (uint32_t j = i + 1; j < M; j++) {
            TwoViewMatches tv;
            tv.image1 = i;
            tv.image2 = j;
            tv.config = (int)TwoViewConfig::Uncalibrated;
            for (uint32_t p = 0; p < NP; p++)
                if (vis[i][p] && vis[j][p]) tv.matches.push_back({p, p});
            if (tv.matches.size() >= 15) w.db.pairs.push_back(std::move(tv));
        }
    s.dupB.assign(NP, 0);
    s.dupR.assign(NP, 0);
    for (uint32_t p = 0; p < NP; p++) {
        int pre = 0, blk = 0, rev = 0;
        for (uint32_t c = 0; c < M; c++) (c < 10 ? pre : c < 20 ? blk : rev) += vis[c][p];
        s.dupB[p] = pre >= 2 && blk >= 2;
        s.dupR[p] = pre >= 2 && rev >= 2;
        if (s.dupB[p] && vis[10][p] && s.clash_p.size() < ND) {
            const uint32_t f = NP + (uint32_t)s.clash_p.size();
            w.feats[10].keypoints[f] = w.feats[10].keypoints[p];
            w.feats[10].keypoints[f].x += off;
            s.clash_p.push_back(p);
        }
    }
    s.seqs.seq.assign(M, 0);
    s.seqs.member.assign(M, 0);
    s.seqs.pos.resize(M);
    for (uint32_t c = 0; c < M; c++) s.seqs.pos[c] = c < 10 ? c : c < 20 ? c + 10 : c + 300;
    s.seqs.length = {330};
    s.seqs.members = {{""}};
    return s;
}

// The true poses, the block scaled about camera 10, each side's copies; the pre side's
// clash points also hold camera 10's second keypoint.
static Reconstruction splitSpanModel(const SpanScene& s, const Reconstruction& base) {
    const Walk& w = s.w;
    Reconstruction m = base;
    const Vec3 pivot = cameraCenter(w.gt[10]);
    for (auto& kv : m.cameras) {
        Camera K = Camera::defaultFor(1, w.W, w.H, w.focal);
        K.id = kv.first;
        kv.second = K;
    }
    for (auto& kv : m.images) {
        Pose p = w.gt[kv.first];
        if (kv.first >= 10 && kv.first < 20) {
            const Vec3 c = pivot + (cameraCenter(p) - pivot) * 2.0;
            const Vec3 t = mul(p.R, c);
            p.t = {-t.x, -t.y, -t.z};
        }
        kv.second.pose = p;
        std::fill(kv.second.point3D_ids.begin(), kv.second.point3D_ids.end(), kInvalidPoint3D);
    }
    m.points3D.clear();
    m.next_point3D_id = 1;
    for (uint32_t p = 0; p < kSpanNP; p++)
        for (uint32_t side = 0; side < 3; side++) {
            std::vector<TrackElement> tr;
            for (uint32_t c = side * 10; c < side * 10 + 10; c++)
                if (w.feats[c].keypoints[p].x >= 0) tr.push_back({c, p});
            if (tr.size() < 2) continue;
            if (side == 0)
                for (uint32_t k = 0; k < s.clash_p.size(); k++)
                    if (s.clash_p[k] == p) tr.push_back({10, kSpanNP + k});
            m.addPoint3D(side == 1   ? pivot + (s.X[p] - pivot) * 2.0
                         : side == 2 ? s.X[p] + Vec3{0, 0, kSpanD}
                                     : s.X[p],
                         tr);
        }
    return m;
}

// Points of `mask` whose pre-side copy shares an id with a copy in cameras [lo, lo + 10).
static size_t spanJoined(const SpanScene& s, const Reconstruction& m,
                         const std::vector<uint8_t>& mask, uint32_t lo) {
    auto idOf = [&](uint32_t c, uint32_t p) {
        auto it = m.images.find(c);
        return it == m.images.end() || !it->second.registered ? kInvalidPoint3D
                                                               : it->second.point3D_ids[p];
    };
    size_t n = 0;
    for (uint32_t p = 0; p < kSpanNP; p++) {
        if (!mask[p]) continue;
        std::vector<uint64_t> pre;
        for (uint32_t c = 0; c < 10; c++)
            if (idOf(c, p) != kInvalidPoint3D) pre.push_back(idOf(c, p));
        bool joined = false;
        for (uint32_t c = lo; c < lo + 10 && !joined; c++)
            joined = idOf(c, p) != kInvalidPoint3D &&
                     std::find(pre.begin(), pre.end(), idOf(c, p)) != pre.end();
        n += joined;
    }
    return n;
}

static void setSpanEnv(const char* n, const char* span, const char* maxspan, const char* clash) {
    const char* names[] = {"SS_SFM_TIE_MERGE", "SS_SFM_TIE_SPAN_IMAGES",
                           "SS_SFM_TIE_MAXSPAN_IMAGES", "SS_SFM_TIE_CLASH"};
    const char* vals[] = {n, span, maxspan, clash};
    for (int i = 0; i < 4; i++) {
        if (vals[i]) setenv(names[i], vals[i], 1);
        else unsetenv(names[i]);
    }
}

// The span scene's model under each knob set: `off` px between camera 10's two keypoints.
static void testTieSpan(int device, float off) {
    SpanScene s = makeSpanScene(off);
    MapperOptions opt;
    opt.focal = s.w.focal;
    opt.device = device;
    setSpanEnv(nullptr, nullptr, nullptr, nullptr);
    setTieEnv(nullptr, nullptr, nullptr);
    std::vector<Reconstruction> base;
    {
        Mapper m(s.w.db, s.w.feats, opt);
        base = m.run();
    }
    check(!base.empty() && base[0].numRegistered() == kSpanM, "span scene: the clean scene registers");
    if (base.empty() || base[0].numRegistered() != kSpanM) return;
    const Reconstruction split = splitSpanModel(s, base[0]);
    std::vector<uint8_t> clash(kSpanNP, 0), plainB = s.dupB;
    for (uint32_t p : s.clash_p) clash[p] = 1, plainB[p] = 0;
    size_t nB = 0, nR = 0;
    for (uint32_t p = 0; p < kSpanNP; p++) nB += plainB[p], nR += s.dupR[p];

    auto run = [&](const char* n, const char* span, const char* maxspan, const char* cl) {
        setSpanEnv(n, span, maxspan, cl);
        Reconstruction out;
        {
            Mapper m(s.w.db, s.w.feats, opt, {}, nullptr, &s.seqs);
            out = m.refine(split);
        }
        setSpanEnv(nullptr, nullptr, nullptr, nullptr);
        return out;
    };
    const Reconstruction off_m = run(nullptr, nullptr, nullptr, nullptr);
    const Reconstruction on = run("3", nullptr, nullptr, nullptr);
    const Reconstruction span0 = run("3", "0", "0", "refuse");
    const Reconstruction wide = run("3", "1000", "1000", nullptr);
    const Reconstruction span = run("3", "60", nullptr, nullptr);
    const Reconstruction maxspan = run("3", nullptr, "60", nullptr);
    const Reconstruction resolve = run("3", "60", nullptr, "resolve");
    struct Row { const char* name; const Reconstruction* m; } rows[] = {
        {"split", &split}, {"off", &off_m}, {"N=3", &on}, {"N=3 span 1000", &wide},
        {"N=3 span 60", &span}, {"N=3 maxspan 60", &maxspan}, {"N=3 span 60 resolve", &resolve}};
    for (const Row& r : rows)
        printf("tie-span: %.0f px %-20s block/pre %.3f, block %zu/%zu, revisit %zu/%zu, clash %zu/%zu "
               "copies one point, %zu points\n", off, r.name, blockRatio(*r.m), spanJoined(s, *r.m, plainB, 10),
               nB, spanJoined(s, *r.m, s.dupR, 20), nR, spanJoined(s, *r.m, clash, 10),
               s.clash_p.size(), r.m->points3D.size());
    check(s.clash_p.size() == kSpanND, "span scene: every clash point placed");
    size_t true_kp = 0;
    for (uint32_t k = 0; k < kSpanND; k++) {
        const Image& im = resolve.images.at(10);
        true_kp += im.point3D_ids[s.clash_p[k]] != kInvalidPoint3D &&
                   im.point3D_ids[kSpanNP + k] == kInvalidPoint3D;
    }
    printf("tie-span: %.0f px: resolve keeps camera 10's true keypoint for %zu/%u clash points\n",
           off, true_kp, kSpanND);
    if (off < 5) {
        // 2 px: both keypoints are inliers, so refusing keeps the clash points apart for good.
        check(spanJoined(s, span, clash, 10) <= 2, "refuse: the clash points stay two points");
        check(spanJoined(s, resolve, clash, 10) >= kSpanND / 2,
              "resolve: half the clash points or more fused");
        return;
    }
    check(sameModel(on, span0) && on.points3D.size() == span0.points3D.size(),
          "SPAN 0, MAXSPAN 0, CLASH=refuse are off");
    check(sameModel(on, wide) && on.points3D.size() == wide.points3D.size(),
          "a span every match is within changes nothing");
    check(blockRatio(off_m) > 1.9 && spanJoined(s, off_m, s.dupR, 20) <= nR / 50,
          "off: the block x2, the revisit's copies apart");
    check(spanJoined(s, on, s.dupR, 20) >= nR / 4, "N=3: the revisit's copies fused (wrong joins)");
    check(std::fabs(blockRatio(span) - 1.0) < 0.03 && spanJoined(s, span, plainB, 10) >= 0.9 * nB &&
              spanJoined(s, span, s.dupR, 20) <= nR / 50,
          "span 60: the block's copies fused, the revisit's (291+ apart) not");
    check(std::fabs(blockRatio(maxspan) - 1.0) < 0.03 && spanJoined(s, maxspan, s.dupR, 20) <= nR / 50,
          "maxspan 60: no union reaches the revisit");
    check(spanJoined(s, resolve, clash, 10) >= kSpanND * 8 / 10 && true_kp >= kSpanND * 8 / 10,
          "resolve (10 px): the clash points fused on camera 10's true keypoint");
    check(std::fabs(blockRatio(resolve) - 1.0) < 0.03, "resolve: the block one scale");
    bool threw = false;
    setSpanEnv("3", nullptr, nullptr, "maybe");
    try {
        Mapper m(s.w.db, s.w.feats, opt);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    setSpanEnv(nullptr, nullptr, nullptr, nullptr);
    check(threw, "an invalid SS_SFM_TIE_CLASH is refused");
}
#endif

static int cmdSequenceTest(int argc, char** argv) {
    int device = -1;
    bool verbose = false, gpu = true;
    for (int i = 0; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--device" && i + 1 < argc) device = std::stoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else if (a == "--no-gpu") gpu = false;
    }
    testTable();
    if (gpu) testMapperWalk(device, verbose);
#ifndef _WIN32
    if (gpu) testTrace(device);
    if (gpu) testTieMerge(device);
    if (gpu) testTieSpan(device, 10);
    if (gpu) testTieSpan(device, 2);
#endif
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) { return sfmTestMain(argc - 1, argv + 1, cmdSequenceTest); }
