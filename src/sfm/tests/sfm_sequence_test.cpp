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
#endif
    printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}

int main(int argc, char** argv) { return sfmTestMain(argc - 1, argv + 1, cmdSequenceTest); }
