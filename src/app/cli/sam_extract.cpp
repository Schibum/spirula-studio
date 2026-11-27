// `spirula-sam extract` -- the sharpest frames of a video, optionally masked.
//
// Decode is Vulkan Video where the build and device have it and ffmpeg where
// they do not (app/FrameDecode.h). Selection is extract_frames.py's -- a
// window of `keep` frames, one written every `skip` -- so every decoder names
// the same source frames alike.
//
//   spirula-sam extract clip.mp4 --skip 10
//   spirula-sam extract clip.mp4 --skip 10 --model sam3-f16.ggml --text "person; car"

#include "app/FfmpegVideo.h"
#include "app/FrameExtract.h"
#include "app/FrameStream.h"
#include "app/LogLuts.h"
#include "app/Tools.h"
#include "app/cli/ModelLicense.h"
#include "i18n/catalog/Cli.h"
#include "i18n/catalog/Data.h"
#include "i18n/catalog/SamHelp.h"
#include "nn/core/Log.h"
#include "birefnet/BiRefNet.h"
#include "sam/Masking.h"
#ifdef SS_TOOL_SFM
#include "sfm/core/Telemetry.h"
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

void set_env(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

// A row of `--help`: identifier on the left, translated sentence on the right,
// wrapped by i18n::wrap so a language without spaces still breaks.
void help_row(const char* flags, const std::string& text) {
    constexpr int kCol = 25;
    std::string left = std::string("  ") + flags;
    if (spirula::i18n::display_width(left) >= kCol) {
        std::fprintf(stderr, "%s\n", left.c_str());
        left.clear();
    }
    left = spirula::i18n::pad_to(left, kCol);
    for (const std::string& line : spirula::i18n::wrap(text, 86 - kCol)) {
        std::fprintf(stderr, "%s%s\n", left.c_str(), line.c_str());
        left.assign(kCol, ' ');
    }
}

void help_row(const char* flags, const spirula::i18n::Msg& m) {
    help_row(flags, std::string(m.get()));
}

void usage() {
    namespace H = spirula::i18n::msg::samhelp;
    std::fprintf(stderr, "%s extract <video> [options]\n\n",
                 app::program_name().c_str());

    std::fprintf(stderr, "%s\n", H::xh_frame_selection.get());
    help_row("-o, --out <dir>", H::xh_out);
    help_row("-s, --skip <n>", H::xh_skip);
    help_row("-k, --keep <n>", H::xh_keep);
    help_row("-n, --max-frames <n>", H::xh_max_frames);
    help_row("    --start <time>", H::xh_start);
    help_row("    --end <time>", H::xh_end);
    help_row("-q, --quality <0..100>", H::xh_quality);
    help_row("-r, --rotate <deg>", H::xh_rotate);
    help_row("    --no-autorotate", H::xh_no_autorotate);
    help_row("    --scale <f>", H::xh_scale);
    std::string luts;
    for (const app::BuiltinLut& b : app::builtin_luts())
        luts += (luts.empty() ? "" : ", ") + std::string(b.id);
    help_row("    --lut <file|name>", spirula::i18n::format(H::xh_lut, {luts}));
    help_row("    --track <i>", H::xh_track);
    help_row("    --sync", H::xh_sync);
    help_row("    --adaptive", H::xh_adaptive);
    help_row("    --adaptive-range <f>", H::xh_adaptive_range);
    help_row("    --blur-reject <f>", H::xh_blur_reject);
    help_row("    --similar <n>", H::xh_similar);
    help_row("    --skipped-dir <dir>", H::xh_skipped_dir);
    help_row("    --report <file>", H::xh_report);
    help_row("    --plan-out <file>", H::xh_plan_out);
    help_row("    --plan-in <file>", H::xh_plan_in);
    help_row("    --progress <file>", H::xh_progress);
    help_row("    --threads <n>", H::xh_threads);
    help_row("    --decoder <d>", H::xh_decoder);
    help_row("    --ffmpeg <exe>", H::xh_ffmpeg);

    std::fprintf(stderr, "\n%s\n", H::xh_360_section.get());
    help_row("    --360 <mode>", H::xh_360);
    help_row("    --360-size <n>", H::xh_360_size);
    help_row("    --360-orient <y,p,r>", H::xh_360_orient);

    std::fprintf(stderr, "\n%s\n", H::xh_masking.get());
    for (const std::string& l : spirula::i18n::wrap(H::model_kinds.get(), 76))
        std::fprintf(stderr, "  %s\n", l.c_str());
    help_row("    --model <file>", H::xh_model);
    help_row("    --detector <id>", H::opt_detector);
    help_row("    --detector-threshold <f>", H::opt_detector_threshold);
    help_row("    --text <phrases>", H::xh_text);
    help_row("    --neg-text <ph>", H::xh_neg_text);
    help_row("    --mask-mode <m>", H::xh_mask_mode);
    help_row("    --mask-keep <w>", H::xh_mask_keep);
    help_row("    --mask-out <dir>", H::xh_mask_out);
    help_row("    --detect-every <n>", H::xh_detect_every);
    help_row("    --memory-frames <n>", H::xh_memory_frames);
    help_row("    --max-size <n>", H::xh_max_size);
    help_row("    --threshold <f>", H::xh_threshold);
    help_row("    --nms <f>", H::xh_nms);
    help_row("    --dilate-ratio <f>", H::mask_dilate);
    help_row("    --overlay", H::xh_overlay);

    std::fprintf(stderr, "\n%s --device <index|name|auto|-1|uuid:hex>  --profile  "
                         "--validate\n",
                 H::label_common.get());
}

// Seconds, or [[h:]m:]s with a fraction.
bool parse_seconds(const char* text, double& out) {
    double total = 0.0;
    int parts = 0;
    for (const char* p = text; *p;) {
        char* end = nullptr;
        const double v = std::strtod(p, &end);
        if (end == p || !(v >= 0.0) || ++parts > 3) return false;
        total = total * 60.0 + v;
        p = end;
        if (*p == ':') p++;
        else if (*p) return false;
    }
    if (parts == 0) return false;
    out = total;
    return true;
}

struct Options {
    std::string input;
    std::string out_dir, mask_dir;
    int    skip = 1, keep = -1, max_frames = 0;
    double clip_start = 0.0, clip_end = 0.0;
    int    quality = 95, rotate = 0;
    bool   auto_rotate = true;
    float  scale = 1.0f;
    std::string lut;
    int    track = -1;
    bool   sync = false;
    bool   adaptive = false;
    float  adaptive_range = 4.0f;
    float  blur_reject = 0.0f;
    int    similar = 0;
    std::string skipped_dir;
    std::string report;
    std::string plan_out, plan_in, progress;
    int    threads = 0;
    app::FrameDecoder decoder = app::FrameDecoder::Auto;
    std::string ffmpeg = "ffmpeg";

    std::string pano_mode = "faces";
    app::Pano360Options pano;

    std::string model, text, neg_text, device, detector;
    float  detector_threshold = 0.3f;   // sam::MaskOptions, same default
    std::string mask_mode = "video";
    std::optional<bool> keep_subject;   // unset: BiRefNet keeps, SAM removes
    int    detect_every = 1, memory_frames = 0, max_size = 1600;
    float  threshold = 0.5f, nms = 0.1f;
    float  dilate_ratio = 0.05f;   // sam::MaskOptions, same default
    bool   overlay = false, profile = false, validate = false;
};

bool parse_args(int argc, char** argv, Options& o) {
    if (argc < 2) return false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s\n",
                             spirula::i18n::format(
                                 spirula::i18n::msg::cli::sam_flag_needs_value,
                                 {what}).c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-o" || a == "--out") o.out_dir = next("--out");
        else if (a == "-s" || a == "--skip") o.skip = std::atoi(next("--skip"));
        else if (a == "-k" || a == "--keep") o.keep = std::atoi(next("--keep"));
        else if (a == "-n" || a == "--max-frames") o.max_frames = std::atoi(next("--max-frames"));
        else if (a == "--start" || a == "--end") {
            const char* v = next(a.c_str());
            double t = 0.0;
            if (!parse_seconds(v, t)) {
                std::fprintf(stderr, "%s\n",
                             spirula::i18n::format(
                                 spirula::i18n::msg::cli::sam_unexpected_argument,
                                 {a + " " + v}).c_str());
                return false;
            }
            (a == "--start" ? o.clip_start : o.clip_end) = t;
        }
        else if (a == "-q" || a == "--quality") o.quality = std::atoi(next("--quality"));
        else if (a == "-r" || a == "--rotate") o.rotate = std::atoi(next("--rotate"));
        else if (a == "--no-autorotate") o.auto_rotate = false;
        else if (a == "--autorotate") o.auto_rotate = true;
        else if (a == "--scale") o.scale = std::strtof(next("--scale"), nullptr);
        else if (a == "--lut") o.lut = next("--lut");
        else if (a == "--track") o.track = std::atoi(next("--track"));
        else if (a == "--sync") o.sync = true;
        else if (a == "--adaptive") o.adaptive = true;
        else if (a == "--adaptive-range")
            o.adaptive_range = std::strtof(next("--adaptive-range"), nullptr);
        else if (a == "--blur-reject")
            o.blur_reject = std::strtof(next("--blur-reject"), nullptr);
        else if (a == "--similar")
            o.similar = std::atoi(next("--similar"));
        else if (a == "--skipped-dir")
            o.skipped_dir = next("--skipped-dir");
        else if (a == "--report")
            o.report = next("--report");
        else if (a == "--plan-out") o.plan_out = next("--plan-out");
        else if (a == "--plan-in") o.plan_in = next("--plan-in");
        else if (a == "--progress") o.progress = next("--progress");
        else if (a == "--threads") o.threads = std::atoi(next("--threads"));
        else if (a == "--decoder") {
            const std::string d = next("--decoder");
            if (d == "auto") o.decoder = app::FrameDecoder::Auto;
            else if (d == "builtin") o.decoder = app::FrameDecoder::Builtin;
            else if (d == "ffmpeg") o.decoder = app::FrameDecoder::Ffmpeg;
            else {
                std::fprintf(stderr, "%s\n",
                             spirula::i18n::format(
                                 spirula::i18n::msg::cli::sam_unexpected_argument,
                                 {"--decoder " + d}).c_str());
                return false;
            }
        }
        else if (a == "--ffmpeg") o.ffmpeg = next("--ffmpeg");
        else if (a == "--360") o.pano_mode = next("--360");
        else if (a == "--360-size") o.pano.size = std::atoi(next("--360-size"));
        else if (a == "--360-orient") {
            const char* v = next("--360-orient");
            if (std::sscanf(v, "%f,%f,%f", &o.pano.yaw, &o.pano.pitch,
                            &o.pano.roll) != 3) {
                std::fprintf(stderr, "%s\n",
                             spirula::i18n::format(
                                 spirula::i18n::msg::cli::sam_flag_needs_value,
                                 {"--360-orient"}).c_str());
                return false;
            }
        }
        else if (a == "--model") o.model = next("--model");
        else if (a == "--text") o.text = next("--text");
        else if (a == "--neg-text") o.neg_text = next("--neg-text");
        else if (a == "--mask-mode") o.mask_mode = next("--mask-mode");
        else if (a == "--mask-keep") o.keep_subject = std::strcmp(next("--mask-keep"), "subject") == 0;
        else if (a == "--detector") o.detector = next("--detector");
        else if (a == "--detector-threshold")
            o.detector_threshold = std::strtof(next("--detector-threshold"), nullptr);
        else if (a == "--mask-out") o.mask_dir = next("--mask-out");
        else if (a == "--detect-every") o.detect_every = std::atoi(next("--detect-every"));
        else if (a == "--memory-frames") o.memory_frames = std::atoi(next("--memory-frames"));
        else if (a == "--max-size") o.max_size = std::atoi(next("--max-size"));
        else if (a == "--threshold") o.threshold = std::strtof(next("--threshold"), nullptr);
        else if (a == "--nms") o.nms = std::strtof(next("--nms"), nullptr);
        else if (a == "--dilate-ratio")
            o.dilate_ratio = std::strtof(next("--dilate-ratio"), nullptr);
        else if (a == "--overlay") o.overlay = true;
        else if (a == "--device") o.device = next("--device");
        else if (a == "--profile") o.profile = true;
        else if (a == "--validate") o.validate = true;
        else if (a == "-h" || a == "--help") return false;
        else if (a[0] == '-') {
            std::fprintf(stderr, "%s\n",
                         spirula::i18n::format(
                             spirula::i18n::msg::cli::sam_unknown_option,
                             {a}).c_str());
            return false;
        } else if (o.input.empty()) {
            o.input = a;
        } else {
            std::fprintf(stderr, "%s\n",
                         spirula::i18n::format(
                             spirula::i18n::msg::cli::sam_unexpected_argument,
                             {a}).c_str());
            return false;
        }
    }
    if (o.input.empty()) return false;
    if (o.skip < 1) o.skip = 1;
    if (o.keep < 0) o.keep = (int)(0.5 * o.skip + 0.5);
    if (o.rotate % 90 != 0) {
        std::fprintf(stderr, "%s\n",
                     spirula::i18n::msg::cli::sam_rotate_multiple_of_90.get());
        return false;
    }
    return true;
}

// The --progress file: JSON lines, at most one each kInterval, written by a
// thread of its own so a writer thread reporting a frame never waits on it.
class ProgressFile {
public:
    bool open(const std::string& path) {
        std::error_code ec;
        const fs::path dir = fs::path(path).parent_path();
        if (!dir.empty()) fs::create_directories(dir, ec);
        f_ = std::fopen(path.c_str(), "w");
        if (!f_) return false;
        flusher_ = std::thread([this] {
            std::unique_lock<std::mutex> lk(mu_);
            for (;;) {
                cv_.wait(lk, [&] { return stop_ || !pending_.empty(); });
                if (stop_) return;
                cv_.wait_until(lk, last_ + kInterval, [&] { return stop_; });
                if (stop_) return;
                flush(lk, {});
            }
        });
        return true;
    }
    void landed(const std::vector<app::FrameLanded>& frames) {
        std::lock_guard<std::mutex> lk(mu_);
        for (const app::FrameLanded& f : frames) {
            pending_.push_back(f);
            if (!f.rejected) ++written_;
        }
        cv_.notify_one();
    }
    // The last line: done with the decoder, or not done with the reason. False
    // when the file could not be written, which is never done.
    bool close(const std::string& error, const std::string& decoder) {
        if (!f_) return true;
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        flusher_.join();
        std::unique_lock<std::mutex> lk(mu_);
        const std::string why = failed_ ? "could not write the progress file" : error;
        flush(lk, why.empty() ? ",\"done\":true,\"decoder\":" + json_quote(decoder)
                              : ",\"done\":false,\"error\":" + json_quote(why));
        const bool closed = std::fclose(f_) == 0;
        f_ = nullptr;
        return closed && !failed_;
    }
    ~ProgressFile() { close("stopped", ""); }

private:
    using Clock = std::chrono::steady_clock;
    static constexpr std::chrono::milliseconds kInterval{250};
    void flush(std::unique_lock<std::mutex>& lk, const std::string& tail) {
        std::vector<app::FrameLanded> frames;
        frames.swap(pending_);
        const int64_t written = written_;
        lk.unlock();
        const std::string line = app::frame_progress_line(written, frames, tail);
        const bool ok = std::fwrite(line.data(), 1, line.size(), f_) == line.size() &&
                        std::fflush(f_) == 0;
        lk.lock();
        if (!ok) failed_ = true;
        last_ = Clock::now();
    }

    std::FILE* f_ = nullptr;
    std::thread flusher_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<app::FrameLanded> pending_;
    int64_t written_ = 0;
    Clock::time_point last_{};
    bool stop_ = false, failed_ = false;
};

}  // namespace

// Declared in sam_main.cpp's dispatcher.
int sam_cli_extract(int argc, char** argv);

int sam_cli_extract(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) {
        usage();
        return 2;
    }
    // The device is NOT set here: a native selection is explicit in the job
    // below, which freezes it before the first decode or model load. Writing
    // SS_VK_DEVICE would leak the choice into unrelated children.
    if (o.validate) set_env("SS_VK_VALIDATION", "1");
    if (o.profile) set_env("SS_PROFILE", "1");

    // Opened first, so a reader learns of every failure from its last line.
    ProgressFile progress;
    if (!o.progress.empty() && !progress.open(o.progress)) {
        std::fprintf(stderr, "%s\n",
                     spirula::i18n::format(spirula::i18n::msg::cli::error_line,
                                           {"could not write " + o.progress}).c_str());
        return 1;
    }
    auto fail = [&](int rc, const std::string& why) {
        progress.close(why, "");
        return rc;
    };

    if (!o.model.empty() && !app::require_model_license(o.model))
        return fail(2, "the model's licence is not accepted");

    const fs::path input(o.input);
    const fs::path base = o.out_dir.empty()
                              ? (input.parent_path() / input.stem() / "images")
                              : fs::path(o.out_dir);

    app::FrameExtractJob job;
    job.input = o.input;
    job.image_dir = base.string();
    job.mask_dir = o.mask_dir;
    // The one device request, from the one flag: the job freezes it before the
    // decoder probe, so the decoder, the masker and the trackers agree.
    job.device = o.device;
    job.skip = o.skip;
    job.keep = o.keep;
    job.max_frames = o.max_frames;
    job.clip_start = o.clip_start;
    job.clip_end = o.clip_end;
    job.quality = o.quality;
    job.rotate = o.rotate;
    job.auto_rotate = o.auto_rotate;
    job.scale = o.scale;
    if (o.lut == "auto") job.lut = app::detect_log_lut(o.input);
    else if (app::find_builtin_lut(o.lut)) job.lut = app::builtin_lut_spec(o.lut);
    else job.lut = o.lut;
    job.track = o.track;
    job.sync_tracks = o.sync;
    job.adaptive = o.adaptive;
    job.adaptive_range = o.adaptive_range;
    job.blur_reject = o.blur_reject;
    job.similar_bits = o.similar;
    job.skipped_dir = o.skipped_dir;
    job.threads = o.threads;
    app::FramePlan plan_in;
    if (!o.plan_in.empty()) {
        app::FramePlan& plan = plan_in;
        std::string err;
        if (app::read_frame_plan(o.plan_in, plan, err)) {
            if (plan.keep != o.keep)
                err = o.plan_in + ": made with --keep " + std::to_string(plan.keep) +
                      ", not " + std::to_string(o.keep);
            else if (plan.ends.empty() && o.adaptive)
                err = o.plan_in + ": holds no adaptive plan";
        }
        if (!err.empty()) {
            std::fprintf(stderr, "%s\n",
                         spirula::i18n::format(spirula::i18n::msg::cli::error_line,
                                               {err}).c_str());
            return fail(2, err);
        }
        job.plan = plan.ends;
        job.keep_existing = true;
    }
    job.decoder = o.decoder;
    job.ffmpeg_exe = o.ffmpeg;
    job.write_overlay = o.overlay;
    // A 360 file is recognised by its packing, not by its name, and only then
    // is there anything for --360 to select.
    if (o.pano_mode != "off") {
        std::vector<std::pair<int, int>> tracks;
#ifdef SS_HAVE_VIDEO
        std::string err;
        tracks = app::video_track_sizes(o.input, err);
#endif
        if (tracks.empty()) {
            app::VideoFacts facts;
            const std::atomic<bool> never{false};
            if (app::ffmpeg_probe_video(o.ffmpeg, o.input, facts, never))
                tracks = facts.tracks;
        }
        app::Pano360Meta meta;
#ifdef SS_TOOL_SFM
        const sfm::VideoProjection pr = sfm::video_projection(o.input);
        meta = app::Pano360Meta{pr.name, pr.mode};
#endif
        if (tracks.size() == 2 && tracks[0] == tracks[1] &&
            app::pano360_detect(2, tracks[0].first, tracks[0].second, meta,
                                job.eac)) {
            o.pano.mode = o.pano_mode == "equirect" ? app::Pano360Mode::Equirect
                                                    : app::Pano360Mode::Faces;
            job.views = app::pano360_views(job.eac, o.pano);
            std::fprintf(stderr, "360: %d x %d tracks -> %zu view(s) of %dx%d\n",
                         job.eac.track_w, job.eac.track_h, job.views.size(),
                         job.views[0].width, job.views[0].height);
        } else {
            job.eac = app::Pano360Layout{};
        }
    }
    if (!o.model.empty()) {
        job.mask.model = o.model;
        job.mask.device = o.device;
        job.mask.text = o.text;
        job.mask.neg_text = o.neg_text;
        job.mask.video = o.mask_mode != "image";
        job.mask.keep_prompted = o.keep_subject.value_or(
            birefnet::find_model_source(o.model) || birefnet::is_checkpoint(o.model));
        job.mask.detector = o.detector;
        job.mask.detector_threshold = o.detector_threshold;
        job.mask.threshold = o.threshold;
        job.mask.nms = o.nms;
        job.mask.dilate_ratio = o.dilate_ratio;
        job.mask.detect_every = o.detect_every;
        job.mask.memory_frames = o.memory_frames;
        job.mask.max_size = o.max_size;
        job.mask.validate = o.validate;
    }

    app::FrameExtractSinks sinks;
    sinks.log = [](const std::string& l) { std::fprintf(stderr, "%s\n", l.c_str()); };
    // One line a chosen frame, for a script: English and stable on purpose.
    FILE* report = nullptr;
    if (!o.report.empty()) {
        std::error_code ec;
        const fs::path dir = fs::path(o.report).parent_path();
        if (!dir.empty()) fs::create_directories(dir, ec);
        report = std::fopen(o.report.c_str(), "w");
        if (!report) {
            std::fprintf(stderr, "%s\n",
                         spirula::i18n::format(spirula::i18n::msg::cli::sam_report_unwritable,
                                               {o.report}).c_str());
            return fail(1, "could not write " + o.report);
        }
    }
    if (report) {
        std::fprintf(report, "frame,edge_width_px,blur_px,verdict\n");
        sinks.judged = [report](const app::FramePick& p, const nn::Image&) {
            static const char* const kVerdict[] = {"kept", "blurry", "similar"};
            std::fprintf(report, "%lld,%.2f,%.2f,%s\n", (long long)p.index, p.width, p.blur,
                         kVerdict[(int)p.verdict]);
        };
    }

    // A reader waiting on a plan that never comes would wait forever, and a
    // resumed run on another plan would give a window a second frame.
    std::string plan_error;
    std::atomic<bool> stop{false};
    if (!o.plan_out.empty() || !o.plan_in.empty()) {
        sinks.cancel = &stop;
        sinks.layout = [&](const app::FramePlan& p) {
            if (!o.plan_in.empty()) {
                const std::string why = app::frame_plan_mismatch(plan_in, p);
                if (!why.empty()) plan_error = o.plan_in + ": " + why;
            }
            if (plan_error.empty() && !o.plan_out.empty()) {
                std::error_code ec;
                const fs::path dir = fs::path(o.plan_out).parent_path();
                if (!dir.empty()) fs::create_directories(dir, ec);
                app::write_text_atomic(o.plan_out, app::frame_plan_json(p), plan_error);
            }
            if (plan_error.empty()) return;
            std::fprintf(stderr, "%s\n",
                         spirula::i18n::format(spirula::i18n::msg::cli::error_line,
                                               {plan_error}).c_str());
            stop = true;
        };
    }
    if (!o.progress.empty())
        sinks.landed = [&](const std::vector<app::FrameLanded>& f) { progress.landed(f); };

    app::FrameExtractStats stats;
    std::string error;
    int rc = 0;
    if (!app::extract_frames(job, sinks, stats, error)) {
        std::fprintf(stderr, "%s\n",
                     spirula::i18n::format(spirula::i18n::msg::cli::error_line,
                                           {error}).c_str());
        rc = 1;
    }
    if (report) std::fclose(report);
    std::printf("%s", app::format_extract_stats(stats, base.string(),
                                                !o.model.empty()).c_str());
    if (stats.write_failures) {
        std::fprintf(stderr, "%s %s\n",
                     spirula::i18n::msg::data::word_warning.get(),
                     spirula::i18n::format(
                         spirula::i18n::msg::data::write_failures,
                         {stats.write_failures}).c_str());
        rc = 1;
    }
    std::string why = !plan_error.empty() ? plan_error : error;
    const bool streaming = !o.progress.empty() || !o.plan_out.empty();
    if (why.empty() && stats.write_failures)
        why = std::to_string(stats.write_failures) + " files could not be written";
    if (why.empty() && streaming && stats.mask_failures)
        why = std::to_string(stats.mask_failures) + " frames could not be masked";
    if (!why.empty()) rc = 1;
    if (!progress.close(why, stats.builtin_decoder ? "vulkan" : "ffmpeg")) rc = 1;
    return rc;
}
