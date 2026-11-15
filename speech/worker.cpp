#include "whisper.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {
constexpr uint32_t max_samples = 16000 * 30;
constexpr uint32_t max_prompt_bytes = 16384;

bool read_exact(void * data, size_t size) {
    return std::fread(data, 1, size, stdin) == size;
}

bool read_u32(uint32_t & value) {
    unsigned char b[4];
    if (!read_exact(b, sizeof(b))) return false;
    value = uint32_t(b[0]) | (uint32_t(b[1]) << 8) | (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24);
    return true;
}

bool reply(uint32_t status, const std::string & text) {
    const uint32_t n = static_cast<uint32_t>(text.size());
    unsigned char header[8];
    for (int i = 0; i < 4; ++i) {
        header[i] = static_cast<unsigned char>(status >> (8 * i));
        header[4 + i] = static_cast<unsigned char>(n >> (8 * i));
    }
    return std::fwrite(header, 1, sizeof(header), stdout) == sizeof(header) &&
           std::fwrite(text.data(), 1, n, stdout) == n && std::fflush(stdout) == 0;
}

whisper_full_params parameters(const std::string & prompt, int beam, bool fallback) {
    auto params = whisper_full_default_params(beam > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY);
    params.n_threads = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
    params.language = "en";
    params.translate = false;
    params.no_context = true;
    params.print_realtime = false;
    params.print_progress = false;
    params.print_timestamps = false;
    params.initial_prompt = prompt.empty() ? nullptr : prompt.c_str();
    if (beam > 1) params.beam_search.beam_size = beam;
    if (!fallback) params.temperature_inc = 0.0f;
    return params;
}
}

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc != 5) {
        std::fprintf(stderr, "usage: stt-worker MODEL cpu|auto BEAM_SIZE fallback|no-fallback\n");
        return 2;
    }
    int beam = 0;
    try {
        size_t used = 0;
        beam = std::stoi(argv[3], &used);
        if (used != std::strlen(argv[3]) || beam < 1 || beam > 10) throw std::invalid_argument("beam");
    } catch (const std::exception &) {
        std::fprintf(stderr, "beam size must be between 1 and 10\n");
        return 2;
    }
    if ((std::strcmp(argv[2], "cpu") && std::strcmp(argv[2], "auto")) ||
        (std::strcmp(argv[4], "fallback") && std::strcmp(argv[4], "no-fallback"))) {
        std::fprintf(stderr, "invalid backend or fallback option\n");
        return 2;
    }
    const bool fallback = std::strcmp(argv[4], "fallback") == 0;
    auto context_params = whisper_context_default_params();
    context_params.use_gpu = std::strcmp(argv[2], "auto") == 0;
    std::unique_ptr<whisper_context, decltype(&whisper_free)> ctx(
        whisper_init_from_file_with_params(argv[1], context_params), whisper_free);
    if (!ctx) {
        reply(1, "Could not load Whisper model");
        return 1;
    }
    const std::string empty;
    std::vector<float> warmup(16000, 0.0f);
    if (whisper_full(ctx.get(), parameters(empty, beam, fallback), warmup.data(), warmup.size()) != 0) {
        reply(1, "Whisper warmup failed");
        return 1;
    }
    if (!reply(0, "ready")) return 1;

    while (true) {
        uint32_t count, prompt_size;
        if (!read_u32(count)) return std::feof(stdin) ? 0 : 1;
        if (!read_u32(prompt_size) || count == 0 || count > max_samples || prompt_size > max_prompt_bytes) {
            std::fprintf(stderr, "Invalid request header\n");
            return 1;
        }
        std::string prompt(prompt_size, '\0');
        std::vector<unsigned char> pcm(count * 2);
        if (!read_exact(prompt.data(), prompt.size()) || !read_exact(pcm.data(), pcm.size())) {
            std::fprintf(stderr, "Incomplete request\n");
            return 1;
        }
        std::vector<float> samples(count);
        for (uint32_t i = 0; i < count; ++i) {
            const uint32_t raw = uint32_t(pcm[2 * i]) | (uint32_t(pcm[2 * i + 1]) << 8);
            const int32_t signed_sample = raw < 32768 ? raw : static_cast<int32_t>(raw) - 65536;
            samples[i] = signed_sample / 32768.0f;
        }
        if (whisper_full(ctx.get(), parameters(prompt, beam, fallback), samples.data(), samples.size()) != 0) {
            if (!reply(1, "Whisper inference failed")) return 1;
            continue;
        }
        std::string text;
        for (int i = 0; i < whisper_full_n_segments(ctx.get()); ++i) {
            text += whisper_full_get_segment_text(ctx.get(), i);
        }
        if (!reply(0, text)) return 1;
    }
}
