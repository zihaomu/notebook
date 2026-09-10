#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <va/va.h>
#include <va/va_drmcommon.h>
#include <drm_fourcc.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavutil/opt.h>
}

namespace py = pybind11;

namespace {

constexpr int kMaxDetections = 300;
constexpr int kMaxStatusLines = 16;
constexpr int kMaxOverlays = kMaxDetections + kMaxStatusLines;
constexpr int kLabelCapacity = 32;
constexpr int kFontScale = 2;
constexpr int kGlyphWidth = 5;
constexpr int kGlyphHeight = 7;
constexpr int kGlyphAdvance = 6;
constexpr int kLabelPadding = 3;
constexpr int kBorderThickness = 3;

struct GpuDetection {
    int x1;
    int y1;
    int x2;
    int y2;
    int class_id;
    int draw_box;
    int label_length;
    char label[kLabelCapacity];
};

std::string ffmpeg_error(int code) {
    char buffer[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(code, buffer, sizeof(buffer));
    return buffer;
}

void check_ffmpeg(int code, const char* expression) {
    if (code < 0) {
        throw std::runtime_error(std::string(expression) + ": " + ffmpeg_error(code));
    }
}

void check_va(VAStatus status, const char* expression) {
    if (status != VA_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(expression) + ": " + vaErrorStr(status));
    }
}

void check_hip(hipError_t status, const char* expression) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(expression) + ": " + hipGetErrorString(status));
    }
}

#define CHECK_FFMPEG(expression) check_ffmpeg((expression), #expression)
#define CHECK_VA(expression) check_va((expression), #expression)
#define CHECK_HIP(expression) check_hip((expression), #expression)
#define GLYPH(r0, r1, r2, r3, r4, r5, r6) \
    (static_cast<uint64_t>(r0) | (static_cast<uint64_t>(r1) << 5) | \
     (static_cast<uint64_t>(r2) << 10) | (static_cast<uint64_t>(r3) << 15) | \
     (static_cast<uint64_t>(r4) << 20) | (static_cast<uint64_t>(r5) << 25) | \
     (static_cast<uint64_t>(r6) << 30))

__device__ uint64_t glyph_bits(char character) {
    switch (character) {
        case 'A': return GLYPH(14, 17, 17, 31, 17, 17, 17);
        case 'B': return GLYPH(30, 17, 17, 30, 17, 17, 30);
        case 'C': return GLYPH(14, 17, 16, 16, 16, 17, 14);
        case 'D': return GLYPH(30, 17, 17, 17, 17, 17, 30);
        case 'E': return GLYPH(31, 16, 16, 30, 16, 16, 31);
        case 'F': return GLYPH(31, 16, 16, 30, 16, 16, 16);
        case 'G': return GLYPH(14, 17, 16, 23, 17, 17, 15);
        case 'H': return GLYPH(17, 17, 17, 31, 17, 17, 17);
        case 'I': return GLYPH(14, 4, 4, 4, 4, 4, 14);
        case 'J': return GLYPH(7, 2, 2, 2, 2, 18, 12);
        case 'K': return GLYPH(17, 18, 20, 24, 20, 18, 17);
        case 'L': return GLYPH(16, 16, 16, 16, 16, 16, 31);
        case 'M': return GLYPH(17, 27, 21, 21, 17, 17, 17);
        case 'N': return GLYPH(17, 25, 21, 19, 17, 17, 17);
        case 'O': return GLYPH(14, 17, 17, 17, 17, 17, 14);
        case 'P': return GLYPH(30, 17, 17, 30, 16, 16, 16);
        case 'Q': return GLYPH(14, 17, 17, 17, 21, 18, 13);
        case 'R': return GLYPH(30, 17, 17, 30, 20, 18, 17);
        case 'S': return GLYPH(15, 16, 16, 14, 1, 1, 30);
        case 'T': return GLYPH(31, 4, 4, 4, 4, 4, 4);
        case 'U': return GLYPH(17, 17, 17, 17, 17, 17, 14);
        case 'V': return GLYPH(17, 17, 17, 17, 17, 10, 4);
        case 'W': return GLYPH(17, 17, 17, 21, 21, 21, 10);
        case 'X': return GLYPH(17, 17, 10, 4, 10, 17, 17);
        case 'Y': return GLYPH(17, 17, 10, 4, 4, 4, 4);
        case 'Z': return GLYPH(31, 1, 2, 4, 8, 16, 31);
        case '0': return GLYPH(14, 17, 19, 21, 25, 17, 14);
        case '1': return GLYPH(4, 12, 4, 4, 4, 4, 14);
        case '2': return GLYPH(14, 17, 1, 2, 4, 8, 31);
        case '3': return GLYPH(30, 1, 1, 14, 1, 1, 30);
        case '4': return GLYPH(2, 6, 10, 18, 31, 2, 2);
        case '5': return GLYPH(31, 16, 16, 30, 1, 1, 30);
        case '6': return GLYPH(14, 16, 16, 30, 17, 17, 14);
        case '7': return GLYPH(31, 1, 2, 4, 8, 8, 8);
        case '8': return GLYPH(14, 17, 17, 14, 17, 17, 14);
        case '9': return GLYPH(14, 17, 17, 15, 1, 1, 14);
        case '.': return GLYPH(0, 0, 0, 0, 0, 12, 12);
        case '-': return GLYPH(0, 0, 0, 31, 0, 0, 0);
        case '%': return GLYPH(17, 2, 4, 8, 16, 17, 0);
        case '/': return GLYPH(1, 2, 2, 4, 8, 8, 16);
        case ':': return GLYPH(0, 12, 12, 0, 12, 12, 0);
        default: return 0;
    }
}

__device__ uint8_t clamp_byte(int value) {
    return static_cast<uint8_t>(max(0, min(255, value)));
}

__device__ void rgb_to_yuv(uint8_t red, uint8_t green, uint8_t blue, uint8_t& y, uint8_t& u, uint8_t& v) {
    y = clamp_byte(((66 * red + 129 * green + 25 * blue + 128) >> 8) + 16);
    u = clamp_byte(((-38 * red - 74 * green + 112 * blue + 128) >> 8) + 128);
    v = clamp_byte(((112 * red - 94 * green - 18 * blue + 128) >> 8) + 128);
}

__global__ void rgb_to_nv12(
    const uint8_t* rgb,
    uint32_t rgb_stride,
    uint8_t* surface,
    uint32_t width,
    uint32_t height,
    uint32_t y_offset,
    uint32_t y_pitch,
    uint32_t uv_offset,
    uint32_t uv_pitch) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y_position = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y_position >= height) {
        return;
    }
    const uint8_t* pixel = rgb + y_position * rgb_stride + x * 3;
    uint8_t luma = 0;
    uint8_t chroma_u = 0;
    uint8_t chroma_v = 0;
    rgb_to_yuv(pixel[0], pixel[1], pixel[2], luma, chroma_u, chroma_v);
    surface[y_offset + y_position * y_pitch + x] = luma;

    if ((x & 1u) == 0 && (y_position & 1u) == 0) {
        int red = 0;
        int green = 0;
        int blue = 0;
        for (uint32_t dy = 0; dy < 2; ++dy) {
            for (uint32_t dx = 0; dx < 2; ++dx) {
                const uint8_t* sample = rgb + (y_position + dy) * rgb_stride + (x + dx) * 3;
                red += sample[0];
                green += sample[1];
                blue += sample[2];
            }
        }
        rgb_to_yuv(
            static_cast<uint8_t>(red / 4),
            static_cast<uint8_t>(green / 4),
            static_cast<uint8_t>(blue / 4),
            luma,
            chroma_u,
            chroma_v);
        uint8_t* uv = surface + uv_offset + (y_position / 2u) * uv_pitch + x;
        uv[0] = chroma_u;
        uv[1] = chroma_v;
    }
}

__device__ void overlay_pixel(
    uint8_t* surface,
    int x,
    int y,
    int width,
    int height,
    uint32_t y_offset,
    uint32_t y_pitch,
    uint32_t uv_offset,
    uint32_t uv_pitch,
    uint8_t luma,
    uint8_t chroma_u,
    uint8_t chroma_v) {
    if (x < 0 || y < 0 || x >= width || y >= height) {
        return;
    }
    surface[y_offset + static_cast<uint32_t>(y) * y_pitch + static_cast<uint32_t>(x)] = luma;
    uint8_t* uv = surface + uv_offset + static_cast<uint32_t>(y / 2) * uv_pitch + static_cast<uint32_t>(x & ~1);
    uv[0] = chroma_u;
    uv[1] = chroma_v;
}

__global__ void draw_detections_nv12(
    uint8_t* surface,
    int width,
    int height,
    uint32_t y_offset,
    uint32_t y_pitch,
    uint32_t uv_offset,
    uint32_t uv_pitch,
    const GpuDetection* detections,
    int detection_count) {
    const int detection_index = blockIdx.x;
    if (detection_index >= detection_count) {
        return;
    }
    const GpuDetection detection = detections[detection_index];
    const int x1 = max(0, min(width - 1, detection.x1));
    const int y1 = max(0, min(height - 1, detection.y1));
    const int x2 = max(x1 + 1, min(width, detection.x2));
    const int y2 = max(y1 + 1, min(height, detection.y2));
    const int box_width = x2 - x1;
    const int box_height = y2 - y1;

    const int seed = detection.class_id * 73 + 41;
    const uint8_t red = detection.class_id < 0 ? 48 : static_cast<uint8_t>(64 + (seed * 3) % 192);
    const uint8_t green = detection.class_id < 0 ? 224 : static_cast<uint8_t>(64 + (seed * 5) % 192);
    const uint8_t blue = detection.class_id < 0 ? 80 : static_cast<uint8_t>(64 + (seed * 7) % 192);
    uint8_t color_y = 0;
    uint8_t color_u = 0;
    uint8_t color_v = 0;
    rgb_to_yuv(red, green, blue, color_y, color_u, color_v);

    if (detection.draw_box) {
        const int horizontal_pixels = box_width * kBorderThickness;
        for (int position = threadIdx.x; position < horizontal_pixels; position += blockDim.x) {
            const int x = x1 + position % box_width;
            const int offset = position / box_width;
            overlay_pixel(surface, x, y1 + offset, width, height, y_offset, y_pitch, uv_offset, uv_pitch, color_y, color_u, color_v);
            overlay_pixel(surface, x, y2 - 1 - offset, width, height, y_offset, y_pitch, uv_offset, uv_pitch, color_y, color_u, color_v);
        }
        const int vertical_pixels = box_height * kBorderThickness;
        for (int position = threadIdx.x; position < vertical_pixels; position += blockDim.x) {
            const int y = y1 + position % box_height;
            const int offset = position / box_height;
            overlay_pixel(surface, x1 + offset, y, width, height, y_offset, y_pitch, uv_offset, uv_pitch, color_y, color_u, color_v);
            overlay_pixel(surface, x2 - 1 - offset, y, width, height, y_offset, y_pitch, uv_offset, uv_pitch, color_y, color_u, color_v);
        }
    }

    const int label_width = min(width - x1, detection.label_length * kGlyphAdvance * kFontScale + 2 * kLabelPadding);
    const int label_height = kGlyphHeight * kFontScale + 2 * kLabelPadding;
    const int label_y = detection.draw_box
        ? (y1 >= label_height ? y1 - label_height : min(height - label_height, y1 + kBorderThickness))
        : min(height - label_height, y1);
    const int label_pixels = max(0, label_width * label_height);
    for (int position = threadIdx.x; position < label_pixels; position += blockDim.x) {
        const int local_x = position % label_width;
        const int local_y = position / label_width;
        const int x = x1 + local_x;
        const int y = label_y + local_y;
        uint8_t luma = static_cast<uint8_t>(max(16, static_cast<int>(color_y) / 2));
        uint8_t chroma_u = color_u;
        uint8_t chroma_v = color_v;

        const int text_x = local_x - kLabelPadding;
        const int text_y = local_y - kLabelPadding;
        if (text_x >= 0 && text_y >= 0) {
            const int character_index = text_x / (kGlyphAdvance * kFontScale);
            const int glyph_x = (text_x / kFontScale) % kGlyphAdvance;
            const int glyph_y = text_y / kFontScale;
            if (character_index < detection.label_length && glyph_x < kGlyphWidth && glyph_y < kGlyphHeight) {
                const uint64_t bits = glyph_bits(detection.label[character_index]);
                const int glyph_bit = kGlyphWidth - 1 - glyph_x;
                if (((bits >> (glyph_y * kGlyphWidth)) >> glyph_bit) & 1u) {
                    luma = 235;
                    chroma_u = 128;
                    chroma_v = 128;
                }
            }
        }
        overlay_pixel(surface, x, y, width, height, y_offset, y_pitch, uv_offset, uv_pitch, luma, chroma_u, chroma_v);
    }
}

struct SurfaceMapping {
    hipExternalMemory_t external_memory = nullptr;
    void* pointer = nullptr;
    uint32_t size = 0;
    uint32_t y_offset = 0;
    uint32_t y_pitch = 0;
    uint32_t uv_offset = 0;
    uint32_t uv_pitch = 0;
};

class HipVaapiEncoder {
public:
    HipVaapiEncoder(
        const std::string& output_path,
        const std::string& render_node,
        int width,
        int height,
        double fps,
        int qp = 24)
        : output_path_(output_path),
          render_node_(render_node),
          width_(width),
          height_(height),
          fps_(fps),
          qp_(qp) {
        if (width_ <= 0 || height_ <= 0 || (width_ & 1) || (height_ & 1) || fps_ <= 0) {
            throw std::invalid_argument("width/height must be positive and even; fps must be positive");
        }
        initialize();
        CHECK_HIP(hipMalloc(&device_detections_, sizeof(GpuDetection) * kMaxOverlays));
    }

    ~HipVaapiEncoder() {
        try {
            close();
        } catch (...) {
        }
    }

    HipVaapiEncoder(const HipVaapiEncoder&) = delete;
    HipVaapiEncoder& operator=(const HipVaapiEncoder&) = delete;

    void write(
        uintptr_t rgb_pointer,
        size_t rgb_stride,
        const std::vector<std::array<float, 6>>& detections,
        const std::vector<std::string>& labels,
        uintptr_t stream_pointer = 0,
        const std::vector<std::string>& status_lines = {}) {
        if (closed_) {
            throw std::runtime_error("encoder is closed");
        }
        if (!rgb_pointer || rgb_stride < static_cast<size_t>(width_ * 3)) {
            throw std::invalid_argument("invalid RGB pointer or stride");
        }
        if (detections.size() != labels.size()) {
            throw std::invalid_argument("detections and labels must have equal length");
        }
        if (detections.size() > kMaxDetections || status_lines.size() > kMaxStatusLines) {
            throw std::invalid_argument("too many detections or status lines");
        }

        AVFrame* frame = av_frame_alloc();
        if (!frame) {
            throw std::runtime_error("av_frame_alloc failed");
        }
        try {
            frame->format = AV_PIX_FMT_VAAPI;
            frame->width = width_;
            frame->height = height_;
            CHECK_FFMPEG(av_hwframe_get_buffer(frames_ref_, frame, 0));
            const auto surface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(frame->data[3]));
            SurfaceMapping& mapping = map_surface(surface);
            CHECK_VA(vaSyncSurface(va_display_, surface));

            std::vector<GpuDetection> host_detections(detections.size() + status_lines.size());
            for (size_t index = 0; index < detections.size(); ++index) {
                GpuDetection& target = host_detections[index];
                target.x1 = static_cast<int>(detections[index][0]);
                target.y1 = static_cast<int>(detections[index][1]);
                target.x2 = static_cast<int>(detections[index][2]);
                target.y2 = static_cast<int>(detections[index][3]);
                target.class_id = static_cast<int>(detections[index][5]);
                target.draw_box = 1;
                std::string label = labels[index];
                std::transform(label.begin(), label.end(), label.begin(), [](unsigned char value) {
                    return static_cast<char>(std::toupper(value));
                });
                target.label_length = std::min(static_cast<int>(label.size()), kLabelCapacity - 1);
                std::memset(target.label, 0, sizeof(target.label));
                std::memcpy(target.label, label.data(), target.label_length);
            }
            for (size_t index = 0; index < status_lines.size(); ++index) {
                GpuDetection& target = host_detections[detections.size() + index];
                target.x1 = 10;
                target.y1 = 10 + static_cast<int>(index) * (kGlyphHeight * kFontScale + 2 * kLabelPadding + 4);
                target.x2 = target.x1 + 1;
                target.y2 = target.y1 + 1;
                target.class_id = -1;
                target.draw_box = 0;
                std::string label = status_lines[index];
                std::transform(label.begin(), label.end(), label.begin(), [](unsigned char value) {
                    return static_cast<char>(std::toupper(value));
                });
                target.label_length = std::min(static_cast<int>(label.size()), kLabelCapacity - 1);
                std::memset(target.label, 0, sizeof(target.label));
                std::memcpy(target.label, label.data(), target.label_length);
            }

            const hipStream_t stream = reinterpret_cast<hipStream_t>(stream_pointer);
            if (!host_detections.empty()) {
                CHECK_HIP(hipMemcpyAsync(
                    device_detections_,
                    host_detections.data(),
                    host_detections.size() * sizeof(GpuDetection),
                    hipMemcpyHostToDevice,
                    stream));
            }
            dim3 block(16, 16);
            dim3 grid(
                (width_ + static_cast<int>(block.x) - 1) / static_cast<int>(block.x),
                (height_ + static_cast<int>(block.y) - 1) / static_cast<int>(block.y));
            hipLaunchKernelGGL(
                rgb_to_nv12,
                grid,
                block,
                0,
                stream,
                reinterpret_cast<const uint8_t*>(rgb_pointer),
                static_cast<uint32_t>(rgb_stride),
                static_cast<uint8_t*>(mapping.pointer),
                static_cast<uint32_t>(width_),
                static_cast<uint32_t>(height_),
                mapping.y_offset,
                mapping.y_pitch,
                mapping.uv_offset,
                mapping.uv_pitch);
            CHECK_HIP(hipGetLastError());
            if (!host_detections.empty()) {
                hipLaunchKernelGGL(
                    draw_detections_nv12,
                    dim3(static_cast<uint32_t>(host_detections.size())),
                    dim3(256),
                    0,
                    stream,
                    static_cast<uint8_t*>(mapping.pointer),
                    width_,
                    height_,
                    mapping.y_offset,
                    mapping.y_pitch,
                    mapping.uv_offset,
                    mapping.uv_pitch,
                    device_detections_,
                    static_cast<int>(host_detections.size()));
                CHECK_HIP(hipGetLastError());
            }
            CHECK_HIP(hipStreamSynchronize(stream));

            frame->pts = frame_index_++;
            send_frame(frame);
            av_frame_free(&frame);
        } catch (...) {
            av_frame_free(&frame);
            throw;
        }
    }

    void close() {
        if (closed_) {
            return;
        }
        closed_ = true;
        if (encoder_) {
            CHECK_FFMPEG(avcodec_send_frame(encoder_, nullptr));
            receive_packets();
        }
        if (format_) {
            CHECK_FFMPEG(av_write_trailer(format_));
            if (!(format_->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&format_->pb);
            }
        }
        if (device_detections_) {
            CHECK_HIP(hipFree(device_detections_));
            device_detections_ = nullptr;
        }
        for (auto& item : mappings_) {
            if (item.second.pointer) {
                CHECK_HIP(hipFree(item.second.pointer));
            }
            if (item.second.external_memory) {
                CHECK_HIP(hipDestroyExternalMemory(item.second.external_memory));
            }
        }
        mappings_.clear();
        avcodec_free_context(&encoder_);
        if (format_) {
            avformat_free_context(format_);
            format_ = nullptr;
        }
        av_buffer_unref(&frames_ref_);
        av_buffer_unref(&device_ref_);
    }

    int frame_count() const {
        return frame_index_;
    }

    py::dict info() const {
        py::dict result;
        result["output"] = output_path_;
        result["render_node"] = render_node_;
        result["width"] = width_;
        result["height"] = height_;
        result["fps"] = fps_;
        result["frames"] = frame_index_;
        result["mapped_surfaces"] = mappings_.size();
        result["pixel_format"] = "NV12";
        result["transport"] = "HIP -> DRM PRIME VAAPI surface -> h264_vaapi";
        return result;
    }

private:
    void initialize() {
        CHECK_HIP(hipSetDevice(0));
        CHECK_FFMPEG(av_hwdevice_ctx_create(
            &device_ref_, AV_HWDEVICE_TYPE_VAAPI, render_node_.c_str(), nullptr, 0));
        auto* device = reinterpret_cast<AVHWDeviceContext*>(device_ref_->data);
        auto* va_device = reinterpret_cast<AVVAAPIDeviceContext*>(device->hwctx);
        va_display_ = va_device->display;

        frames_ref_ = av_hwframe_ctx_alloc(device_ref_);
        if (!frames_ref_) {
            throw std::runtime_error("av_hwframe_ctx_alloc failed");
        }
        auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref_->data);
        frames->format = AV_PIX_FMT_VAAPI;
        frames->sw_format = AV_PIX_FMT_NV12;
        frames->width = width_;
        frames->height = height_;
        frames->initial_pool_size = 16;
        linear_modifier_ = DRM_FORMAT_MOD_LINEAR;
        modifier_list_.num_modifiers = 1;
        modifier_list_.modifiers = &linear_modifier_;
        modifier_attribute_.type = VASurfaceAttribDRMFormatModifiers;
        modifier_attribute_.flags = VA_SURFACE_ATTRIB_SETTABLE;
        modifier_attribute_.value.type = VAGenericValueTypePointer;
        modifier_attribute_.value.value.p = &modifier_list_;
        auto* va_frames = reinterpret_cast<AVVAAPIFramesContext*>(frames->hwctx);
        va_frames->attributes = &modifier_attribute_;
        va_frames->nb_attributes = 1;
        CHECK_FFMPEG(av_hwframe_ctx_init(frames_ref_));

        const AVCodec* codec = avcodec_find_encoder_by_name("h264_vaapi");
        if (!codec) {
            throw std::runtime_error("h264_vaapi encoder is unavailable");
        }
        CHECK_FFMPEG(avformat_alloc_output_context2(
            &format_, nullptr, nullptr, output_path_.c_str()));
        if (!format_) {
            throw std::runtime_error("avformat_alloc_output_context2 returned null");
        }
        stream_ = avformat_new_stream(format_, nullptr);
        if (!stream_) {
            throw std::runtime_error("avformat_new_stream failed");
        }
        encoder_ = avcodec_alloc_context3(codec);
        if (!encoder_) {
            throw std::runtime_error("avcodec_alloc_context3 failed");
        }
        encoder_->width = width_;
        encoder_->height = height_;
        encoder_->time_base = av_d2q(1.0 / fps_, 1000000);
        encoder_->framerate = av_d2q(fps_, 1000000);
        encoder_->pix_fmt = AV_PIX_FMT_VAAPI;
        encoder_->gop_size = std::max(1, static_cast<int>(fps_));
        encoder_->max_b_frames = 0;
        encoder_->hw_frames_ctx = av_buffer_ref(frames_ref_);
        if (format_->oformat->flags & AVFMT_GLOBALHEADER) {
            encoder_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        CHECK_FFMPEG(av_opt_set_int(encoder_->priv_data, "qp", qp_, 0));
        CHECK_FFMPEG(avcodec_open2(encoder_, codec, nullptr));
        CHECK_FFMPEG(avcodec_parameters_from_context(stream_->codecpar, encoder_));
        stream_->time_base = encoder_->time_base;
        if (!(format_->oformat->flags & AVFMT_NOFILE)) {
            CHECK_FFMPEG(avio_open(&format_->pb, output_path_.c_str(), AVIO_FLAG_WRITE));
        }
        CHECK_FFMPEG(avformat_write_header(format_, nullptr));
    }

    SurfaceMapping& map_surface(VASurfaceID surface) {
        auto existing = mappings_.find(surface);
        if (existing != mappings_.end()) {
            return existing->second;
        }
        VADRMPRIMESurfaceDescriptor descriptor{};
        CHECK_VA(vaExportSurfaceHandle(
            va_display_,
            surface,
            VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
            VA_EXPORT_SURFACE_READ_WRITE | VA_EXPORT_SURFACE_COMPOSED_LAYERS,
            &descriptor));
        if (descriptor.num_objects != 1 || descriptor.num_layers != 1 ||
            descriptor.layers[0].num_planes != 2 || descriptor.fourcc != VA_FOURCC_NV12 ||
            descriptor.objects[0].drm_format_modifier != DRM_FORMAT_MOD_LINEAR) {
            for (uint32_t index = 0; index < descriptor.num_objects; ++index) {
                ::close(descriptor.objects[index].fd);
            }
            throw std::runtime_error("expected one linear NV12 DRM PRIME object");
        }

        SurfaceMapping mapping{};
        mapping.size = descriptor.objects[0].size;
        mapping.y_offset = descriptor.layers[0].offset[0];
        mapping.y_pitch = descriptor.layers[0].pitch[0];
        mapping.uv_offset = descriptor.layers[0].offset[1];
        mapping.uv_pitch = descriptor.layers[0].pitch[1];
        hipExternalMemoryHandleDesc handle{};
        handle.type = hipExternalMemoryHandleTypeOpaqueFd;
        handle.handle.fd = descriptor.objects[0].fd;
        handle.size = descriptor.objects[0].size;
        CHECK_HIP(hipImportExternalMemory(&mapping.external_memory, &handle));
        descriptor.objects[0].fd = -1;
        hipExternalMemoryBufferDesc buffer{};
        buffer.offset = 0;
        buffer.size = mapping.size;
        CHECK_HIP(hipExternalMemoryGetMappedBuffer(
            &mapping.pointer, mapping.external_memory, &buffer));
        for (uint32_t index = 0; index < descriptor.num_objects; ++index) {
            if (descriptor.objects[index].fd >= 0) {
                ::close(descriptor.objects[index].fd);
            }
        }
        auto inserted = mappings_.emplace(surface, mapping);
        return inserted.first->second;
    }

    void send_frame(AVFrame* frame) {
        int status = avcodec_send_frame(encoder_, frame);
        if (status == AVERROR(EAGAIN)) {
            receive_packets();
            status = avcodec_send_frame(encoder_, frame);
        }
        CHECK_FFMPEG(status);
        receive_packets();
    }

    void receive_packets() {
        AVPacket* packet = av_packet_alloc();
        if (!packet) {
            throw std::runtime_error("av_packet_alloc failed");
        }
        try {
            while (true) {
                const int status = avcodec_receive_packet(encoder_, packet);
                if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
                    break;
                }
                CHECK_FFMPEG(status);
                av_packet_rescale_ts(packet, encoder_->time_base, stream_->time_base);
                packet->duration = av_rescale_q(1, encoder_->time_base, stream_->time_base);
                packet->stream_index = stream_->index;
                CHECK_FFMPEG(av_interleaved_write_frame(format_, packet));
                av_packet_unref(packet);
            }
            av_packet_free(&packet);
        } catch (...) {
            av_packet_free(&packet);
            throw;
        }
    }

    std::string output_path_;
    std::string render_node_;
    int width_;
    int height_;
    double fps_;
    int qp_;
    int frame_index_ = 0;
    bool closed_ = false;
    AVBufferRef* device_ref_ = nullptr;
    AVBufferRef* frames_ref_ = nullptr;
    AVFormatContext* format_ = nullptr;
    AVCodecContext* encoder_ = nullptr;
    AVStream* stream_ = nullptr;
    VADisplay va_display_ = nullptr;
    uint64_t linear_modifier_ = DRM_FORMAT_MOD_LINEAR;
    VADRMFormatModifierList modifier_list_{};
    VASurfaceAttrib modifier_attribute_{};
    GpuDetection* device_detections_ = nullptr;
    std::unordered_map<VASurfaceID, SurfaceMapping> mappings_;
};

}  // namespace

PYBIND11_MODULE(hip_vaapi_bridge, module) {
    module.doc() = "HIP RGB overlay and direct VAAPI H.264 encoding";
    py::class_<HipVaapiEncoder>(module, "HipVaapiEncoder")
        .def(
            py::init<const std::string&, const std::string&, int, int, double, int>(),
            py::arg("output_path"),
            py::arg("render_node"),
            py::arg("width"),
            py::arg("height"),
            py::arg("fps"),
            py::arg("qp") = 24)
        .def(
            "write",
            &HipVaapiEncoder::write,
            py::arg("rgb_pointer"),
            py::arg("rgb_stride"),
            py::arg("detections"),
            py::arg("labels"),
            py::arg("stream_pointer") = 0,
            py::arg("status_lines") = std::vector<std::string>{},
            py::call_guard<py::gil_scoped_release>())
        .def("close", &HipVaapiEncoder::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("frame_count", &HipVaapiEncoder::frame_count)
        .def("info", &HipVaapiEncoder::info);
}
