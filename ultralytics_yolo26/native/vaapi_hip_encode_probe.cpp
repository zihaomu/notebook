#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>
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

namespace {

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

__global__ void fill_nv12_pattern(
    uint8_t* base,
    uint32_t width,
    uint32_t height,
    uint32_t y_offset,
    uint32_t y_pitch,
    uint32_t uv_offset,
    uint32_t uv_pitch,
    int frame_index) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    constexpr uint8_t levels[4] = {32, 80, 144, 220};
    uint8_t luma = levels[min(3u, (x * 4u) / width)];
    const uint32_t box_x = static_cast<uint32_t>((frame_index * 23) % (width - 160));
    const uint32_t box_y = height / 2 - 60;
    if (x >= box_x && x < box_x + 160 && y >= box_y && y < box_y + 120) {
        luma = 235;
    }
    base[y_offset + y * y_pitch + x] = luma;

    if ((x & 1u) == 0 && (y & 1u) == 0) {
        uint8_t* uv = base + uv_offset + (y / 2u) * uv_pitch + x;
        uv[0] = 128;
        uv[1] = 128;
    }
}

void write_packet(AVFormatContext* format, AVCodecContext* encoder, AVStream* stream) {
    AVPacket* packet = av_packet_alloc();
    if (!packet) {
        throw std::runtime_error("av_packet_alloc failed");
    }
    while (true) {
        const int status = avcodec_receive_packet(encoder, packet);
        if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) {
            break;
        }
        CHECK_FFMPEG(status);
        av_packet_rescale_ts(packet, encoder->time_base, stream->time_base);
        packet->stream_index = stream->index;
        CHECK_FFMPEG(av_interleaved_write_frame(format, packet));
        av_packet_unref(packet);
    }
    av_packet_free(&packet);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <render-node> <output.mp4> [frames]\n";
        return 2;
    }
    const char* render_node = argv[1];
    const char* output_path = argv[2];
    const int frame_count = argc > 3 ? std::stoi(argv[3]) : 30;
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr int fps = 30;

    AVBufferRef* device_ref = nullptr;
    AVBufferRef* frames_ref = nullptr;
    AVFormatContext* format = nullptr;
    AVCodecContext* encoder = nullptr;
    AVStream* stream = nullptr;

    try {
        CHECK_HIP(hipSetDevice(0));
        CHECK_FFMPEG(av_hwdevice_ctx_create(
            &device_ref, AV_HWDEVICE_TYPE_VAAPI, render_node, nullptr, 0));

        frames_ref = av_hwframe_ctx_alloc(device_ref);
        if (!frames_ref) {
            throw std::runtime_error("av_hwframe_ctx_alloc failed");
        }
        auto* frames = reinterpret_cast<AVHWFramesContext*>(frames_ref->data);
        frames->format = AV_PIX_FMT_VAAPI;
        frames->sw_format = AV_PIX_FMT_NV12;
        frames->width = width;
        frames->height = height;
        frames->initial_pool_size = 16;

        uint64_t linear_modifier = DRM_FORMAT_MOD_LINEAR;
        VADRMFormatModifierList modifier_list{};
        modifier_list.num_modifiers = 1;
        modifier_list.modifiers = &linear_modifier;
        VASurfaceAttrib modifier_attribute{};
        modifier_attribute.type = VASurfaceAttribDRMFormatModifiers;
        modifier_attribute.flags = VA_SURFACE_ATTRIB_SETTABLE;
        modifier_attribute.value.type = VAGenericValueTypePointer;
        modifier_attribute.value.value.p = &modifier_list;
        auto* va_frames = reinterpret_cast<AVVAAPIFramesContext*>(frames->hwctx);
        va_frames->attributes = &modifier_attribute;
        va_frames->nb_attributes = 1;
        CHECK_FFMPEG(av_hwframe_ctx_init(frames_ref));

        const AVCodec* codec = avcodec_find_encoder_by_name("h264_vaapi");
        if (!codec) {
            throw std::runtime_error("h264_vaapi encoder is unavailable");
        }
        CHECK_FFMPEG(avformat_alloc_output_context2(&format, nullptr, nullptr, output_path));
        if (!format) {
            throw std::runtime_error("avformat_alloc_output_context2 returned null");
        }
        stream = avformat_new_stream(format, nullptr);
        if (!stream) {
            throw std::runtime_error("avformat_new_stream failed");
        }

        encoder = avcodec_alloc_context3(codec);
        if (!encoder) {
            throw std::runtime_error("avcodec_alloc_context3 failed");
        }
        encoder->width = width;
        encoder->height = height;
        encoder->time_base = AVRational{1, fps};
        encoder->framerate = AVRational{fps, 1};
        encoder->pix_fmt = AV_PIX_FMT_VAAPI;
        encoder->gop_size = fps;
        encoder->max_b_frames = 0;
        encoder->hw_frames_ctx = av_buffer_ref(frames_ref);
        if (format->oformat->flags & AVFMT_GLOBALHEADER) {
            encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }
        CHECK_FFMPEG(av_opt_set_int(encoder->priv_data, "qp", 24, 0));
        CHECK_FFMPEG(avcodec_open2(encoder, codec, nullptr));
        CHECK_FFMPEG(avcodec_parameters_from_context(stream->codecpar, encoder));
        stream->time_base = encoder->time_base;
        if (!(format->oformat->flags & AVFMT_NOFILE)) {
            CHECK_FFMPEG(avio_open(&format->pb, output_path, AVIO_FLAG_WRITE));
        }
        CHECK_FFMPEG(avformat_write_header(format, nullptr));

        auto* device = reinterpret_cast<AVHWDeviceContext*>(device_ref->data);
        auto* va_device = reinterpret_cast<AVVAAPIDeviceContext*>(device->hwctx);
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

        for (int index = 0; index < frame_count; ++index) {
            AVFrame* frame = av_frame_alloc();
            if (!frame) {
                throw std::runtime_error("av_frame_alloc failed");
            }
            frame->format = AV_PIX_FMT_VAAPI;
            frame->width = width;
            frame->height = height;
            CHECK_FFMPEG(av_hwframe_get_buffer(frames_ref, frame, 0));
            const auto surface = static_cast<VASurfaceID>(reinterpret_cast<uintptr_t>(frame->data[3]));

            VADRMPRIMESurfaceDescriptor descriptor{};
            CHECK_VA(vaExportSurfaceHandle(
                va_device->display,
                surface,
                VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2,
                VA_EXPORT_SURFACE_READ_WRITE | VA_EXPORT_SURFACE_COMPOSED_LAYERS,
                &descriptor));
            if (descriptor.num_objects != 1 || descriptor.num_layers != 1 ||
                descriptor.layers[0].num_planes != 2 ||
                descriptor.objects[0].drm_format_modifier != DRM_FORMAT_MOD_LINEAR) {
                throw std::runtime_error("VAAPI encoder surface is not a single-object linear NV12 surface");
            }

            const int imported_fd = dup(descriptor.objects[0].fd);
            if (imported_fd < 0) {
                throw std::runtime_error("dup DRM PRIME fd failed");
            }
            hipExternalMemoryHandleDesc handle{};
            handle.type = hipExternalMemoryHandleTypeOpaqueFd;
            handle.handle.fd = imported_fd;
            handle.size = descriptor.objects[0].size;
            hipExternalMemory_t external_memory = nullptr;
            CHECK_HIP(hipImportExternalMemory(&external_memory, &handle));
            hipExternalMemoryBufferDesc buffer{};
            buffer.offset = 0;
            buffer.size = descriptor.objects[0].size;
            void* mapped = nullptr;
            CHECK_HIP(hipExternalMemoryGetMappedBuffer(&mapped, external_memory, &buffer));

            hipLaunchKernelGGL(
                fill_nv12_pattern,
                grid,
                block,
                0,
                nullptr,
                static_cast<uint8_t*>(mapped),
                descriptor.width,
                descriptor.height,
                descriptor.layers[0].offset[0],
                descriptor.layers[0].pitch[0],
                descriptor.layers[0].offset[1],
                descriptor.layers[0].pitch[1],
                index);
            CHECK_HIP(hipGetLastError());
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipFree(mapped));
            CHECK_HIP(hipDestroyExternalMemory(external_memory));
            for (uint32_t object = 0; object < descriptor.num_objects; ++object) {
                close(descriptor.objects[object].fd);
            }

            frame->pts = index;
            CHECK_FFMPEG(avcodec_send_frame(encoder, frame));
            av_frame_free(&frame);
            write_packet(format, encoder, stream);
        }

        CHECK_FFMPEG(avcodec_send_frame(encoder, nullptr));
        write_packet(format, encoder, stream);
        CHECK_FFMPEG(av_write_trailer(format));

        if (!(format->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format->pb);
        }
        avcodec_free_context(&encoder);
        avformat_free_context(format);
        av_buffer_unref(&frames_ref);
        av_buffer_unref(&device_ref);
        std::cout << "VAAPI_HIP_ENCODE=PASS frames=" << frame_count << " output=" << output_path << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "VAAPI_HIP_ENCODE=FAIL: " << error.what() << "\n";
        if (encoder) {
            avcodec_free_context(&encoder);
        }
        if (format) {
            if (format->pb) {
                avio_closep(&format->pb);
            }
            avformat_free_context(format);
        }
        av_buffer_unref(&frames_ref);
        av_buffer_unref(&device_ref);
        return 1;
    }
}
