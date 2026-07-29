#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace {
llaisys::fp16_t float_to_half_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000);
    const uint32_t exponent_bits = (bits >> 23) & 0xff;
    const uint32_t mantissa = bits & 0x7fffff;

    if (exponent_bits == 0xff) {
        const uint16_t payload = mantissa == 0 ? 0 : static_cast<uint16_t>((mantissa >> 13) | 0x0200);
        return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7c00 | payload)};
    }

    const int32_t exponent = static_cast<int32_t>(exponent_bits) - 127 + 15;
    if (exponent >= 31) {
        return llaisys::fp16_t{static_cast<uint16_t>(sign | 0x7c00)};
    }
    if (exponent <= 0) {
        if (exponent < -10) {
            return llaisys::fp16_t{sign};
        }
        const uint32_t normalized_mantissa = mantissa | 0x800000;
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t half_mantissa = normalized_mantissa >> shift;
        const uint32_t remainder = normalized_mantissa & ((uint32_t{1} << shift) - 1);
        const uint32_t halfway = uint32_t{1} << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1))) {
            ++half_mantissa;
        }
        return llaisys::fp16_t{static_cast<uint16_t>(sign | half_mantissa)};
    }

    uint16_t half = static_cast<uint16_t>(sign | (static_cast<uint16_t>(exponent) << 10) | (mantissa >> 13));
    const uint32_t remainder = mantissa & 0x1fff;
    if (remainder > 0x1000 || (remainder == 0x1000 && (half & 1))) {
        ++half;
    }
    return llaisys::fp16_t{half};
}

template <typename T>
T round_to(float value) {
    if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        return float_to_half_rne(value);
    } else {
        return llaisys::utils::cast<T>(value);
    }
}

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t width, float eps) {
    for (size_t row = 0; row < rows; ++row) {
        const T *input_row = in + row * width;
        T *output_row = out + row * width;
        float square_sum = 0.0f;
        for (size_t column = 0; column < width; ++column) {
            const float value = llaisys::utils::cast<float>(input_row[column]);
            square_sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(square_sum / static_cast<float>(width) + eps);
        for (size_t column = 0; column < width; ++column) {
            const float value = llaisys::utils::cast<float>(input_row[column]);
            const float weight_value = llaisys::utils::cast<float>(weight[column]);
            output_row[column] = round_to<T>(value * scale * weight_value);
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out,
              const std::byte *in,
              const std::byte *weight,
              llaisysDataType_t type,
              size_t rows,
              size_t width,
              float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                         reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight),
                         rows,
                         width,
                         eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<fp16_t *>(out),
                         reinterpret_cast<const fp16_t *>(in),
                         reinterpret_cast<const fp16_t *>(weight),
                         rows,
                         width,
                         eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<bf16_t *>(out),
                         reinterpret_cast<const bf16_t *>(in),
                         reinterpret_cast<const bf16_t *>(weight),
                         rows,
                         width,
                         eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
