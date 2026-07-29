#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <vector>

namespace {
template <typename T>
void rope_(T *out,
           const T *in,
           const int64_t *pos_ids,
           size_t sequence_length,
           size_t head_count,
           size_t head_dim,
           float theta) {
    const size_t half_dim = head_dim / 2;
    std::vector<float> frequency_divisor(half_dim);
    for (size_t column = 0; column < half_dim; ++column) {
        const float exponent = 2.0f * static_cast<float>(column) / static_cast<float>(head_dim);
        frequency_divisor[column] = std::pow(theta, exponent);
    }

    std::vector<float> sin_table(sequence_length * half_dim);
    std::vector<float> cos_table(sequence_length * half_dim);
    for (size_t sequence = 0; sequence < sequence_length; ++sequence) {
        for (size_t column = 0; column < half_dim; ++column) {
            const size_t table_index = sequence * half_dim + column;
            const float angle = static_cast<float>(pos_ids[sequence]) / frequency_divisor[column];
            sin_table[table_index] = std::sin(angle);
            cos_table[table_index] = std::cos(angle);
        }
    }

    for (size_t sequence = 0; sequence < sequence_length; ++sequence) {
        for (size_t head = 0; head < head_count; ++head) {
            const size_t vector_offset = (sequence * head_count + head) * head_dim;
            for (size_t column = 0; column < half_dim; ++column) {
                const size_t table_index = sequence * half_dim + column;
                const float a = llaisys::utils::cast<float>(in[vector_offset + column]);
                const float b = llaisys::utils::cast<float>(in[vector_offset + half_dim + column]);
                const float sine = sin_table[table_index];
                const float cosine = cos_table[table_index];
                out[vector_offset + column] = llaisys::utils::cast<T>(a * cosine - b * sine);
                out[vector_offset + half_dim + column] = llaisys::utils::cast<T>(b * cosine + a * sine);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t type,
          size_t sequence_length,
          size_t head_count,
          size_t head_dim,
          float theta) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                     reinterpret_cast<const float *>(in),
                     pos_ids,
                     sequence_length,
                     head_count,
                     head_dim,
                     theta);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<fp16_t *>(out),
                     reinterpret_cast<const fp16_t *>(in),
                     pos_ids,
                     sequence_length,
                     head_count,
                     head_dim,
                     theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<bf16_t *>(out),
                     reinterpret_cast<const bf16_t *>(in),
                     pos_ids,
                     sequence_length,
                     head_count,
                     head_dim,
                     theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
