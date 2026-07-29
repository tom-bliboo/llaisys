#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace {
template <typename T>
void embedding_(T *out,
                const int64_t *index,
                const T *weight,
                size_t index_count,
                size_t vocabulary_size,
                size_t embedding_dim) {
    for (size_t i = 0; i < index_count; ++i) {
        CHECK_ARGUMENT(index[i] >= 0 && static_cast<size_t>(index[i]) < vocabulary_size,
                       "Embedding index is out of range");
        std::memcpy(out + i * embedding_dim,
                    weight + static_cast<size_t>(index[i]) * embedding_dim,
                    embedding_dim * sizeof(T));
    }
}
} // namespace

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t index_count,
               size_t vocabulary_size,
               size_t embedding_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out),
                          index,
                          reinterpret_cast<const float *>(weight),
                          index_count,
                          vocabulary_size,
                          embedding_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<fp16_t *>(out),
                          index,
                          reinterpret_cast<const fp16_t *>(weight),
                          index_count,
                          vocabulary_size,
                          embedding_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<bf16_t *>(out),
                          index,
                          reinterpret_cast<const bf16_t *>(weight),
                          index_count,
                          vocabulary_size,
                          embedding_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
