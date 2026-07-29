#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void embedding(std::byte *out,
               const int64_t *index,
               const std::byte *weight,
               llaisysDataType_t type,
               size_t index_count,
               size_t vocabulary_size,
               size_t embedding_dim);
}
