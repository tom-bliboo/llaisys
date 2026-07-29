#pragma once
#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::cpu {
void rope(std::byte *out,
          const std::byte *in,
          const int64_t *pos_ids,
          llaisysDataType_t type,
          size_t sequence_length,
          size_t head_count,
          size_t head_dim,
          float theta);
}
