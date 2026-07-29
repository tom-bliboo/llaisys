#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace {
template <typename T>
float attention_dot(const T *query, const T *key, size_t head_dim) {
    float sum = 0.0f;
    for (size_t column = 0; column < head_dim; ++column) {
        sum += llaisys::utils::cast<float>(query[column])
            * llaisys::utils::cast<float>(key[column]);
    }
    return sum;
}

template <typename T>
void self_attention_(T *attn_val,
                     const T *q,
                     const T *k,
                     const T *v,
                     size_t query_length,
                     size_t kv_length,
                     size_t query_heads,
                     size_t kv_heads,
                     size_t head_dim,
                     size_t value_dim,
                     float scale) {
    const size_t heads_per_kv = query_heads / kv_heads;
    std::vector<float> scores(kv_length);

    for (size_t query_pos = 0; query_pos < query_length; ++query_pos) {
        const size_t attended_keys = kv_length - query_length + query_pos + 1;
        for (size_t query_head = 0; query_head < query_heads; ++query_head) {
            const size_t kv_head = query_head / heads_per_kv;
            const T *query = q + (query_pos * query_heads + query_head) * head_dim;

            float max_score = std::numeric_limits<float>::lowest();
            for (size_t key_pos = 0; key_pos < attended_keys; ++key_pos) {
                const T *key = k + (key_pos * kv_heads + kv_head) * head_dim;
                scores[key_pos] = attention_dot(query, key, head_dim) * scale;
                max_score = std::max(max_score, scores[key_pos]);
            }

            float denominator = 0.0f;
            for (size_t key_pos = 0; key_pos < attended_keys; ++key_pos) {
                scores[key_pos] = std::exp(scores[key_pos] - max_score);
                denominator += scores[key_pos];
            }

            T *output = attn_val + (query_pos * query_heads + query_head) * value_dim;
            for (size_t column = 0; column < value_dim; ++column) {
                float value = 0.0f;
                for (size_t key_pos = 0; key_pos < attended_keys; ++key_pos) {
                    const T *value_row = v + (key_pos * kv_heads + kv_head) * value_dim;
                    value += scores[key_pos] / denominator
                        * llaisys::utils::cast<float>(value_row[column]);
                }
                output[column] = llaisys::utils::cast<T>(value);
            }
        }
    }
}
} // namespace

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val,
                    const std::byte *q,
                    const std::byte *k,
                    const std::byte *v,
                    llaisysDataType_t type,
                    size_t query_length,
                    size_t kv_length,
                    size_t query_heads,
                    size_t kv_heads,
                    size_t head_dim,
                    size_t value_dim,
                    float scale) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                               reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v),
                               query_length,
                               kv_length,
                               query_heads,
                               kv_heads,
                               head_dim,
                               value_dim,
                               scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<fp16_t *>(attn_val),
                               reinterpret_cast<const fp16_t *>(q),
                               reinterpret_cast<const fp16_t *>(k),
                               reinterpret_cast<const fp16_t *>(v),
                               query_length,
                               kv_length,
                               query_heads,
                               kv_heads,
                               head_dim,
                               value_dim,
                               scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<bf16_t *>(attn_val),
                               reinterpret_cast<const bf16_t *>(q),
                               reinterpret_cast<const bf16_t *>(k),
                               reinterpret_cast<const bf16_t *>(v),
                               query_length,
                               kv_length,
                               query_heads,
                               kv_heads,
                               head_dim,
                               value_dim,
                               scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
