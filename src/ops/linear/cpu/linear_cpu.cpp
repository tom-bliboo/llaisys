#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <thread>
#include <vector>

namespace {
template <typename T>
std::vector<float> to_float(const T *data, size_t size) {
    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = llaisys::utils::cast<float>(data[i]);
    }
    return result;
}

float dot_product(const float *a, const float *b, size_t size) {
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        sum0 += a[i] * b[i];
        sum1 += a[i + 1] * b[i + 1];
        sum2 += a[i + 2] * b[i + 2];
        sum3 += a[i + 3] * b[i + 3];
    }
    float sum = (sum0 + sum1) + (sum2 + sum3);
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

void linear_float(float *out,
                  const float *in,
                  const float *weight,
                  const float *bias,
                  size_t rows,
                  size_t out_features,
                  size_t in_features) {
    if (rows == 0 || out_features == 0) {
        return;
    }

    const size_t available_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
    const size_t thread_count = std::min<size_t>(rows, std::min<size_t>(available_threads, 16));
    auto worker = [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
            const float *input_row = in + row * in_features;
            float *output_row = out + row * out_features;
            for (size_t column = 0; column < out_features; ++column) {
                const float value = dot_product(input_row, weight + column * in_features, in_features);
                output_row[column] = value + (bias ? bias[column] : 0.0f);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    for (size_t thread = 0; thread < thread_count; ++thread) {
        const size_t begin = rows * thread / thread_count;
        const size_t end = rows * (thread + 1) / thread_count;
        threads.emplace_back(worker, begin, end);
    }
    for (auto &thread : threads) {
        thread.join();
    }
}

template <typename T>
void linear_(T *out,
             const T *in,
             const T *weight,
             const T *bias,
             size_t rows,
             size_t out_features,
             size_t in_features) {
    const auto input_float = to_float(in, rows * in_features);
    const auto weight_float = to_float(weight, out_features * in_features);
    const auto bias_float = bias ? to_float(bias, out_features) : std::vector<float>{};
    std::vector<float> output_float(rows * out_features);

    linear_float(output_float.data(),
                 input_float.data(),
                 weight_float.data(),
                 bias ? bias_float.data() : nullptr,
                 rows,
                 out_features,
                 in_features);
    for (size_t i = 0; i < output_float.size(); ++i) {
        out[i] = llaisys::utils::cast<T>(output_float[i]);
    }
}
} // namespace

namespace llaisys::ops::cpu {
void linear(std::byte *out,
            const std::byte *in,
            const std::byte *weight,
            const std::byte *bias,
            llaisysDataType_t type,
            size_t rows,
            size_t out_features,
            size_t in_features) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_float(reinterpret_cast<float *>(out),
                            reinterpret_cast<const float *>(in),
                            reinterpret_cast<const float *>(weight),
                            reinterpret_cast<const float *>(bias),
                            rows,
                            out_features,
                            in_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<fp16_t *>(out),
                       reinterpret_cast<const fp16_t *>(in),
                       reinterpret_cast<const fp16_t *>(weight),
                       reinterpret_cast<const fp16_t *>(bias),
                       rows,
                       out_features,
                       in_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<bf16_t *>(out),
                       reinterpret_cast<const bf16_t *>(in),
                       reinterpret_cast<const bf16_t *>(weight),
                       reinterpret_cast<const bf16_t *>(bias),
                       rows,
                       out_features,
                       in_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
