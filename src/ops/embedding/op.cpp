#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_ARGUMENT(index->ndim() == 1, "Embedding index must be a 1D tensor");
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding index must use int64");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2, "Embedding weight and output must be 2D tensors");
    CHECK_ARGUMENT(out->shape()[0] == index->shape()[0], "Embedding output row count must match index length");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "Embedding output width must match weight width");
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "Embedding: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              index->numel(),
                              weight->shape()[0],
                              weight->shape()[1]);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(),
                              reinterpret_cast<const int64_t *>(index->data()),
                              weight->data(),
                              out->dtype(),
                              index->numel(),
                              weight->shape()[0],
                              weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
