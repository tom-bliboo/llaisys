#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2, "RMSNorm input and output must be 2D tensors");
    CHECK_ARGUMENT(weight->ndim() == 1, "RMSNorm weight must be a 1D tensor");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_ARGUMENT(weight->shape()[0] == in->shape()[1], "RMSNorm weight width must match input width");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(eps >= 0.0f, "RMSNorm epsilon must be non-negative");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "RMSNorm: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             in->shape()[0],
                             in->shape()[1],
                             eps);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(),
                             in->data(),
                             weight->data(),
                             out->dtype(),
                             in->shape()[0],
                             in->shape()[1],
                             eps);
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
