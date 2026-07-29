#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_ARGUMENT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
                   "Linear input, weight, and output must be 2D tensors");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "Linear input width must match weight width");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "Linear output row count must match input row count");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "Linear output width must match weight row count");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Linear: input, weight, and output must be contiguous.");

    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->shape()[0] == weight->shape()[0],
                       "Linear bias shape must match output width");
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "Linear: bias must be contiguous.");
    }

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           in->shape()[0],
                           weight->shape()[0],
                           in->shape()[1]);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(),
                           in->data(),
                           weight->data(),
                           bias ? bias->data() : nullptr,
                           out->dtype(),
                           in->shape()[0],
                           weight->shape()[0],
                           in->shape()[1]);
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
