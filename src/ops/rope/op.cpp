#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_ARGUMENT(out->ndim() == 3 && in->ndim() == 3, "RoPE input and output must be 3D tensors");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->shape()[0] == in->shape()[0],
                   "RoPE position IDs must contain one value per sequence position");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE position IDs must use int64");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "RoPE head dimension must be even");
    CHECK_ARGUMENT(theta > 0.0f, "RoPE theta must be positive");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "RoPE: all tensors must be contiguous.");

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         in->shape()[0],
                         in->shape()[1],
                         in->shape()[2],
                         theta);
    }

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(),
                         in->data(),
                         reinterpret_cast<const int64_t *>(pos_ids->data()),
                         out->dtype(),
                         in->shape()[0],
                         in->shape()[1],
                         in->shape()[2],
                         theta);
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
